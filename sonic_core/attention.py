"""Attention + transformer blocks for Sonic UNet.

Port of diffusers 0.24 attention pieces with the Sonic audio IP-Adapter built in.
All Linear/GroupNorm/LayerNorm go through comfy `operations` so weight offload and
dtype casting behave like every other comfy model.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimestepEmbedding, Timesteps


def _sdpa(query, key, value):
    return F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)


def mask_downsample(mask, batch_size, num_queries, value_embed_dim):
    """Exact port of diffusers IPAdapterMaskProcessor.downsample."""
    import math

    o_h = mask.shape[1]
    o_w = mask.shape[2]
    ratio = o_w / o_h
    mask_h = int(math.sqrt(num_queries / ratio))
    mask_h = int(mask_h) + int((num_queries % int(mask_h)) != 0)
    mask_w = num_queries // mask_h

    mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode="bicubic").squeeze(0)

    if mask_downsample.shape[0] < batch_size:
        mask_downsample = mask_downsample.repeat(batch_size, 1, 1)

    mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)

    downsampled_area = mask_h * mask_w
    if downsampled_area < num_queries:
        mask_downsample = F.pad(mask_downsample, (0, num_queries - mask_downsample.shape[1]), value=0.0)
    if downsampled_area > num_queries:
        mask_downsample = mask_downsample[:, :num_queries]

    mask_downsample = mask_downsample.view(mask_downsample.shape[0], mask_downsample.shape[1], 1).repeat(
        1, 1, value_embed_dim
    )
    return mask_downsample


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, operations):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        return F.gelu(gate)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    """diffusers-named: net.0.proj (GEGLU), net.2 (out proj)."""

    def __init__(self, dim, dim_out=None, mult=4, operations=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([])
        self.net.append(GEGLU(dim, inner_dim, operations))
        self.net.append(nn.Dropout(0.0))
        self.net.append(operations.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CrossAttention(nn.Module):
    """diffusers Attention with optional Sonic audio IP (processor.to_k_ip / to_v_ip).

    When `with_audio_ip=True` a `processor` submodule is created so that the
    checkpoint keys `attn2.processor.to_k_ip.0.weight` load 1:1 via strict load.
    """

    def __init__(self, query_dim, cross_attention_dim=None, heads=8, dim_head=64,
                 with_audio_ip=False, ip_scale=1.0, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        inner_dim = dim_head * heads
        self.heads = heads
        self.inner_dim = inner_dim
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False)
        self.to_k = operations.Linear(self.cross_attention_dim, inner_dim, bias=False)
        self.to_v = operations.Linear(self.cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([operations.Linear(inner_dim, query_dim), nn.Dropout(0.0)])

        self.scale = 1.0
        self.ip_scale = ip_scale
        if with_audio_ip:
            self.processor = nn.Module()
            self.processor.to_k_ip = nn.ModuleList(
                [operations.Linear(self.cross_attention_dim, inner_dim, bias=False)]
            )
            self.processor.to_v_ip = nn.ModuleList(
                [operations.Linear(self.cross_attention_dim, inner_dim, bias=False)]
            )

    def forward(self, hidden_states, encoder_hidden_states=None, ip_hidden_states=None, ip_mask=None):
        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        head_dim = self.inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = _sdpa(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_hidden_states is not None and hasattr(self, "processor"):
            ip_key = self.processor.to_k_ip[0](ip_hidden_states)
            ip_value = self.processor.to_v_ip[0](ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            ip_attn = _sdpa(query, ip_key, ip_value)
            ip_attn = ip_attn.transpose(1, 2).reshape(batch_size, -1, self.inner_dim).to(query.dtype)

            if ip_mask is not None:
                # official passes mask[:, i] -> [num_images, H, W]; we have a single mask
                m_in = ip_mask[:, 0] if ip_mask.ndim == 4 else ip_mask
                m = mask_downsample(m_in, batch_size, ip_attn.shape[1], ip_attn.shape[2])
                m = m.to(dtype=query.dtype, device=query.device)
                ip_attn = ip_attn * m
            hidden_states = hidden_states + self.ip_scale * ip_attn

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states + residual if getattr(self, "residual_connection", False) else hidden_states


class BasicTransformerBlock(nn.Module):
    """Spatial block: norm1/attn1, norm2/attn2(+audio ip), norm3/ff."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim=None,
                 ip_scale=1.0, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.attn1 = CrossAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim,
                                    operations=operations)
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.attn2 = CrossAttention(query_dim=dim, cross_attention_dim=cross_attention_dim,
                                    heads=num_attention_heads, dim_head=attention_head_dim,
                                    with_audio_ip=True, ip_scale=ip_scale, operations=operations)
        self.norm3 = operations.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.ff = FeedForward(dim, operations=operations)

    def forward(self, hidden_states, encoder_hidden_states=None, ip_hidden_states=None, ip_mask=None):
        norm_hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attn1(norm_hidden_states)

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states + self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                ip_hidden_states=ip_hidden_states,
                ip_mask=ip_mask,
            )

        norm_hidden_states = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(norm_hidden_states)
        return hidden_states


class TemporalBasicTransformerBlock(nn.Module):
    """diffusers 0.24 TemporalBasicTransformerBlock with Sonic audio IP on attn2."""

    def __init__(self, dim, time_mix_inner_dim, num_attention_heads, attention_head_dim,
                 cross_attention_dim=None, ip_scale=1.0, operations=None):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim

        self.norm_in = operations.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=time_mix_inner_dim, operations=operations)

        self.norm1 = operations.LayerNorm(time_mix_inner_dim)
        self.attn1 = CrossAttention(query_dim=time_mix_inner_dim, heads=num_attention_heads,
                                    dim_head=attention_head_dim, operations=operations)

        if cross_attention_dim is not None:
            self.norm2 = operations.LayerNorm(time_mix_inner_dim)
            self.attn2 = CrossAttention(query_dim=time_mix_inner_dim, cross_attention_dim=cross_attention_dim,
                                        heads=num_attention_heads, dim_head=attention_head_dim,
                                        with_audio_ip=True, ip_scale=ip_scale, operations=operations)
        else:
            self.norm2 = None
            self.attn2 = None

        self.norm3 = operations.LayerNorm(time_mix_inner_dim)
        self.ff = FeedForward(time_mix_inner_dim, operations=operations)

    def forward(self, hidden_states, num_frames, encoder_hidden_states=None, ip_hidden_states=None):
        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        # [B*F, H*W, C] -> [B*H*W, F, C]
        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                                     ip_hidden_states=ip_hidden_states)
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_frames, seq_length, channels)
        return hidden_states


class AlphaBlender(nn.Module):
    """diffusers 0.24 AlphaBlender (learned_with_images)."""

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(self, alpha, merge_strategy="learned_with_images", switch_spatial_to_temporal_mix=False):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix
        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        else:
            self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))

    def get_alpha(self, image_only_indicator, ndims):
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                torch.sigmoid(self.mix_factor)[..., None],
            )
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")
        else:
            raise NotImplementedError
        return alpha

    def forward(self, x_spatial, x_temporal, image_only_indicator=None):
        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.to(x_spatial.dtype)
        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha
        return alpha * x_spatial + (1.0 - alpha) * x_temporal


class TransformerSpatioTemporalModel(nn.Module):
    """diffusers 0.24 TransformerSpatioTemporalModel with Sonic tuple conditioning.

    encoder_hidden_states is (clip_ctx, audio_ctx):
      clip_ctx:  [B*F, 1, C]      spatial; frame-mean broadcast for temporal
      audio_ctx: [B*F, T, C]      spatial per-frame; frame-mean broadcast for temporal
    """

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, out_channels=None,
                 num_layers=1, cross_attention_dim=None, ip_scale=1.0, operations=None):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = operations.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim,
                                      cross_attention_dim=cross_attention_dim, ip_scale=ip_scale,
                                      operations=operations)
                for _ in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(inner_dim, time_mix_inner_dim, num_attention_heads,
                                              attention_head_dim, cross_attention_dim=cross_attention_dim,
                                              ip_scale=ip_scale, operations=operations)
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels, operations=operations)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        self.out_channels = in_channels if out_channels is None else out_channels
        self.proj_out = operations.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states, image_only_indicator=None, ip_mask=None):
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        clip_hidden_states, audio_hidden_states = encoder_hidden_states

        def spatial2time(context):
            context = context.reshape(batch_size, num_frames, context.shape[-2], context.shape[-1])
            context = context.mean(dim=(1,), keepdim=True)
            context = context.repeat(1, height * width, 1, 1)
            context = context.reshape(batch_size * height * width, -1, context.shape[-1])
            return context

        encoder_hidden_states_time = (spatial2time(clip_hidden_states), spatial2time(audio_hidden_states))

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            hidden_states = block(hidden_states, encoder_hidden_states=clip_hidden_states,
                                  ip_hidden_states=audio_hidden_states, ip_mask=ip_mask)

            hidden_states_mix = hidden_states + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=encoder_hidden_states_time[0],
                ip_hidden_states=encoder_hidden_states_time[1],
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        return hidden_states + residual
