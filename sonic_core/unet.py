"""Sonic UNet: SVD UNetSpatioTemporalConditionModel + audio IP-Adapter, diffusers-named.

Weights load 1:1 from F:/test/ComfyUI/models/sonic/unet.pth via strict=True.
Called from comfy BaseModel._apply_model as:
    diffusion_model(xc, t, context=..., control=..., transformer_options=..., **extra_conds)
where xc = cat([scaled_latent, concat_latent], dim=1) flattened to [B*F, 8, H, W].
"""
from torch import nn

import torch

from .attention import TransformerSpatioTemporalModel
from .blocks import (
    CrossAttnDownBlockSpatioTemporal,
    CrossAttnUpBlockSpatioTemporal,
    DownBlockSpatioTemporal,
    UNetMidBlockSpatioTemporal,
    UpBlockSpatioTemporal,
)
from .embeddings import TimestepEmbedding, Timesteps


class SonicUNet(nn.Module):
    def __init__(self, operations=None, device=None, ip_scale=1.0, **kwargs):
        # Accept (and ignore) comfy openaimodel unet_config kwargs.
        super().__init__()
        if operations is None:
            import torch.nn as operations

        in_channels = 8
        out_channels = 4
        block_out_channels = (320, 640, 1280, 1280)
        addition_time_embed_dim = 256
        projection_class_embeddings_input_dim = 768
        layers_per_block = 2
        cross_attention_dim = 1024
        transformer_layers_per_block = 1
        num_attention_heads = (5, 10, 20, 20)

        self.sample_size = 96
        self.dtype = kwargs.get("dtype", None) or torch.float16
        self.config = {
            "cross_attention_dim": cross_attention_dim,
            "block_out_channels": list(block_out_channels),
        }

        self.conv_in = operations.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, operations=operations)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim,
                                               operations=operations)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(block_out_channels)
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(block_out_channels)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(block_out_channels)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(block_out_channels)

        blocks_time_embed_dim = time_embed_dim

        output_channel = block_out_channels[0]
        for i, _ in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block_type = "DownBlockSpatioTemporal" if i == 3 else "CrossAttnDownBlockSpatioTemporal"
            if down_block_type == "CrossAttnDownBlockSpatioTemporal":
                down_block = CrossAttnDownBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    add_downsample=not is_final_block,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    ip_scale=ip_scale,
                    operations=operations,
                )
            else:
                down_block = DownBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    num_layers=layers_per_block[i],
                    add_downsample=not is_final_block,
                    operations=operations,
                )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
            ip_scale=ip_scale,
            operations=operations,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        up_block_types = ["UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal",
                          "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlockSpatioTemporal":
                up_block = CrossAttnUpBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    add_upsample=add_upsample,
                    resnet_eps=1e-5,
                    resolution_idx=i,
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    ip_scale=ip_scale,
                    operations=operations,
                )
            else:
                up_block = UpBlockSpatioTemporal(
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    num_layers=reversed_layers_per_block[i] + 1,
                    add_upsample=add_upsample,
                    resnet_eps=1e-5,
                    resolution_idx=i,
                    operations=operations,
                )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = operations.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = operations.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample, timestep, context=None, control=None, transformer_options=None,
                added_time_ids=None, sonic_clip=None, sonic_audio=None, sonic_mask=None, **kwargs):
        import torch

        if transformer_options is None:
            transformer_options = {}

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None]

        batch_size = added_time_ids.shape[0] if added_time_ids is not None else 1
        num_frames = sample.shape[0] // batch_size

        timesteps = timestep.expand(batch_size)
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # sample already flattened to [B*F, C, H, W] by BaseModel._apply_model
        emb = emb.repeat_interleave(num_frames, dim=0)

        # accept both [B, F, T, C] and pre-flattened [B*F, T, C]
        clip_ctx = sonic_clip.flatten(0, 1) if sonic_clip.ndim == 4 else sonic_clip
        audio_ctx = sonic_audio.flatten(0, 1) if sonic_audio.ndim == 4 else sonic_audio
        encoder_hidden_states = (clip_ctx, audio_ctx)

        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    ip_mask=sonic_mask,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
            ip_mask=sonic_mask,
        )

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    ip_mask=sonic_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample  # [B*F, 4, H, W]
