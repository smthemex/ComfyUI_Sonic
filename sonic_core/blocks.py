"""Resnet/down/up blocks + SVD spatio-temporal UNet blocks (port of diffusers 0.24)."""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .attention import TransformerSpatioTemporalModel


class Upsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv",
                 operations=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = operations.Conv2d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if output_size is None:
            hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = torch.nn.functional.interpolate(hidden_states, size=output_size, mode="nearest")

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        return hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv", operations=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = operations.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = torch.nn.functional.pad(hidden_states, pad, mode="constant", value=0)

        return self.conv(hidden_states)


class ResnetBlock2D(nn.Module):
    """diffusers ResnetBlock2D (time_embedding_norm="default", swish)."""

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        temb_channels=512,
        groups=32,
        groups_out=None,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        operations=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = operations.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = operations.Linear(temb_channels, out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm}")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(0.0)
        self.conv2 = operations.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = operations.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return (input_tensor + hidden_states) / self.output_scale_factor


class TemporalResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512, eps=1e-6, operations=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = operations.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        if temb_channels is not None:
            self.time_emb_proj = operations.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(0.0)
        self.conv2 = operations.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = operations.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return input_tensor + hidden_states


class SpatioTemporalResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        temb_channels=512,
        eps=1e-6,
        temporal_eps=None,
        merge_factor=0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix=False,
        operations=None,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
            operations=operations,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
            operations=operations,
        )

        self.time_mixer = _AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(self, hidden_states, temb=None, image_only_indicator=None):
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states


class _AlphaBlender(nn.Module):
    """Local copy so blocks.py does not depend on attention.py import order."""

    def __init__(self, alpha, merge_strategy="learned_with_images", switch_spatial_to_temporal_mix=False):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix
        self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))

    def forward(self, x_spatial, x_temporal, image_only_indicator=None):
        if image_only_indicator is None:
            raise ValueError("image_only_indicator required")
        ndims = 5
        alpha = torch.where(
            image_only_indicator.bool(),
            torch.ones(1, 1, device=image_only_indicator.device),
            torch.sigmoid(self.mix_factor)[..., None],
        )
        alpha = alpha[:, None, :, None, None].to(x_spatial.dtype)
        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha
        return alpha * x_spatial + (1.0 - alpha) * x_temporal


class UNetMidBlockSpatioTemporal(nn.Module):
    def __init__(self, in_channels, temb_channels, num_layers=1, transformer_layers_per_block=1,
                 num_attention_heads=1, cross_attention_dim=1280, ip_scale=1.0, operations=None):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnets = [
            SpatioTemporalResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=1e-5,
                operations=operations,
            )
        ]
        attentions = []

        for i in range(num_layers):
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    ip_scale=ip_scale,
                    operations=operations,
                )
            )
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                    operations=operations,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, image_only_indicator=None,
                ip_mask=None):
        hidden_states = self.resnets[0](hidden_states, temb, image_only_indicator=image_only_indicator)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                ip_mask=ip_mask,
            )
            hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
        return hidden_states


class DownBlockSpatioTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=1, add_downsample=True, operations=None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                    operations=operations,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, use_conv=True, out_channels=out_channels, name="op", operations=operations)]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, image_only_indicator=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockSpatioTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=1, transformer_layers_per_block=1,
                 num_attention_heads=1, cross_attention_dim=1280, add_downsample=True, ip_scale=1.0,
                 operations=None):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            # NOTE: eps=1e-6 here is exactly what the Sonic / diffusers code hardcodes.
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                    operations=operations,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    ip_scale=ip_scale,
                    operations=operations,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=1, name="op",
                              operations=operations)]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, image_only_indicator=None,
                ip_mask=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                ip_mask=ip_mask,
            )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UpBlockSpatioTemporal(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, temb_channels, resolution_idx=None,
                 num_layers=1, resnet_eps=1e-6, add_upsample=True, operations=None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    operations=operations,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels, operations=operations)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, image_only_indicator=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class CrossAttnUpBlockSpatioTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, resolution_idx=None,
                 num_layers=1, transformer_layers_per_block=1, resnet_eps=1e-6, num_attention_heads=1,
                 cross_attention_dim=1280, add_upsample=True, ip_scale=1.0, operations=None):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    operations=operations,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    ip_scale=ip_scale,
                    operations=operations,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels, operations=operations)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None,
                image_only_indicator=None, ip_mask=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                ip_mask=ip_mask,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states
