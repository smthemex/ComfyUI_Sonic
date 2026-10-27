"""SonicModel: comfy BaseModel with the Sonic UNet + Sonic-specific extra conds."""
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.conds
import comfy.utils
import comfy.supported_models
import comfy.model_detection
import torch

from .unet import SonicUNet


class SonicModel(comfy.model_base.BaseModel):
    """BaseModel wired to SonicUNet.

    - model_config comes from native SVD checkpoint detection (gives correct
      V_PREDICTION_EDM-compatible sampling_settings sigma_min/sigma_max, SD15 latent
      format, manual_cast dtype handling).
    - extra conds: added_time_ids ('y'), concat_latent_image ('c_concat'),
      sonic_clip, sonic_audio, sonic_mask (all passed straight into the unet by
      BaseModel._apply_model).
    """

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.V_PREDICTION_EDM, device=None,
                 ip_scale=1.0):
        import functools

        super().__init__(model_config, model_type, device=device,
                         unet_model=functools.partial(SonicUNet, ip_scale=ip_scale))

    def extra_conds(self, **kwargs):
        out = {}
        added_time_ids = kwargs.get("added_time_ids", None)
        if added_time_ids is not None:
            out['y'] = comfy.conds.CONDRegular(added_time_ids)

        latent_image = kwargs.get("concat_latent_image", None)
        if latent_image is not None:
            out['c_concat'] = comfy.conds.CONDNoiseShape(latent_image)

        clip = kwargs.get("sonic_clip", None)
        if clip is not None:
            out['sonic_clip'] = comfy.conds.CONDRegular(clip)

        audio = kwargs.get("sonic_audio", None)
        if audio is not None:
            out['sonic_audio'] = comfy.conds.CONDRegular(audio)

        mask = kwargs.get("sonic_mask", None)
        if mask is not None:
            out['sonic_mask'] = comfy.conds.CONDRegular(mask)
        return out


def load_sonic_model(ckpt_path, sonic_unet_path, ip_scale=1.0, dtype=torch.float16, device=None):
    """Build SonicModel + ModelPatcher from the native SVD checkpoint and sonic unet.pth."""
    if device is None:
        device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    sd = comfy.utils.load_torch_file(ckpt_path)
    if not any(k.startswith("model.diffusion_model.") for k in sd.keys()):
        raise ValueError("Not an SVD checkpoint (no model.diffusion_model prefix): {}".format(ckpt_path))
    unet_sd = comfy.utils.state_dict_prefix_replace(sd, {"model.diffusion_model.": ""}, filter_keys=True)

    model_config = comfy.model_detection.model_config_from_unet(unet_sd, "")
    if model_config is None:
        unet_config = comfy.model_detection.detect_unet_config(unet_sd, "")
        model_config = comfy.supported_models.SVD_img2vid(unet_config)
    del unet_sd

    model = SonicModel(model_config, model_type=comfy.model_base.ModelType.V_PREDICTION_EDM,
                       device=offload_device, ip_scale=ip_scale)

    sonic_sd = torch.load(sonic_unet_path, map_location="cpu", weights_only=True)
    sonic_sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sonic_sd.items()}
    missing, unexpected = model.diffusion_model.load_state_dict(sonic_sd, strict=True)
    del sonic_sd
    model.diffusion_model.to(dtype)
    model.diffusion_model.eval()
    model.diffusion_model.requires_grad_(False)

    patcher = comfy.model_patcher.ModelPatcher(model, load_device=device, offload_device=offload_device)

    # VAE (raw latents, native comfy VAE)
    vae_sd = comfy.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
    vae = comfy.sd.VAE(sd=vae_sd)

    # CLIP vision from the SVD conditioner
    clip_vision = comfy.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    del sd
    return patcher, vae, clip_vision
