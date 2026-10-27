"""Sonic sliding-window sampling loop, driven by comfy native apply_model.

Faithful port of SonicPipeline.__call__ (any-frame long video):
- karras sigma table identical to diffusers EulerDiscreteScheduler
  (sigma_min 0.002 / sigma_max 700 / rho 7 / final sigma 0), timestep = 0.25*ln(sigma)
  which is exactly comfy ModelSamplingContinuousEDM -> native apply_model works as-is.
- 3-branch CFG: uncond / drop_audio / cond with
  pred = u + s1*(da-u) + s2*(c-da)
- per-timestep window sweep with modulo wrap, accumulating shift_offset per step.
"""
import numpy as np
import torch


def karras_sigmas(num_inference_steps, sigma_min=0.002, sigma_max=700.0, rho=7.0, final_sigma=0.0):
    """diffusers EulerDiscreteScheduler._convert_to_karras + final append."""
    ramp = np.linspace(0, 1, num_inference_steps, dtype=np.float64)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = np.concatenate([sigmas, [final_sigma]]).astype(np.float32)
    return torch.from_numpy(sigmas)


def get_timesteps_for_strength(num_inference_steps, strength):
    """diffusers get_timesteps: cut the schedule tail for img2img strength."""
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    return t_start  # index into the base sigma table


@torch.no_grad()
def sonic_sample(
    model_patcher,
    ref_latent,          # [1, 4, h, w] RAW comfy VAE latent of the conditioning image
    image_embeds,        # [1, 1, 1024] clip vision image embeds
    audio_tensor_list,   # list of [32, 1024], len == num_frames
    uncond_audio_tensor_list,
    motion_buckets,      # list/tensor [num_frames, 2]
    face_mask,           # [1, 1, H, W] in 0..1
    noise,               # [1, F, 4, h, w]
    num_frames,
    fps=12.5,
    num_inference_steps=25,
    min_guidance_scale1=2.0,
    max_guidance_scale1=2.0,
    min_guidance_scale2=7.5,
    max_guidance_scale2=7.5,
    motion_bucket_scale=1.0,
    overlap=0,
    shift_offset=7,
    frames_per_batch=None,
    i2i_noise_strength=1.0,
    scaling_factor=0.18215,
):
    import comfy.model_management

    base = model_patcher.model
    device = model_patcher.load_device

    # --- ComfyUI memory management integration ---------------------------------
    # The PreData node loaded the VAE + CLIP_VISION onto the GPU. If we just call
    # load_models_gpu([unet]) with default args, ComfyUI may leave the UNet *staged*
    # (partial) loaded on NORMAL_VRAM when VRAM is tight -> every forward pass
    # re-streams weights host->device -> slow.
    #
    # Passing force_full_load=True skips the lowvram partial-load branch so the whole
    # ~3.5GB UNet is resident on the GPU. load_models_gpu() itself calls free_memory(),
    # which evicts every other model (VAE/CLIP) not in keep_loaded to make room -- the
    # unet patcher is kept (and stays resident across runs, like the original wrapper's
    # global pipe), so repeated runs don't pay a reload cost.
    comfy.model_management.load_models_gpu([model_patcher], force_full_load=True)

    if frames_per_batch is None:
        # Official SonicPipeline is called with frames_per_batch=config.n_sample_frames (25),
        # NOT int(fps).  Using a small window splits the clip, and the last window wraps around
        # (`idx % num_frames`), mixing the final frames with frame 0..k in one temporal
        # attention pass -- that is what produces warped first/last frames.
        frames_per_batch = 25
    dtype = base.get_dtype_inference()

    # c_concat (reference conditioning) = RAW comfy-VAE latent, NOT scaled by 0.18215.
    # This matches the working ComfyUI_Sonic wrapper pipeline exactly:
    #   image_latents = cat([zeros, img_latent, img_latent])          (raw)
    #   ref init      = img_latent * 0.18215                          (scaled, add_noise only)
    # The upstream jixiaozhong/Sonic repo pipeline scales c_concat by 0.18215, which weakens
    # the reference 5.5x; on marginal inputs (anime / chibi) the model then collapses to its
    # training domain (photoreal humans) from frame 0.
    ref_raw = ref_latent.to(device, torch.float32)                                       # [1,4,h,w]
    ref_scaled = (ref_raw * scaling_factor).unsqueeze(1)                                 # init only

    sigmas_all = karras_sigmas(num_inference_steps)
    t_start = get_timesteps_for_strength(num_inference_steps, i2i_noise_strength)
    sigmas = sigmas_all[t_start:].to(device)
    steps = len(sigmas) - 1

    sigma_start = sigmas[0]
    latents_all = (ref_scaled + sigma_start * noise.to(device, torch.float32)).to(dtype)

    # ---- build static conditioning tensors (batch dim 3 = uncond / drop_audio / cond)
    emb = image_embeds.to(device, dtype).unsqueeze(1).repeat(1, num_frames, 1, 1)     # [1,F,1,1024]
    image_cond = torch.cat([torch.zeros_like(emb), emb, emb], dim=0)                  # [3,F,1,1024]

    audio_cond_frame = torch.stack(audio_tensor_list, dim=0).to(device, dtype).unsqueeze(0)      # [1,F,32,1024]
    uncond_audio_frame = torch.stack(uncond_audio_tensor_list, dim=0).to(device, dtype).unsqueeze(0)
    audio_cond = torch.cat([uncond_audio_frame, uncond_audio_frame, audio_cond_frame], dim=0)    # [3,F,32,1024]

    image_latents = ref_raw.clone().unsqueeze(1)                                              # [1,1,4,h,w] RAW
    image_latents = torch.cat([torch.zeros_like(image_latents), image_latents, image_latents], dim=0)
    image_latents = image_latents.repeat(1, num_frames, 1, 1, 1)                              # [3,F,4,h,w]

    motion_buckets = torch.stack([torch.as_tensor(m, device=device, dtype=torch.float32).view(-1)
                                  for m in motion_buckets], dim=0)                                 # [F,2]
    motion_buckets = motion_buckets * motion_bucket_scale

    face_mask = face_mask.to(device=device, dtype=dtype)
    if face_mask.ndim == 3:
        face_mask = face_mask.unsqueeze(0)
    face_mask = face_mask[:, :1]

    guidance_scale1 = torch.linspace(min_guidance_scale1, max_guidance_scale1, num_inference_steps).to(device)
    guidance_scale2 = torch.linspace(min_guidance_scale2, max_guidance_scale2, num_inference_steps).to(device)

    latent_h, latent_w = ref_scaled.shape[-2:]
    shift = 0

    pbar = comfy.utils.ProgressBar(num_inference_steps)
    for i in range(steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        pred_latents = torch.zeros_like(latents_all, dtype=torch.float32)
        counter = torch.zeros((1, num_frames, 1, 1, 1), dtype=torch.float32, device=latents_all.device)

        for index_start in range(0, num_frames, frames_per_batch - overlap):
            index_start = index_start - shift

            idx_list = [(index_start + j) % num_frames for j in range(frames_per_batch)]

            def indice_slice(tensor, idxs):
                return torch.stack([tensor[:, j] for j in idxs], dim=1)

            lat = indice_slice(latents_all, idx_list)                    # [1,fpb,4,h,w]
            c_concat = indice_slice(image_latents, idx_list)             # [3,fpb,4,h,w]
            batch_clip = indice_slice(image_cond, idx_list)              # [3,fpb,1,1024]
            batch_audio = indice_slice(audio_cond, idx_list)             # [3,fpb,32,1024]

            mb = indice_slice(motion_buckets.unsqueeze(0), idx_list)     # [1,fpb,2]
            mb = mb.mean(dim=1).squeeze(0)                               # [2]

            added_time_ids = torch.tensor(
                [[float(fps), float(mb[0]), float(mb[1])]], dtype=torch.float32
            ).to(device).repeat(3, 1)                                    # [3,3]

            x_in = torch.cat([lat] * 3, dim=0).flatten(0, 1).to(dtype)   # [3*fpb,4,h,w]
            conds = dict(
                c_concat=c_concat.flatten(0, 1),
                added_time_ids=added_time_ids,
                sonic_clip=batch_clip,
                sonic_audio=batch_audio,
                sonic_mask=face_mask,
            )

            denoised = base.apply_model(x_in, sigma, transformer_options={}, **conds)
            denoised = denoised.view(3, frames_per_batch, 4, latent_h, latent_w).float()

            d_uncond, d_drop_audio, d_cond = denoised[0], denoised[1], denoised[2]
            s1 = guidance_scale1[i]
            s2 = guidance_scale2[i]
            denoised_comb = d_uncond + s1 * (d_drop_audio - d_uncond) + s2 * (d_cond - d_drop_audio)

            lat = lat.float() + (sigma_next - sigma) * (lat.float() - denoised_comb) / sigma
            for j in range(frames_per_batch):
                pred_latents[:, idx_list[j]] += lat[:, j]
                counter[:, idx_list[j]] += 1

        shift += shift_offset
        latents_all = (pred_latents / counter).to(dtype)
        pbar.update(1)

    # scaled latent space -> raw comfy latent space for native VAEDecode
    latents_raw = latents_all.float() / scaling_factor
    return latents_raw, sigmas
