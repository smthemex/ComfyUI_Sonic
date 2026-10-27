"""ComfyUI nodes for Sonic (native refactor).

Chain: SONICNativeLoader -> SONICNativePreData -> SONICNativeSampler -> native VAEDecode -> (optional) SONICNativeRife
No diffusers / transformers imports anywhere.
"""
import os

import numpy as np
import torch

import folder_paths

from .sonic_core.audio import Audio2bucketModel, AudioProjModel, WhisperEncoder, log_mel_spectrogram
from .sonic_core.face_align.align import AlignImage
from .sonic_core.rife.RIFE_HDv3 import RIFEModel
from .sonic_core.sampler import sonic_sample
from .sonic_core.sonic_model import load_sonic_model

MAX_SEED = np.iinfo(np.int32).max

_models_dir = os.path.join(folder_paths.models_dir, "sonic")
if not os.path.exists(_models_dir):
    os.makedirs(_models_dir)
folder_paths.add_model_folder_path("sonic", _models_dir)

_cache = {}


def _device():
    # Mirror the original ComfyUI_Sonic wrapper: cuda -> mps -> cpu.
    # (RIFE grid_sample and the yolo face detector already fall back to CPU
    #  internally when MPS is active, so no extra branching is needed there.)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_whisper_and_adapters():
    key = "audio_front"
    if key in _cache:
        return _cache[key]
    dev = _device()
    whisper = WhisperEncoder().to(dev).eval()
    whisper.load_hf_safetensors(os.path.join(_models_dir, "whisper-tiny", "model.safetensors"))
    whisper.requires_grad_(False)

    audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024,
                                 output_dim=1024, context_tokens=32).to(dev).eval()
    sd = torch.load(os.path.join(_models_dir, "audio2token.pth"), map_location="cpu", weights_only=True)
    audio2token.load_state_dict(sd, strict=True)

    audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024,
                                     intermediate_dim=1024, output_dim=1, context_tokens=2).to(dev).eval()
    sd = torch.load(os.path.join(_models_dir, "audio2bucket.pth"), map_location="cpu", weights_only=True)
    audio2bucket.load_state_dict(sd, strict=True)

    _cache[key] = (whisper, audio2token, audio2bucket)
    return _cache[key]


def _get_face_detector():
    key = "face_det"
    if key in _cache:
        return _cache[key]
    det = AlignImage(_device(), det_path=os.path.join(_models_dir, "yoloface_v5m.pt"))
    _cache[key] = det
    return _cache[key]


def _get_rife():
    key = "rife"
    if key in _cache:
        return _cache[key]
    rife = RIFEModel(device=_device())
    # RIFEModel.load_model() appends "flownet.pkl" internally -> pass the DIRECTORY.
    rife.load_model(os.path.join(_models_dir, "RIFE"))
    _cache[key] = rife
    return _cache[key]


def encode_audio_features(waveform, sample_rate, duration):
    """waveform [1, T, C] -> (audio_feature [80, T*3000], audio_len) mirroring official preprocessing.

    `duration` is a MAX cap on the processed audio length (seconds). When `duration <= 0`,
    the full input audio is used (no truncation) and the clip length is driven by the
    actual audio duration. The generated frame count is always derived from the audio,
    so duration never pads a short clip -- it only shortens a long one.
    """
    import torchaudio.functional as AF

    wav = waveform
    if wav.ndim == 3:      # [B, T, C]
        wav = wav.mean(dim=0)
    if wav.ndim == 2:      # [T, C] -> mono
        wav = wav.mean(dim=0)
    if sample_rate != 16000:
        wav = AF.resample(wav.float(), sample_rate, 16000)
    # Only truncate when an explicit positive duration is requested. duration <= 0 means
    # "use the whole audio" (no truncation); otherwise wav[:0] would silently empty it.
    if duration and duration > 0:
        max_samples = int(duration * 16000)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]

    device = _device()
    whisper, _, _ = _get_whisper_and_adapters()

    window = 750 * 640  # 480000 samples = 30 s
    audio_prompts = []
    last_audio_prompts = []
    with torch.no_grad():
        for i in range(0, wav.shape[0], window):
            mel = log_mel_spectrogram(wav[i:i + window].to(device))
            hs, last = whisper(mel)
            audio_prompts.append(torch.stack(hs, dim=2))        # [1, 1500, 5, 384]
            last_audio_prompts.append(last.unsqueeze(-2))       # [1, 1500, 1, 384] (official unsqueeze(-2))

    audio_prompts = torch.cat(audio_prompts, dim=1)        # [1, T, 5, 384]
    last_audio_prompts = torch.cat(last_audio_prompts, dim=1)
    audio_len = wav.shape[0] // 640
    return audio_prompts, last_audio_prompts, audio_len


class SONICNativeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svd_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "sonic_unet": (folder_paths.get_filename_list("sonic"),),
                "ip_audio_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE", "CLIP_VISION")
    RETURN_NAMES = ("model", "vae", "clip_vision")
    FUNCTION = "load"
    CATEGORY = "SONIC_Native"

    def load(self, svd_checkpoint, sonic_unet, ip_audio_scale, dtype):
        # fp32 is the safe choice on Apple Silicon (MPS has no bf16 and fp16 is
        # numerically unstable there); fp16/bf16 remain for CUDA.
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        ckpt = folder_paths.get_full_path("checkpoints", svd_checkpoint)
        unet = folder_paths.get_full_path("sonic", sonic_unet)
        patcher, vae, clip_vision = load_sonic_model(ckpt, unet, ip_scale=ip_audio_scale, dtype=dt)
        return (patcher, vae, clip_vision)


class SONICNativePreData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "min_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "duration": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SONIC_NATIVE_DATA",)
    FUNCTION = "predata"
    CATEGORY = "SONIC_Native"

    def predata(self, clip_vision, vae, audio, image, min_resolution, duration):
        dev = _device()
        image0 = image[:1]  # [1,H,W,C] in 0..1

        # ---- face detect. Official uses the box ONLY for the face mask; there is NO crop.
        np_img = (image0[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # RGB
        face_det = _get_face_detector()
        _, _, bboxes = face_det(np_img[:, :, ::-1], maxface=True)  # det expects BGR
        if len(bboxes) == 0:
            raise Exception("SONIC: no face detected in the input image")
        x1, y1, ww, hh = [float(v) for v in bboxes[0]]
        x2, y2 = x1 + ww, y1 + hh
        h, w = np_img.shape[:2]

        # Out-of-domain guard: Sonic is trained on head-and-shoulders human portraits.
        face_frac = hh / float(h)
        if face_frac < 0.18:
            print("[SONIC Native] WARNING: detected face occupies only {:.0%} of the image "
                  "height. Sonic is trained on head-and-shoulders HUMAN PORTRAIT videos; "
                  "full-body, chibi or cartoon inputs are out of domain and can produce "
                  "distorted or inappropriate results. Crop the input to the "
                  "head-and-shoulders region for best quality.".format(face_frac))

        # ---- official sizing: scale = image_size / min(w,h), rounded to 64 multiples
        scale = min_resolution / min(h, w)
        new_w = max(64, round(w * scale / 64) * 64)
        new_h = max(64, round(h * scale / 64) * 64)

        # ---- face mask: area=1.1 box on the ORIGINAL image, then LANCZOS to (new_w,new_h)
        mask_img = np.zeros_like(np_img)
        aw, ah = (x2 - x1) * 1.1, (y2 - y1) * 1.1
        cx, cy = (x2 + x1) // 2, (y2 + y1) // 2
        mx1, my1 = int(max(cx - aw // 2, 0)), int(max(cy - ah // 2, 0))
        mx2, my2 = int(min(cx + aw // 2, w)), int(min(cy + ah // 2, h))
        mask_img[my1:my2, mx1:mx2] = 255
        if (new_h, new_w) != (h, w):
            from PIL import Image as _Image
            mask_img = np.array(_Image.fromarray(mask_img).resize((new_w, new_h), _Image.LANCZOS))
        # mask_img is [H,W,3] (RGB); official uses ToTensor -> [3,H,W] then [:, :1]
        face_mask = torch.from_numpy(mask_img.astype(np.float32) / 255.0).permute(2, 0, 1)[:1]  # [1,H,W]

        # ---- reference image = the WHOLE image, LANCZOS resize to 64-multiple
        # (official test_preprocess.py uses PIL Image.LANCZOS; nearest-exact downscaling
        #  a large source image aliases badly and degrades the face)
        from PIL import Image as _Image
        ref_pil = _Image.fromarray(np_img).resize((new_w, new_h), _Image.LANCZOS)
        ref01 = torch.from_numpy(np.array(ref_pil).astype(np.float32) / 255.0)  # [h,w,3] 0..1

        # ---- audio
        audio_prompts, last_audio_prompts, audio_len = encode_audio_features(
            audio["waveform"], audio["sample_rate"], duration)

        whisper, audio2token, audio2bucket = _get_whisper_and_adapters()
        step = 2
        P = audio_prompts[:, : audio_len * step]
        P = torch.cat([torch.zeros_like(P[:, :4]), P, torch.zeros_like(P[:, :6])], dim=1)     # [1,T+10,5,384]
        L = last_audio_prompts[:, : audio_len * step]
        L = torch.cat([torch.zeros_like(L[:, :24]), L, torch.zeros_like(L[:, :26])], dim=1)   # [1,T+50,384]

        with torch.no_grad():
            image_embeds = clip_vision.encode_image(image0).image_embeds.float().to(dev)  # [1,1024]

            audio_tensor_list = []
            uncond_audio_tensor_list = []
            motion_buckets = []
            for i in range(audio_len // step):
                audio_clip = P[:, i * 2 * step: i * 2 * step + 10].unsqueeze(0)          # [1,1,10,5,384]
                bucket_clip = L[:, i * 2 * step: i * 2 * step + 50].unsqueeze(0)         # [1,1,50,384]

                mb = audio2bucket(bucket_clip, image_embeds) * 16 + 16                   # [1,1,2,1]
                motion_buckets.append(mb[0].view(-1).float())                            # [2]

                cond = audio2token(audio_clip).squeeze(0)[0]                             # [32,1024]
                uncond = audio2token(torch.zeros_like(audio_clip)).squeeze(0)[0]
                audio_tensor_list.append(cond.float())
                uncond_audio_tensor_list.append(uncond.float())

        num_frames = len(audio_tensor_list)
        if num_frames < 1:
            raise Exception("SONIC: audio too short")

        # ---- conditioning latent (raw). Official: vae.encode(whole image resized to 64-multiple).
        # pipeline has no such branch; the reference latent is always the whole image.
        enc_img = ref01.unsqueeze(0)
        with torch.no_grad():
            ref_latent = vae.encode(enc_img.to(device=dev, dtype=torch.float32))  # [1,4,h,w] raw

        data = {
            "ref_latent": ref_latent.float().cpu(),
            "image_embeds": image_embeds.cpu(),
            "audio_tensor_list": [a.cpu() for a in audio_tensor_list],
            "uncond_audio_tensor_list": [a.cpu() for a in uncond_audio_tensor_list],
            "motion_buckets": [m.cpu() for m in motion_buckets],
            "face_mask": face_mask.unsqueeze(0).float().cpu(),   # [1,1,H,W]
            "num_frames": num_frames,
            "ref_size": (new_h, new_w),
        }
        return (data,)


class SONICNativeSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sonic_data": ("SONIC_NATIVE_DATA",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "dynamic_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 12.5, "min": 5.0, "max": 120.0, "step": 0.5}),
                "appearance_guidance": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "audio_guidance": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.5}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Official SonicPipeline is called with frames_per_batch=config.n_sample_frames
                # (25). Smaller values split the clip, and the trailing window wraps around the
                # frame axis, mixing the last frames with frame 0..k in one temporal attention
                # pass -> warped first/last frames. Exposed so the behaviour can be A/B'd.
                "frames_per_batch": ("INT", {"default": 25, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("latent", "fps")
    FUNCTION = "sample"
    CATEGORY = "SONIC_Native"

    def sample(self, model, sonic_data, seed, steps, dynamic_scale, fps, appearance_guidance,
               audio_guidance, strength, frames_per_batch=None):
        num_frames = sonic_data["num_frames"]
        new_h, new_w = sonic_data["ref_size"]
        dev = _device()

        # Official/wrapper noise RNG: global torch.manual_seed(seed) + randn ON CUDA in fp16
        # (diffusers randn_tensor(generator=None)). comfy prepare_noise uses a CPU generator
        # in fp32 -> a completely different noise stream for the same seed. On marginal
        # inputs that stream decides whether the reference style holds, so replicate the
        # official stream exactly: first draw = the pipeline's noise_aug consumption
        # (shape [1,3,H,W], value unused because noise_aug_strength=0.0), second draw = latent noise.
        torch.manual_seed(seed)
        nptype = torch.float16 if dev.type == "cuda" else torch.float32
        _ = torch.randn((1, 3, new_h, new_w), dtype=nptype, device=dev)
        noise = torch.randn((1, num_frames, 4, new_h // 8, new_w // 8), dtype=nptype, device=dev)

        latents_raw, _ = sonic_sample(
            model,
            ref_latent=sonic_data["ref_latent"],
            image_embeds=sonic_data["image_embeds"],
            audio_tensor_list=sonic_data["audio_tensor_list"],
            uncond_audio_tensor_list=sonic_data["uncond_audio_tensor_list"],
            motion_buckets=sonic_data["motion_buckets"],
            face_mask=sonic_data["face_mask"],
            noise=noise,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=steps,
            min_guidance_scale1=appearance_guidance,
            max_guidance_scale1=appearance_guidance,
            min_guidance_scale2=audio_guidance,
            max_guidance_scale2=audio_guidance,
            motion_bucket_scale=dynamic_scale,
            i2i_noise_strength=strength,
            frames_per_batch=frames_per_batch,
        )
        latent_out = latents_raw.flatten(0, 1)  # [F,4,h,w] raw -> native VAEDecode
        return ({"samples": latent_out.cpu()}, fps)


class SONICNativeRife:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 12.5, "min": 1.0, "max": 120.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "interpolate"
    CATEGORY = "SONIC_Native"

    def interpolate(self, images, fps):
        rife = _get_rife()
        frames = images.to(_device()).movedim(-1, 1)  # [F,C,H,W]
        results = []
        n = frames.shape[0]
        with torch.no_grad():
            for idx in range(n - 1):
                I1 = frames[idx: idx + 1]
                I2 = frames[idx + 1: idx + 2]
                middle = rife.inference(I1, I2).clamp(0, 1).detach()  # [1,C,H,W]
                # keep everything [1,C,H,W]: frames[idx] would drop the batch dim
                results.append(frames[idx: idx + 1])
                results.append(middle)
            results.append(frames[n - 1: n])
        out = torch.cat(results, dim=0).movedim(1, -1).cpu()  # [2F,C,H,W] -> [2F,H,W,C]
        return (out, fps * 2.0)


NODE_CLASS_MAPPINGS = {
    "SONICNativeLoader": SONICNativeLoader,
    "SONICNativePreData": SONICNativePreData,
    "SONICNativeSampler": SONICNativeSampler,
    "SONICNativeRife": SONICNativeRife,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SONICNativeLoader": "SONIC Native Loader",
    "SONICNativePreData": "SONIC Native PreData",
    "SONICNativeSampler": "SONIC Native Sampler",
    "SONICNativeRife": "SONIC Native RIFE (interpolate x2)",
}
