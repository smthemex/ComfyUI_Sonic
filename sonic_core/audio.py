"""Pure-torch audio front-end for Sonic (no transformers / no diffusers).

- WhisperEncoder: whisper-tiny encoder (4 layers, d=384) loading HF model.safetensors 1:1.
- log_mel: whisper log-mel spectrogram (slaney/htk mel bank, n_fft=400, hop=160).
- AudioProjModel / Audio2bucketModel: Sonic's audio adapters (diffusers ModelMixin dropped).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- mel filters
def mel_filter_bank(sampling_rate=16000, n_fft=400, n_mels=80, f_min=0.0, f_max=8000.0):
    """librosa.filters.mel equivalent (slaney mel scale + slaney norm).

    Mel filterbank in Hz space, exactly as librosa does it:
      mel_f = mel_to_hz(linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels+2))  (Hz)
      lower = (fftfreqs - mel_f[i])   / fdiff[i]
      upper = (mel_f[i+2] - fftfreqs) / fdiff[i+1]
      weights = max(0, min(lower, upper)); slaney norm *= 2/(mel_f[i+2]-mel_f[i])
    """
    n_freqs = n_fft // 2 + 1
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    def hz_to_mel(freqs):
        mels = freqs / f_sp
        log_t = freqs >= min_log_hz
        mels = torch.where(log_t, min_log_mel + torch.log(freqs / min_log_hz) / logstep, mels)
        return mels

    def mel_to_hz(mels):
        freqs = f_sp * mels
        log_t = mels >= min_log_mel
        freqs = torch.where(log_t, min_log_hz * torch.exp(logstep * (mels - min_log_mel)), freqs)
        return freqs

    f_max = min(f_max, sampling_rate / 2)
    mels = torch.linspace(hz_to_mel(torch.tensor(float(f_min))).item(),
                          hz_to_mel(torch.tensor(float(f_max))).item(), n_mels + 2)
    mel_f = mel_to_hz(mels)  # Hz

    fft_freqs = torch.linspace(0, sampling_rate / 2, n_freqs)

    fdiff = mel_f[1:] - mel_f[:-1]
    ramps = mel_f.unsqueeze(1) - fft_freqs.unsqueeze(0)  # [n_mels+2, n_freqs]

    weights = torch.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = torch.clamp(torch.minimum(lower, upper), min=0.0)

    # slaney norm
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels]).unsqueeze(1)
    weights = weights * enorm
    return weights


# ---------------------------------------------------------------- whisper mel
def log_mel_spectrogram(audio, n_fft=400, hop_length=160, num_mels=80):
    """Whisper log-mel for ONE 30 s chunk.

    audio: [N] float waveform @16 kHz, N <= 480000 (zero-padded to 480000
    BEFORE the STFT, matching WhisperFeatureExtractor's "pad audio with silence"
    semantics -- verified empirically: padded tail lands on the log clamp floor,
    not on a 0.0 padded log frame).

    Returns [1, 80, 3000].
    """
    chunk_samples = 480000
    if audio.numel() < chunk_samples:
        audio = F.pad(audio, (0, chunk_samples - audio.numel()), value=0.0)
    else:
        audio = audio[:chunk_samples]

    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2  # [201, 3000]

    filters = mel_filter_bank(n_mels=num_mels, n_fft=n_fft)
    filters = filters.to(audio.device, dtype=magnitudes.dtype)
    mel_spec = filters @ magnitudes  # [80, 3000]

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.unsqueeze(0)  # [1, 80, 3000]


# ---------------------------------------------------------------- whisper mini
class WhisperAttention(nn.Module):
    def __init__(self, embed_dim, heads, operations=None):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        # checkpoint has q_proj.bias / v_proj.bias but no k_proj.bias
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = operations.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, t, c)
        return self.out_proj(o)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, ffn_dim, operations=None):
        super().__init__()
        self.self_attn = WhisperAttention(d_model, heads, operations)
        self.self_attn_layer_norm = operations.LayerNorm(d_model)
        self.fc1 = operations.Linear(d_model, ffn_dim)
        self.fc2 = operations.Linear(ffn_dim, d_model)
        self.final_layer_norm = operations.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        h = self.final_layer_norm(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + h


class WhisperEncoder(nn.Module):
    """whisper-tiny encoder; state dict loads directly from HF model.safetensors."""

    def __init__(self, n_mels=80, d_model=384, layers=4, heads=6, ffn_dim=1536, max_positions=1500,
                 operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.conv1 = operations.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = operations.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_positions, d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(d_model, heads, ffn_dim, operations) for _ in range(layers)]
        )
        self.layer_norm = operations.LayerNorm(d_model)

    def load_hf_safetensors(self, path):
        from safetensors.torch import load_file

        sd = load_file(path)
        prefix = "model.encoder."
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v.float()
        missing, unexpected = self.load_state_dict(out, strict=False)
        assert len(missing) == 0, f"whisper encoder missing: {missing}"
        return unexpected

    def forward(self, mel):
        """mel: [B, 80, T] -> (hidden_states tuple of len layers+1, last_hidden), both [B, T, C].

        Transformers 4.46 WhisperEncoder collects states BEFORE each layer (so index 0 is the
        conv+positional embedding) and appends the FINAL layer-normed output last.  The last
        entry is therefore layer_norm(L4), not raw L4 -- getting this wrong silently corrupts
        1 of the 5 audio hidden levels.
        """
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # [B, T, C]
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.embed_positions(positions)[None, :, :]

        hidden_states = []
        for layer in self.layers:
            hidden_states.append(x)
            x = layer(x)
        x = self.layer_norm(x)
        hidden_states.append(x)
        return tuple(hidden_states), x


# ---------------------------------------------------------------- sonic adapters
class AudioProjModel(nn.Module):
    def __init__(self, seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024,
                 context_tokens=32, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = operations.Linear(self.input_dim, intermediate_dim)
        self.proj2 = operations.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = operations.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = operations.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = audio_embeds.reshape(-1, *audio_embeds.shape[2:])  # (bz f) w b c
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)
        context_tokens = self.norm(context_tokens)
        context_tokens = context_tokens.reshape(-1, video_length, *context_tokens.shape[1:])
        return context_tokens


class Audio2bucketModel(nn.Module):
    def __init__(self, seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024,
                 output_dim=1, context_tokens=2, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels + clip_channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = operations.Linear(self.input_dim, intermediate_dim)
        self.proj2 = operations.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = operations.Linear(intermediate_dim, context_tokens * output_dim)
        self.act = nn.SiLU()

    def forward(self, audio_embeds, clip_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = audio_embeds.reshape(-1, *audio_embeds.shape[2:])
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)
        audio_embeds = torch.cat([audio_embeds, clip_embeds], dim=-1)

        audio_embeds = self.act(self.proj1(audio_embeds))
        audio_embeds = self.act(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)
        context_tokens = context_tokens.reshape(-1, video_length, *context_tokens.shape[1:])
        return context_tokens
