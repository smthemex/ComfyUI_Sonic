# Issue #98: MPS VAE decode workaround

This is a candidate workaround, not an M1 Ultra-verified fix or a PyTorch fix.
It limits the actual MPS VAE to one-frame decode calls. CPU/CUDA VAEs honor
`config.decode_chunk_size` (the shipped value is 8, previously ignored by the
final decode). ComfyUI retains control of VAE placement during encoding and
decoding, including when Sonic runs on MPS but the VAE is configured on CPU.

Only the MPS `Can't be indexed using 32-bit iterator` RuntimeError triggers a
separate CPU/fp32 VAE built from copies of the loaded weights. That VAE is reused
for the rest of the video; the original VAE's device, patcher, dtype, and weights
are not directly rewritten. Other errors, including failed CPU retries, still
propagate. CPU fallback is slower and temporarily adds a CPU weight copy.

## Automated checks

From the custom-node repository with PyTorch installed:

```bash
python -m unittest discover -s tests -v
git diff --check
```

The tests execute the actual `sonic.py` functions through AST extraction to avoid
loading unrelated audio/face/pipeline dependencies. They use real CPU tensors
and stub the ComfyUI VAE/model-manager contract. MPS/CUDA device labels only
exercise branching; they do not test those accelerators. A passing suite is not
proof that a full ComfyUI Desktop workflow or CPU fallback works on a Mac.

## M1 Ultra hardware acceptance

1. Save local changes before trying the candidate branch. Use the same image,
   audio, checkpoints, seed, and installed Torch versions as the failing run.
2. Install the PR's changes to **both** `sonic.py` and `sonic_node.py`, or test
   the candidate branch in a clean checkout. Fully restart ComfyUI Desktop.
3. Use the fp32 workflow already known to produce usable output at minimum
   resolution 256. Confirm that control still works, then repeat at 512 with a
   short audio clip first. Record actual output dimensions and frame count.
4. Confirm generation completes, video is non-black, frames remain ordered,
   and temporal quality is acceptable. Single-frame decoding changes the
   temporal VAE's context, so compare flicker/motion as well as crash behavior.
5. Record elapsed time and peak unified-memory usage under comparable conditions.
   Smaller VAE decode activations do not guarantee lower total workflow memory;
   no numeric memory improvement has been measured for this change.
6. If the CPU/fp32 fallback message appears, confirm the CPU decode finishes.
   If not, share the **new full traceback**, ComfyUI Desktop/core versions,
   Python/Torch versions, VAE checkpoint, workflow settings, and dimensions.
   Redact local personal paths and any credentials before sharing logs.

Hardware validation, model-backed ComfyUI integration, CPU-fallback performance,
and peak-memory measurements remain outstanding. Keep #98 open until the
requester confirms the original 512px problem is actually resolved.
