"""CPU-side contracts for Sonic decoding; MPS labels do not run the MPS backend.

Execute the real functions from sonic.py via AST to avoid importing the unrelated
audio/face/pipeline dependencies. Tensors are real PyTorch CPU tensors; ComfyUI's
VAE construction and model management are stubbed, not integration-tested here.
Run: python -m unittest discover -s tests -v
"""

import ast
import contextlib
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch


ROOT = Path(__file__).resolve().parents[1]
ITERATOR_ERROR = "Can't be indexed using 32-bit iterator"


def load_functions(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"decode_latents_", "test"}
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


SONIC = load_functions(ROOT / "sonic.py")


class FailurePayload:
    pass


class TrackingVAE:
    """Simulate ComfyUI placement while using CPU tensors for every device label."""

    def __init__(self, device="cpu", load_device=None, dtype=torch.float32, failure=None, fail_at=1):
        self.device = torch.device(device)
        self.patcher = SimpleNamespace(load_device=torch.device(load_device or device))
        self.vae_dtype = dtype
        self.output_device = torch.device("cpu")
        self.failure = failure
        self.fail_at = fail_at
        self.calls = []
        self.failure_payload = None
        self.state = {
            "decoder.weight": torch.ones(3, 4, 1, 1, dtype=dtype),
            "counter": torch.tensor(7, dtype=torch.int64),
        }

    def decode(self, chunk):
        self.calls.append(chunk.detach().clone())
        if self.failure and len(self.calls) == self.fail_at:
            payload = FailurePayload()
            self.failure_payload = weakref.ref(payload)
            raise RuntimeError(self.failure)
        return chunk[:, :3].movedim(1, -1).to(dtype=self.vae_dtype)

    def get_sd(self):
        return self.state


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.latents = torch.arange(5 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 5, 4, 2, 3)
        self.cpu_vaes = []
        self.constructor_args = []
        self.cpu_failure = None
        self.before_cpu_construct = None

        def cpu_constructor(*, sd, device, dtype):
            if self.before_cpu_construct:
                self.before_cpu_construct()
            self.constructor_args.append((sd, device, dtype))
            vae = TrackingVAE(device=device, dtype=dtype, failure=self.cpu_failure)
            vae.state = sd
            self.cpu_vaes.append(vae)
            return vae

        comfy = ModuleType("comfy")
        comfy.__path__ = []
        comfy_sd = ModuleType("comfy.sd")
        comfy_sd.VAE = Mock(side_effect=cpu_constructor)
        comfy.sd = comfy_sd
        self.constructor = comfy_sd.VAE
        self.modules = patch.dict(sys.modules, {"comfy": comfy, "comfy.sd": comfy_sd})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def decode(self, vae, device="cpu", size=3, latents=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return SONIC["decode_latents_"](
                self.latents if latents is None else latents, vae, torch.device(device), size
            )

    def assert_output(self, result, latents=None):
        latents = self.latents if latents is None else latents
        expected = (latents.flatten(0, 1) * (1 / 0.18215))[:, :3]
        expected = expected.movedim(1, -1).unsqueeze(0).permute(0, 4, 1, 2, 3).float()
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")

    def test_cpu_respects_requested_chunk_size(self):
        vae = TrackingVAE()
        result = self.decode(vae, size=2)
        self.assertEqual([len(chunk) for chunk in vae.calls], [2, 2, 1])
        self.assert_output(result)
        self.constructor.assert_not_called()

    def test_exact_mps_error_switches_to_cpu_for_remaining_frames(self):
        vae = TrackingVAE("mps", failure=ITERATOR_ERROR)
        result = self.decode(vae, device="mps")
        self.constructor.assert_called_once()
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual([len(chunk) for chunk in self.cpu_vaes[0].calls], [1] * 5)
        self.assert_output(result)

    def test_fallback_preserves_frames_decoded_before_the_failure(self):
        vae = TrackingVAE("mps", failure=ITERATOR_ERROR, fail_at=3)
        result = self.decode(vae, device="mps")
        self.assertEqual(len(vae.calls), 3)
        self.assertEqual(len(self.cpu_vaes[0].calls), 3)
        self.assert_output(result)

    def test_fallback_copies_weights_to_cpu_float32_without_mutating_original(self):
        vae = TrackingVAE("mps", dtype=torch.float16, failure=ITERATOR_ERROR)
        original_patcher = vae.patcher
        original_weight = vae.state["decoder.weight"]
        result = self.decode(vae, device="mps")
        sd, device, dtype = self.constructor_args[0]
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(sd["decoder.weight"].dtype, torch.float32)
        self.assertEqual(sd["decoder.weight"].device.type, "cpu")
        self.assertNotEqual(sd["decoder.weight"].data_ptr(), original_weight.data_ptr())
        self.assertEqual(sd["counter"].dtype, torch.int64)
        self.assertEqual(sd["counter"].item(), 7)
        self.assertIs(vae.patcher, original_patcher)
        self.assertIs(vae.state["decoder.weight"], original_weight)
        self.assertEqual(vae.vae_dtype, torch.float16)
        self.assertEqual(vae.device, torch.device("mps"))
        self.assertEqual(vae.patcher.load_device, torch.device("mps"))
        self.assertEqual(self.cpu_vaes[0].output_device, torch.device("cpu"))
        self.assertTrue(all(chunk.dtype == torch.float32 for chunk in self.cpu_vaes[0].calls))
        self.assert_output(result)

    def test_failed_decode_traceback_is_released_before_cpu_copy(self):
        vae = TrackingVAE("mps", failure=ITERATOR_ERROR)

        def check_released():
            self.assertIsNotNone(vae.failure_payload)
            self.assertIsNone(vae.failure_payload())

        self.before_cpu_construct = check_released
        self.assert_output(self.decode(vae, device="mps"))

    def test_known_error_on_other_devices_propagates(self):
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                vae = TrackingVAE(device, failure=ITERATOR_ERROR)
                with self.assertRaisesRegex(RuntimeError, "32-bit iterator"):
                    self.decode(vae, device=device)
                self.assertEqual(len(vae.calls), 1)
        self.constructor.assert_not_called()

    def test_unrelated_mps_errors_propagate(self):
        for message in ("out of memory", "unrelated 32-bit iterator error", "bad shape"):
            with self.subTest(message=message):
                vae = TrackingVAE("mps", failure=message)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.decode(vae, device="mps")
        self.constructor.assert_not_called()

    def test_cpu_retry_failure_propagates_without_another_retry(self):
        vae = TrackingVAE("mps", failure=ITERATOR_ERROR)
        self.cpu_failure = ITERATOR_ERROR
        with self.assertRaisesRegex(RuntimeError, "32-bit iterator"):
            self.decode(vae, device="mps")
        self.constructor.assert_called_once()
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual(len(self.cpu_vaes[0].calls), 1)
        self.assertEqual(vae.device, torch.device("mps"))

    def test_cpu_constructor_failure_does_not_mutate_original(self):
        vae = TrackingVAE("mps", failure=ITERATOR_ERROR)
        self.constructor.side_effect = RuntimeError("CPU constructor failed")
        with self.assertRaisesRegex(RuntimeError, "CPU constructor failed"):
            self.decode(vae, device="mps")
        self.assertEqual(vae.device, torch.device("mps"))
        self.assertEqual(vae.patcher.load_device, torch.device("mps"))

    def test_non_runtime_errors_propagate(self):
        vae = TrackingVAE("mps")
        vae.decode = Mock(side_effect=ValueError("invalid input"))
        with self.assertRaisesRegex(ValueError, "invalid input"):
            self.decode(vae, device="mps")
        self.constructor.assert_not_called()

    def test_wrapper_device_is_used_when_no_patcher_exists(self):
        vae = TrackingVAE("mps")
        del vae.patcher
        self.assert_output(self.decode(vae))
        self.assertEqual([len(chunk) for chunk in vae.calls], [1] * 5)

    def test_argument_device_is_used_when_wrapper_has_no_device(self):
        vae = TrackingVAE()
        del vae.patcher
        del vae.device
        self.assert_output(self.decode(vae, device="mps"))
        self.assertEqual([len(chunk) for chunk in vae.calls], [1] * 5)
        self.assertFalse(hasattr(vae, "device"))

    def test_half_precision_output_is_cast_to_float32(self):
        latents = self.latents.to(torch.float16)
        result = self.decode(TrackingVAE(dtype=torch.float16), latents=latents)
        self.assert_output(result, latents)

    def test_flattened_batch_and_frame_order_is_preserved(self):
        latents = torch.cat([self.latents, self.latents + 500], dim=0)
        result = self.decode(TrackingVAE(), latents=latents, size=3)
        self.assertEqual(tuple(result.shape), (1, 3, 10, 2, 3))
        self.assert_output(result, latents)

    def test_cuda_respects_requested_chunk_size(self):
        vae = TrackingVAE("cuda:0")
        result = self.decode(vae, device="cuda:0", size=3)
        self.assertEqual([len(chunk) for chunk in vae.calls], [3, 2])
        self.assert_output(result)
        self.constructor.assert_not_called()

    def test_mps_decodes_single_frames(self):
        vae = TrackingVAE("mps")
        result = self.decode(vae, device="mps", size=14)
        self.assertEqual([len(chunk) for chunk in vae.calls], [1] * 5)
        self.assert_output(result)
        self.constructor.assert_not_called()

    def test_patcher_device_takes_precedence_over_stale_wrapper_device(self):
        vae = TrackingVAE("cpu", load_device="mps")
        result = self.decode(vae, size=8)
        self.assertEqual([len(chunk) for chunk in vae.calls], [1] * 5)
        self.assertEqual(vae.device, torch.device("cpu"))
        self.assert_output(result)

    def test_cpu_vae_is_not_overridden_by_mps_sonic_device(self):
        vae = TrackingVAE("cpu")
        result = self.decode(vae, device="mps", size=3)
        self.assertEqual([len(chunk) for chunk in vae.calls], [3, 2])
        self.assertEqual(vae.device, torch.device("cpu"))
        self.assertEqual(vae.patcher.load_device, torch.device("cpu"))
        self.assert_output(result)

    def test_invalid_chunk_sizes_fail_before_decode(self):
        for size in (0, -1, True, False, 1.5, "2", None):
            with self.subTest(size=size):
                vae = TrackingVAE("mps")
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.decode(vae, device="mps", size=size)
                self.assertEqual(vae.calls, [])
        self.constructor.assert_not_called()


class CallerTests(unittest.TestCase):
    def test_configured_chunk_size_reaches_the_final_decode(self):
        latents = torch.zeros(1, 3, 4, 2, 2)
        output = object()
        decoder = Mock(return_value=output)
        pipe = Mock(return_value=SimpleNamespace(frames=latents))
        config = SimpleNamespace(
            decode_chunk_size=2, motion_bucket_scale=1.0, noise_aug_strength=0.0,
            min_appearance_guidance_scale=2.0, max_appearance_guidance_scale=2.0,
            audio_guidance_scale=7.5, overlap=0, shift_offset=7,
            num_inference_steps=25, i2i_noise_strength=1.0,
        )
        vae = object()
        device = torch.device("cpu")
        with patch.dict(SONIC, {"decode_latents_": decoder}):
            result = SONIC["test"](
                pipe, config, [None] * 3, [None] * 3, [None] * 3,
                16, 16, {"ref_img": None, "face_mask": None}, None,
                25, None, vae, device,
            )
        self.assertIs(result, output)
        decoder.assert_called_once_with(latents, vae, device, decode_chunk_size=2)
        self.assertEqual(pipe.call_args.kwargs["decode_chunk_size"], 2)
        pipe.to.assert_called_once_with(device=torch.device("cpu"))

    def test_predata_does_not_assign_to_the_shared_vae_device(self):
        tree = ast.parse((ROOT / "sonic_node.py").read_text(encoding="utf-8"))
        predata = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SONIC_PreData")
        assignments = [node for node in ast.walk(predata) if isinstance(node, ast.Assign)]
        for assignment in assignments:
            for target in assignment.targets:
                self.assertFalse(
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "vae" and target.attr == "device",
                    "ComfyUI must own VAE device selection",
                )


if __name__ == "__main__":
    unittest.main()
