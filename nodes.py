"""Classic ComfyUI nodes for SwiftVR video restoration."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .swiftvr_loader import clear_pipeline_cache, get_pipeline
except ImportError:  # Allows direct import during local smoke tests.
    from swiftvr_loader import clear_pipeline_cache, get_pipeline


def _parse_resolution(value: str):
    text = (value or "").strip().lower()
    if not text:
        return None
    if "x" not in text:
        raise RuntimeError("Resolution must be empty or formatted like 1920x1080.")
    parts = text.split("x", 1)
    try:
        width, height = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise RuntimeError("Resolution must contain integer width and height, e.g. 3840x2160.") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError("Resolution width and height must be greater than zero.")
    return (width, height)


def _stats_json(stats: dict[str, Any]) -> str:
    return json.dumps(stats, indent=2, ensure_ascii=False)


class SwiftVRModelLoader:
    """Load a SwiftVR checkpoint once and output a reusable pipeline object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint_dir": ("STRING", {"default": ""}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            "attention_backend": (["auto", "sdpa", "flash_attn_3", "flash_attn_2", "sageattention", "xformers"], {"default": "auto"}),
            "torch_compile": ("BOOLEAN", {"default": False}),
            "upscale_mode": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
            "reae_filename": ("STRING", {"default": "reae.safetensors"}),
            "transformer_subfolder": ("STRING", {"default": "transformer"}),
            "prompt_embedding_filename": ("STRING", {"default": "prompt_embedding.safetensors"}),
        }}

    RETURN_TYPES = ("SWIFTVR_PIPE",)
    RETURN_NAMES = ("swiftvr_pipe",)
    FUNCTION = "load"
    CATEGORY = "SwiftVR"

    def load(self, checkpoint_dir, device, dtype, attention_backend, torch_compile,
             upscale_mode, reae_filename, transformer_subfolder, prompt_embedding_filename):
        pipe = get_pipeline(checkpoint_dir, device, dtype, attention_backend, torch_compile,
                            upscale_mode, reae_filename, transformer_subfolder,
                            prompt_embedding_filename)
        return (pipe,)


class SwiftVRRestoreVideoPath:
    """Restore a video file or image-folder path and write output to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "swiftvr_pipe": ("SWIFTVR_PIPE",),
            "input_path": ("STRING", {"default": ""}),
            "output_path": ("STRING", {"default": ""}),
            "resolution": ("STRING", {"default": ""}),
            "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            "clip_len": ("INT", {"default": 24, "min": 4, "max": 128, "step": 4}),
            "dit_overlap": ("INT", {"default": 0, "min": 0, "max": 8}),
            "fps": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "quality": ("INT", {"default": 85, "min": 0, "max": 100}),
            "png_save": ("BOOLEAN", {"default": False}),
            "save_format": (["", "yuv444p"],),
            "ffmpeg_preset": ("STRING", {"default": ""}),
            "queue_size": ("INT", {"default": 3, "min": 1, "max": 16}),
            "verbose": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "stats_json")
    FUNCTION = "restore"
    CATEGORY = "SwiftVR"
    OUTPUT_NODE = True

    def restore(self, swiftvr_pipe, input_path, output_path, resolution, upscale, clip_len,
                dit_overlap, fps, quality, png_save, save_format, ffmpeg_preset, queue_size, verbose):
        try:
            if not Path(input_path).expanduser().exists():
                raise RuntimeError(f"Input path does not exist: {input_path}")
            stats = swiftvr_pipe.restore_video(
                str(Path(input_path).expanduser()), str(Path(output_path).expanduser()),
                resolution=_parse_resolution(resolution), upscale=upscale, clip_len=clip_len,
                dit_overlap=dit_overlap, fps=None if fps <= 0 else fps, quality=quality,
                png_save=png_save, save_format=save_format, ffmpeg_preset=ffmpeg_preset,
                queue_size=queue_size, verbose=verbose,
            )
            return (str(stats.get("output", output_path)), _stats_json(stats))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"SwiftVR Restore Video Path failed: {exc}") from exc


class SwiftVRRestoreImageBatch:
    """Restore a ComfyUI IMAGE batch by round-tripping through PNG frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "swiftvr_pipe": ("SWIFTVR_PIPE",),
            "images": ("IMAGE",),
            "resolution": ("STRING", {"default": ""}),
            "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            "clip_len": ("INT", {"default": 24, "min": 4, "max": 128, "step": 4}),
            "dit_overlap": ("INT", {"default": 0, "min": 0, "max": 8}),
            "quality": ("INT", {"default": 95, "min": 0, "max": 100}),
            "keep_temp": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "output_dir", "stats_json")
    FUNCTION = "restore"
    CATEGORY = "SwiftVR"

    def restore(self, swiftvr_pipe, images, resolution, upscale, clip_len, dit_overlap, quality, keep_temp):
        temp_root = Path(tempfile.mkdtemp(prefix="comfyui_swiftvr_"))
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np
            import torch
            from PIL import Image

            frames = images.detach().cpu().clamp(0, 1).numpy()
            for idx, frame in enumerate(frames):
                arr = (frame * 255.0).round().astype(np.uint8)
                Image.fromarray(arr).save(input_dir / f"{idx:08d}.png")

            stats = swiftvr_pipe.restore_video(
                str(input_dir), str(output_dir), resolution=_parse_resolution(resolution),
                upscale=upscale, clip_len=clip_len, dit_overlap=dit_overlap, fps=None,
                quality=quality, png_save=True, save_format="", ffmpeg_preset="",
                queue_size=3, verbose=True,
            )
            produced = sorted(output_dir.glob("*.png"))
            if not produced:
                produced = sorted(output_dir.rglob("*.png"))
            if not produced:
                raise RuntimeError(f"SwiftVR did not produce PNG frames in {output_dir}")

            restored = []
            for path in produced:
                with Image.open(path) as img:
                    restored.append(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
            tensor = torch.from_numpy(np.stack(restored, axis=0))
            stats = dict(stats)
            stats["input_frames"] = int(frames.shape[0])
            stats["returned_frames"] = int(tensor.shape[0])
            stats["note"] = "Returned all PNG frames produced by SwiftVR; count may reflect its internal 4k+1 handling."
            final_output = str(output_dir if keep_temp else Path(stats.get("output", output_dir)))
            return (tensor, final_output, _stats_json(stats))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"SwiftVR Restore Image Batch failed: {exc}") from exc
        finally:
            if not keep_temp:
                shutil.rmtree(temp_root, ignore_errors=True)


class SwiftVRClearCache:
    """Clear cached SwiftVR pipelines and CUDA allocator state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"anything": ("*",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "clear"
    CATEGORY = "SwiftVR"

    def clear(self, anything=None):  # noqa: ARG002
        return (clear_pipeline_cache(),)


NODE_CLASS_MAPPINGS = {
    "SwiftVRModelLoader": SwiftVRModelLoader,
    "SwiftVRRestoreVideoPath": SwiftVRRestoreVideoPath,
    "SwiftVRRestoreImageBatch": SwiftVRRestoreImageBatch,
    "SwiftVRClearCache": SwiftVRClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwiftVRModelLoader": "SwiftVR Model Loader",
    "SwiftVRRestoreVideoPath": "SwiftVR Restore Video Path",
    "SwiftVRRestoreImageBatch": "SwiftVR Restore Image Batch",
    "SwiftVRClearCache": "SwiftVR Clear Cache",
}
