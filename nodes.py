"""Classic ComfyUI nodes for SwiftVR video restoration."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .swiftvr_loader import clear_pipeline_cache, get_pipeline
except ImportError:  # Allows direct import during local smoke tests.
    from swiftvr_loader import clear_pipeline_cache, get_pipeline


def _comfy_root() -> Path:
    try:
        import folder_paths  # pylint: disable=import-outside-toplevel

        return Path(folder_paths.base_path)
    except Exception:
        root = Path(__file__).resolve().parent
        if root.parent.name == "custom_nodes":
            return root.parent.parent
        return Path.cwd()


def _resolve_comfy_path(value: str) -> Path:
    path = Path((value or "").strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_comfy_root() / path).resolve()


def _stats_json(stats: dict[str, Any]) -> str:
    return json.dumps(stats, indent=2, ensure_ascii=False)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        common = os.path.commonpath((os.path.normcase(str(path)), os.path.normcase(str(base))))
    except ValueError:
        return False
    return common == os.path.normcase(str(base))


def _output_locator(path: Path) -> dict[str, str] | None:
    """Return a ComfyUI /view locator when the file is under a served folder."""
    try:
        import folder_paths  # pylint: disable=import-outside-toplevel

        candidates = (
            (Path(folder_paths.get_output_directory()).resolve(), "output"),
            (Path(folder_paths.get_temp_directory()).resolve(), "temp"),
            (Path(folder_paths.get_input_directory()).resolve(), "input"),
        )
    except Exception:
        return None

    resolved = path.resolve()
    for base, folder_type in candidates:
        if not _is_relative_to(resolved, base):
            continue
        rel = Path(os.path.relpath(str(resolved), str(base)))
        subfolder = "" if rel.parent == Path(".") else str(rel.parent).replace("\\", "/")
        return {"filename": rel.name, "subfolder": subfolder, "type": folder_type}
    return None


def _default_output_path(input_path: Path, output_dir_value: str, filename_value: str) -> Path:
    output_dir = _resolve_comfy_path(output_dir_value or "output/swiftvr")
    filename = (filename_value or "").strip()
    if not filename:
        filename = f"{input_path.stem}_swiftvr.mp4"
    if Path(filename).suffix == "":
        filename = f"{filename}.mp4"
    return (output_dir / filename).resolve()


def _default_stats_path(output_path: Path, stats_filename_value: str) -> Path:
    stats_filename = (stats_filename_value or "").strip()
    if stats_filename:
        stats_path = Path(stats_filename).expanduser()
        if stats_path.is_absolute():
            return stats_path.resolve()
        return (output_path.parent / stats_path).resolve()
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_stats.json")
    return output_path / "swiftvr_stats.json"


def _temp_video_path() -> Path:
    try:
        import folder_paths  # pylint: disable=import-outside-toplevel

        temp_dir = Path(folder_paths.get_temp_directory())
    except Exception:
        temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="swiftvr_input_", suffix=".mp4", dir=str(temp_dir))
    os.close(fd)
    Path(path).unlink(missing_ok=True)
    return Path(path)


def _video_source_path(video: Any) -> tuple[Path, Path | None]:
    """Resolve a Comfy VIDEO input to a file path SwiftVR can stream from."""
    if isinstance(video, dict) and "input_path" in video:
        return Path(video["input_path"]).expanduser().resolve(), None
    if isinstance(video, (str, os.PathLike)):
        return _resolve_comfy_path(str(video)), None
    if not hasattr(video, "get_stream_source"):
        raise TypeError("SwiftVR Restore Video expects a Comfy VIDEO input from the Load Video node.")

    source = video.get_stream_source()
    start_time, duration = 0.0, 0.0
    if hasattr(video, "get_active_trim_window"):
        try:
            start_time, duration = video.get_active_trim_window()
        except Exception:
            start_time, duration = 0.0, 0.0

    has_trim = abs(float(start_time or 0.0)) > 1e-9 or abs(float(duration or 0.0)) > 1e-9
    if isinstance(source, (str, os.PathLike)) and not has_trim:
        resolved = Path(source).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"SwiftVR input video does not exist: {resolved}")
        return resolved, None

    if not hasattr(video, "save_to"):
        raise TypeError("SwiftVR could not materialize the linked VIDEO input to a local file.")
    temp_path = _temp_video_path()
    video.save_to(str(temp_path))
    return temp_path.resolve(), temp_path


def _advanced_defaults() -> dict[str, Any]:
    return {
        "upscale": 2,
        "clip_len": 16,
        "dit_overlap": 0,
        "fps": 0.0,
        "quality": 85,
        "save_format": "",
        "ffmpeg_preset": "",
        "queue_size": 2,
        "verbose": True,
        "clear_cache_after": True,
    }


def _normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
    merged = _advanced_defaults()
    if options:
        merged.update(options)
    return merged


def _make_ui(output_path: Path, stats: dict[str, Any], cache_message: str) -> dict[str, Any]:
    text = _stats_json(stats)
    if cache_message:
        text = f"{text}\n\n{cache_message}"

    ui: dict[str, Any] = {"text": (text,)}
    locator = _output_locator(output_path)
    if locator is not None and output_path.is_file():
        ui["images"] = [locator]
        ui["animated"] = (True,)
    return ui


def _clear_after_restore() -> str:
    try:
        return clear_pipeline_cache()
    except Exception as exc:  # noqa: BLE001
        return f"SwiftVR cache clear failed: {exc}"


class SwiftVRModelLoader:
    """Load a SwiftVR checkpoint and output a pipeline object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint_dir": ("STRING", {"default": "auto"}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            "attention_backend": (["auto", "sdpa", "flash_attn_3", "flash_attn_2", "sageattention", "xformers"], {"default": "auto"}),
            "torch_compile": ("BOOLEAN", {"default": False}),
            "upscale_mode": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
            "reae_filename": ("STRING", {"default": "reae.safetensors", "advanced": True}),
            "transformer_subfolder": ("STRING", {"default": "transformer", "advanced": True}),
            "prompt_embedding_filename": ("STRING", {"default": "prompt_embedding.safetensors", "advanced": True}),
        }}

    RETURN_TYPES = ("SWIFTVR_PIPE",)
    RETURN_NAMES = ("swiftvr_pipe",)
    FUNCTION = "load"
    CATEGORY = "SwiftVR"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):  # noqa: ARG003
        # The restore node clears the model after each run, so do not let ComfyUI
        # keep an old GPU pipeline alive through output caching.
        return time.time()

    def load(self, checkpoint_dir, device, dtype, attention_backend, torch_compile,
             upscale_mode, reae_filename, transformer_subfolder, prompt_embedding_filename):
        pipe = get_pipeline(checkpoint_dir, device, dtype, attention_backend, torch_compile,
                            upscale_mode, reae_filename, transformer_subfolder,
                            prompt_embedding_filename)
        return (pipe,)


class SwiftVRAdvancedOptions:
    """Optional runtime options for long videos or constrained GPUs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "upscale": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1}),
            "clip_len": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4}),
            "dit_overlap": ("INT", {"default": 0, "min": 0, "max": 8}),
            "fps": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "quality": ("INT", {"default": 85, "min": 0, "max": 100}),
            "save_format": (["", "yuv444p"], {"default": ""}),
            "ffmpeg_preset": ("STRING", {"default": ""}),
            "queue_size": ("INT", {"default": 2, "min": 1, "max": 16}),
            "verbose": ("BOOLEAN", {"default": True}),
            "clear_cache_after": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("SWIFTVR_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "SwiftVR/Settings"

    def build(self, upscale, clip_len, dit_overlap, fps, quality, save_format, ffmpeg_preset,
              queue_size, verbose, clear_cache_after):
        return ({
            "upscale": upscale,
            "clip_len": clip_len,
            "dit_overlap": dit_overlap,
            "fps": fps,
            "quality": quality,
            "save_format": save_format,
            "ffmpeg_preset": ffmpeg_preset,
            "queue_size": queue_size,
            "verbose": verbose,
            "clear_cache_after": clear_cache_after,
        },)


class SwiftVRRestoreVideo:
    """Restore a Comfy VIDEO, show the output, then clear SwiftVR cache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "swiftvr_pipe": ("SWIFTVR_PIPE",),
                "video": ("VIDEO",),
                "output_dir": ("STRING", {"default": "output/swiftvr"}),
                "filename": ("STRING", {"default": "test_swiftvr.mp4"}),
            },
            "optional": {
                "options": ("SWIFTVR_OPTIONS",),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "restore"
    CATEGORY = "SwiftVR"
    OUTPUT_NODE = True

    def restore(self, swiftvr_pipe, video, output_dir, filename, options=None):
        opts = _normalize_options(options)
        input_path, temp_input_path = _video_source_path(video)
        output_path = _default_output_path(input_path, output_dir, filename)
        stats_path = _default_stats_path(output_path, "")
        cache_message = ""
        upscale = int(opts["upscale"])

        try:
            stats = swiftvr_pipe.restore_video(
                str(input_path),
                str(output_path),
                resolution=None,
                upscale=upscale,
                clip_len=int(opts["clip_len"]),
                dit_overlap=int(opts["dit_overlap"]),
                fps=None if float(opts["fps"]) <= 0 else float(opts["fps"]),
                quality=int(opts["quality"]),
                png_save=False,
                save_format=opts["save_format"],
                ffmpeg_preset=opts["ffmpeg_preset"],
                queue_size=int(opts["queue_size"]),
                verbose=bool(opts["verbose"]),
            )
        except Exception as exc:  # noqa: BLE001
            if opts["clear_cache_after"]:
                swiftvr_pipe = None
                _clear_after_restore()
            if exc.__class__.__name__ == "InterruptProcessingException":
                raise
            raise RuntimeError(f"SwiftVR Restore Video failed: {exc}") from exc
        finally:
            if temp_input_path is not None:
                temp_input_path.unlink(missing_ok=True)

        stats = dict(stats)
        preview_path = output_path.expanduser().resolve()
        stats["input"] = str(input_path)
        stats["output"] = str(preview_path)
        stats["stats"] = str(stats_path)
        stats["upscale"] = upscale

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(_stats_json(stats) + "\n", encoding="utf-8")

        if opts["clear_cache_after"]:
            swiftvr_pipe = None
            cache_message = _clear_after_restore()
            stats["cache"] = cache_message
            stats_path.write_text(_stats_json(stats) + "\n", encoding="utf-8")

        return {"ui": _make_ui(preview_path, stats, cache_message)}


class SwiftVRRestoreVideoPath:
    """Deprecated all-in-one path restore node kept for older workflows."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "swiftvr_pipe": ("SWIFTVR_PIPE",),
            "input_path": ("STRING", {"default": ""}),
            "output_path": ("STRING", {"default": "output/swiftvr/restored.mp4"}),
            "stats_path": ("STRING", {"default": ""}),
            "resolution": ("STRING", {"default": "", "advanced": True}),
            "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            "clip_len": ("INT", {"default": 24, "min": 4, "max": 128, "step": 4}),
            "dit_overlap": ("INT", {"default": 0, "min": 0, "max": 8}),
            "fps": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "quality": ("INT", {"default": 85, "min": 0, "max": 100}),
            "png_save": ("BOOLEAN", {"default": False}),
            "save_format": (["", "yuv444p"], {"default": ""}),
            "ffmpeg_preset": ("STRING", {"default": ""}),
            "queue_size": ("INT", {"default": 3, "min": 1, "max": 16}),
            "verbose": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "stats_json")
    FUNCTION = "restore"
    CATEGORY = "SwiftVR/Legacy"
    OUTPUT_NODE = True

    def restore(self, swiftvr_pipe, input_path, output_path, stats_path, resolution, upscale, clip_len,
                dit_overlap, fps, quality, png_save, save_format, ffmpeg_preset, queue_size, verbose):
        input_path = _resolve_comfy_path(input_path)
        output_path = _resolve_comfy_path(output_path)
        stats_path = _resolve_comfy_path(stats_path) if (stats_path or "").strip() else _default_stats_path(output_path, "")
        cache_message = ""
        try:
            stats = swiftvr_pipe.restore_video(
                str(input_path), str(output_path), resolution=None,
                upscale=upscale, clip_len=clip_len, dit_overlap=dit_overlap,
                fps=None if fps <= 0 else fps, quality=quality, png_save=png_save,
                save_format=save_format, ffmpeg_preset=ffmpeg_preset,
                queue_size=queue_size, verbose=verbose,
            )
            stats = dict(stats)
            preview_path = Path(stats.get("output", output_path)).expanduser().resolve()
            stats["input"] = str(input_path)
            stats["output"] = str(preview_path)
            stats["stats"] = str(stats_path)
            stats["upscale"] = int(upscale)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(_stats_json(stats) + "\n", encoding="utf-8")
            swiftvr_pipe = None
            cache_message = _clear_after_restore()
            stats["cache"] = cache_message
            stats_path.write_text(_stats_json(stats) + "\n", encoding="utf-8")
            return {
                "ui": _make_ui(preview_path, stats, cache_message),
                "result": (str(preview_path), _stats_json(stats)),
            }
        except Exception as exc:  # noqa: BLE001
            swiftvr_pipe = None
            _clear_after_restore()
            if exc.__class__.__name__ == "InterruptProcessingException":
                raise
            raise RuntimeError(f"SwiftVR Restore Video Path failed: {exc}") from exc


class SwiftVRRestoreImageBatch:
    """Restore a ComfyUI IMAGE batch by round-tripping through PNG frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "swiftvr_pipe": ("SWIFTVR_PIPE",),
            "images": ("IMAGE",),
            "resolution": ("STRING", {"default": "", "advanced": True}),
            "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            "clip_len": ("INT", {"default": 24, "min": 4, "max": 128, "step": 4}),
            "dit_overlap": ("INT", {"default": 0, "min": 0, "max": 8}),
            "quality": ("INT", {"default": 95, "min": 0, "max": 100}),
            "keep_temp": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "output_dir", "stats_json")
    FUNCTION = "restore"
    CATEGORY = "SwiftVR/Legacy"

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
                str(input_dir), str(output_dir), resolution=None,
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
            swiftvr_pipe = None
            _clear_after_restore()
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
    CATEGORY = "SwiftVR/Utility"

    def clear(self, anything=None):  # noqa: ARG002
        return (clear_pipeline_cache(),)


NODE_CLASS_MAPPINGS = {
    "SwiftVRModelLoader": SwiftVRModelLoader,
    "SwiftVRAdvancedOptions": SwiftVRAdvancedOptions,
    "SwiftVRRestoreVideo": SwiftVRRestoreVideo,
    "SwiftVRRestoreVideoPath": SwiftVRRestoreVideoPath,
    "SwiftVRRestoreImageBatch": SwiftVRRestoreImageBatch,
    "SwiftVRClearCache": SwiftVRClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwiftVRModelLoader": "SwiftVR Model Loader",
    "SwiftVRAdvancedOptions": "SwiftVR Advanced Options",
    "SwiftVRRestoreVideo": "SwiftVR Restore Video",
    "SwiftVRRestoreVideoPath": "SwiftVR Restore Video Path (Legacy)",
    "SwiftVRRestoreImageBatch": "SwiftVR Restore Image Batch (Legacy)",
    "SwiftVRClearCache": "SwiftVR Clear Cache",
}
