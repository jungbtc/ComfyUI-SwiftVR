"""Loading and utility helpers for the ComfyUI SwiftVR nodes."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

_PIPELINE_CACHE: dict[tuple[str, str, str, str, bool, str, str, str, str], Any] = {}
_HF_REPO_ID = "H-oliday/SwiftVR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _comfy_root() -> Path | None:
    """Return the ComfyUI root when the custom node lives under custom_nodes."""
    root = _repo_root()
    if root.parent.name == "custom_nodes":
        return root.parent.parent
    return None


def _default_checkpoint_dir() -> Path:
    try:
        import folder_paths  # pylint: disable=import-outside-toplevel

        models_dir = Path(folder_paths.models_dir)
    except Exception:
        comfy_root = _comfy_root()
        models_dir = (comfy_root / "models") if comfy_root else (_repo_root() / "models")
    return models_dir / "SwiftVR"


def _is_comfy_root(path: Path) -> bool:
    return (path / "main.py").is_file() and (path / "custom_nodes").is_dir()


def _resolve_checkpoint_dir(checkpoint_dir: str) -> Path:
    text = (checkpoint_dir or "").strip()
    if not text or text.lower() == "auto":
        return _default_checkpoint_dir().expanduser().resolve()

    root = Path(text).expanduser().resolve()
    if _is_comfy_root(root):
        return _default_checkpoint_dir().expanduser().resolve()
    return root


def _download_checkpoint(root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SwiftVR checkpoint was not found and huggingface_hub is unavailable, "
            "so it cannot be downloaded automatically. Install huggingface_hub or "
            f"download {_HF_REPO_ID} into {root}."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(root),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to auto-download SwiftVR checkpoint from {_HF_REPO_ID} to {root}: {exc}"
        ) from exc


def _import_pipeline():
    """Import SwiftVR lazily so ComfyUI startup survives missing optional deps."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from swiftvr import SwiftVRPipeline  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001 - converted to a ComfyUI-friendly error.
        raise RuntimeError(
            "SwiftVR could not be imported. Install this custom node's requirements "
            "from the ComfyUI Python environment, then restart ComfyUI. Original error: "
            f"{exc}"
        ) from exc
    return SwiftVRPipeline


def validate_checkpoint(
    checkpoint_dir: str,
    reae_filename: str,
    transformer_subfolder: str,
    prompt_embedding_filename: str,
) -> Path:
    """Validate the expected SwiftVR checkpoint layout and return the root path."""
    root = _resolve_checkpoint_dir(checkpoint_dir)
    if not root.exists():
        _download_checkpoint(root)
    if not root.is_dir():
        raise RuntimeError(f"SwiftVR checkpoint path is not a directory: {root}")

    missing: list[str] = []
    for rel in (reae_filename, prompt_embedding_filename, transformer_subfolder):
        if not (root / rel).exists():
            missing.append(str(root / rel))
    if missing:
        if root == _default_checkpoint_dir().expanduser().resolve():
            _download_checkpoint(root)
            missing = [
                str(root / rel)
                for rel in (reae_filename, prompt_embedding_filename, transformer_subfolder)
                if not (root / rel).exists()
            ]
    if missing:
        raise RuntimeError(
            "SwiftVR checkpoint is incomplete. Missing required path(s):\n- "
            + "\n- ".join(missing)
            + "\nExpected layout includes reae.safetensors, prompt_embedding.safetensors, "
              "and a transformer/ subfolder unless you changed those inputs."
        )
    return root


def get_pipeline(
    checkpoint_dir: str,
    device: str,
    dtype: str,
    attention_backend: str,
    torch_compile: bool,
    upscale_mode: str,
    reae_filename: str,
    transformer_subfolder: str,
    prompt_embedding_filename: str,
):
    """Load, prepare, and cache a SwiftVRPipeline for repeated ComfyUI executions."""
    root = validate_checkpoint(
        checkpoint_dir, reae_filename, transformer_subfolder, prompt_embedding_filename
    )
    key = (
        str(root), device, dtype, attention_backend, bool(torch_compile), upscale_mode,
        reae_filename, transformer_subfolder, prompt_embedding_filename,
    )
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    SwiftVRPipeline = _import_pipeline()
    try:
        pipe = SwiftVRPipeline.from_pretrained(
            str(root),
            reae_filename=reae_filename,
            transformer_subfolder=transformer_subfolder,
            prompt_embedding_filename=prompt_embedding_filename,
            upscale_mode=upscale_mode,
        )
        pipe.to(
            device,
            dtype=dtype,
            attention_backend=attention_backend,
            torch_compile=bool(torch_compile),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load SwiftVR checkpoint from {root}: {exc}") from exc

    _PIPELINE_CACHE[key] = pipe
    return pipe


def clear_pipeline_cache() -> str:
    """Clear cached SwiftVR models and release CUDA memory where available."""
    count = len(_PIPELINE_CACHE)
    _PIPELINE_CACHE.clear()
    gc.collect()
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return f"Cleared {count} cached SwiftVR pipeline(s)."
