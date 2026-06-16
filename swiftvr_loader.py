"""Loading and utility helpers for the ComfyUI SwiftVR nodes."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

_HF_REPO_ID = "H-oliday/SwiftVR"
_AUTO_DIR_NAMES = {"", "auto", "default"}
_PIPELINE_CACHE: dict[tuple[str, str, str, str, bool, str, str, str, str], Any] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_checkpoint_dir() -> Path:
    """Return ComfyUI/models/SwiftVR when ComfyUI is importable."""
    try:
        import folder_paths  # pylint: disable=import-outside-toplevel

        return Path(folder_paths.models_dir) / "SwiftVR"
    except Exception:
        return _repo_root() / "models" / "SwiftVR"


def _missing_checkpoint_paths(
    root: Path,
    reae_filename: str,
    transformer_subfolder: str,
    prompt_embedding_filename: str,
) -> list[str]:
    missing: list[str] = []
    for rel in (reae_filename, prompt_embedding_filename, transformer_subfolder):
        if not (root / rel).exists():
            missing.append(str(root / rel))
    return missing


def _looks_like_comfyui_root(path: Path) -> bool:
    """Detect the common accidental case where an empty widget became ComfyUI cwd."""
    return (path / "custom_nodes").is_dir() and (
        (path / "main.py").exists() or (path / "models").is_dir()
    )


def _download_checkpoint(root: Path) -> None:
    """Download the official SwiftVR checkpoint into the requested folder."""
    try:
        from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SwiftVR checkpoint auto-download requires huggingface_hub. Install this "
            "custom node's requirements.txt with the Python environment that runs ComfyUI."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(root),
            ignore_patterns=["*.md", ".gitattributes"],
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download SwiftVR checkpoint from {_HF_REPO_ID} to {root}: {exc}"
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
    """Validate checkpoint layout, auto-downloading the default checkpoint if requested."""
    raw_checkpoint_dir = (checkpoint_dir or "").strip()
    use_auto_dir = raw_checkpoint_dir.lower() in _AUTO_DIR_NAMES
    root = (_default_checkpoint_dir() if use_auto_dir
            else Path(raw_checkpoint_dir).expanduser()).resolve()

    if use_auto_dir and _missing_checkpoint_paths(
        root, reae_filename, transformer_subfolder, prompt_embedding_filename
    ):
        _download_checkpoint(root)

    # Older workflows / widgets can accidentally hand ComfyUI's root directory
    # to the loader when the field visually looks empty. Do not ask users to put
    # SwiftVR files next to ComfyUI's main.py; redirect to the auto model folder.
    if (
        not use_auto_dir
        and _looks_like_comfyui_root(root)
        and _missing_checkpoint_paths(root, reae_filename, transformer_subfolder, prompt_embedding_filename)
    ):
        root = _default_checkpoint_dir().resolve()
        if _missing_checkpoint_paths(root, reae_filename, transformer_subfolder, prompt_embedding_filename):
            _download_checkpoint(root)

    if not root.exists():
        raise RuntimeError(f"SwiftVR checkpoint directory does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"SwiftVR checkpoint path is not a directory: {root}")

    missing = _missing_checkpoint_paths(
        root, reae_filename, transformer_subfolder, prompt_embedding_filename
    )
    if missing:
        raise RuntimeError(
            "SwiftVR checkpoint is incomplete. Missing required path(s):\n- "
            + "\n- ".join(missing)
            + "\nExpected layout includes reae.safetensors, prompt_embedding.safetensors, "
              "and a transformer/ subfolder unless you changed those inputs. Leave "
              "checkpoint_dir empty or set it to 'auto' to download the official "
              f"{_HF_REPO_ID} checkpoint into {_default_checkpoint_dir()}."
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
