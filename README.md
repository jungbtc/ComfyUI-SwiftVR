# ComfyUI-SwiftVR

<img width="874" height="687" alt="Screenshot 2026-06-16 at 3 59 31 PM" src="https://github.com/user-attachments/assets/2d7318d7-c6ce-4ad6-85f2-2421acbd859a" />

ComfyUI-SwiftVR adds ComfyUI nodes for SwiftVR video restoration. It loads the
SwiftVR checkpoint, accepts ComfyUI's standard `VIDEO` input, restores/upscales
the video, previews the result, writes a sidecar stats file, and clears the model
cache after a run.

## Installation

Clone this repository into your own ComfyUI `custom_nodes` folder, then install
the requirements with the same Python environment that runs ComfyUI.

### Windows

Open a terminal in your ComfyUI folder, then run:

```bat
git clone https://github.com/jungbtc/ComfyUI-SwiftVR.git custom_nodes\ComfyUI-SwiftVR
python -m pip install -r custom_nodes\ComfyUI-SwiftVR\requirements.txt
python main.py
```

If ComfyUI runs from a conda environment, activate it first:

```bat
conda activate comfyui_2026_testbed
```

If ComfyUI runs from a local `.venv`, use that Python instead:

```bat
.venv\Scripts\python.exe -m pip install -r custom_nodes\ComfyUI-SwiftVR\requirements.txt
```

### Linux/macOS

Open a terminal in your ComfyUI folder, then run:

```bash
git clone https://github.com/jungbtc/ComfyUI-SwiftVR.git custom_nodes/ComfyUI-SwiftVR
python -m pip install -r custom_nodes/ComfyUI-SwiftVR/requirements.txt
python main.py
```

Restart ComfyUI after installing the node.

Optional acceleration backends such as `flash_attn_3`, `flash_attn_2`,
`sageattention`, and `xformers` are not installed automatically. Install them
separately only if they match your CUDA, PyTorch, and platform versions.

## Model Setup

The **SwiftVR Model Loader** node can use `checkpoint_dir=auto`. In that mode it
looks for the model in:

```text
ComfyUI/models/SwiftVR/
```

Expected layout:

```text
models/SwiftVR/
  reae.safetensors
  prompt_embedding.safetensors
  transformer/
    ...
```

If the folder is missing, the loader can download the official model from
`H-oliday/SwiftVR` using `huggingface_hub`.

You can also download it manually:

```bash
huggingface-cli download H-oliday/SwiftVR --local-dir models/SwiftVR
```

## Nodes

### SwiftVR Model Loader

Loads the SwiftVR checkpoint and outputs a `SWIFTVR_PIPE`.

Important inputs:

- `checkpoint_dir`: use `auto` for `ComfyUI/models/SwiftVR`, or enter a custom path.
- `device`: usually `cuda`.
- `dtype`: usually `bfloat16` or `float16` on CUDA.
- `attention_backend`: use `auto` or `sdpa` unless you installed another backend.
- `torch_compile`: leave off for first runs.
- `upscale_mode`: interpolation used before SwiftVR restoration.

### Load Video

Use ComfyUI's built-in **Load Video** node from **Basics**. It previews the chosen
video and outputs the standard `VIDEO` type.

Connect its `VIDEO` output directly into **SwiftVR Restore Video**.

### SwiftVR Advanced Options

Optional settings for runtime, upscale, and encoding.

- `upscale`: output scale from `2` to `4`.
- `clip_len`: temporal chunk size; must be a multiple of 4.
- `dit_overlap`: temporal overlap for DiT blending. `0` is the default.
- `fps`: `0` keeps the source FPS.
- `quality`: x265 output quality, 0-100.
- `save_format`: use `yuv444p` only when you specifically need 4:4:4 output.
- `ffmpeg_preset`: optional x265 preset such as `fast` or `medium`.
- `queue_size`: lower this if VRAM is tight.
- `clear_cache_after`: enabled by default to release SwiftVR from VRAM after a run.

### SwiftVR Restore Video

Restores the linked `VIDEO` and writes an MP4.

Inputs:

- `swiftvr_pipe`: connect from **SwiftVR Model Loader**.
- `video`: connect from ComfyUI **Load Video**.
- `options`: optional connection from **SwiftVR Advanced Options**.
- `output_dir`: output folder, relative to the ComfyUI root unless absolute.
- `filename`: output MP4 filename.

Behavior:

- Runs SwiftVR video restoration/upscale.
- Always writes MP4 output.
- Automatically writes a stats JSON next to the MP4.
- Shows the restored video preview inside the node UI when ComfyUI can serve the file.
- Clears the SwiftVR model cache after completion when `clear_cache_after` is enabled.
- Does not expose `output_path` or `stats_json` sockets.

### Legacy Nodes

Legacy nodes are kept only for older workflows:

- **SwiftVR Restore Video Path**
- **SwiftVR Restore Image Batch**

Prefer **Load Video** plus **SwiftVR Restore Video** for new workflows.

### SwiftVR Clear Cache

Manually clears cached SwiftVR pipelines, runs Python garbage collection, and
empties the CUDA cache when available.

## Local Testbed Background

This Windows setup was used while building and smoke-testing the node:

- GPU: NVIDIA GeForce RTX 4090
- NVIDIA driver: 591.86
- VRAM: 24564 MiB, about 24 GB
- System RAM: 63.83 GiB, about 64 GB
- Python: 3.11.15 in the `comfyui_2026_testbed` conda environment
- PyTorch: 2.12.0+cu130
- CUDA runtime reported by PyTorch: 13.0

## Notes

- SwiftVR is VRAM-heavy, especially at 4K output.
- `clip_len` must be a multiple of 4.
- Output frame counts may follow SwiftVR's internal `4k+1` chunk handling.
- x265/MP4 writing requires ffmpeg support; `imageio-ffmpeg` usually provides it.
- CPU execution is useful for wiring tests but is expected to be very slow.
- I basically vibe-coded this to work in my own ComfyUI setup. I’ll spend as many tokens as needed trying to fix your error, but don’t expect too much.
