# ComfyUI-SwiftVR

ComfyUI-SwiftVR adds classic ComfyUI custom nodes for the included [`swiftvr`](./swiftvr) Python package. It provides nodes to load SwiftVR checkpoints, restore a video/image-folder path, restore a ComfyUI `IMAGE` batch, and clear the model cache.

## Installation

### Option 1: ComfyUI Manager (Recommended)

Use this option once the node is available in ComfyUI Manager / the ComfyUI Registry.

1. Open **ComfyUI Manager** in your ComfyUI interface.
2. Click **Custom Nodes Manager**.
3. Search for **ComfyUI-SwiftVR**.
4. Click **Install**.
5. Restart ComfyUI.

Registry Link: [ComfyUI Registry - ComfyUI-SwiftVR](https://registry.comfy.org/nodes/ComfyUI-SwiftVR)

### Option 2: Manual Installation

Clone the repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI
git clone https://github.com/H-oliday/ComfyUI-SwiftVR.git custom_nodes/ComfyUI-SwiftVR
```

Install dependencies using the standalone Python environment that runs ComfyUI.

**Windows:**

```bat
.venv\Scripts\python.exe -m pip install -r custom_nodes\ComfyUI-SwiftVR\requirements.txt
```

**Linux/macOS:**

```bash
.venv/bin/python -m pip install -r custom_nodes/ComfyUI-SwiftVR/requirements.txt
```

Restart ComfyUI after installation.

> Optional acceleration backends such as `flash_attn_3`, `flash_attn_2`, `sageattention`, and `xformers` are not installed automatically. Install them separately only if they match your CUDA, PyTorch, and platform versions.

## Model Installation

SwiftVR checkpoints are loaded from the path you enter in the **SwiftVR Model Loader** node. They do not have to be placed in a specific ComfyUI model folder.

Recommended location:

```text
ComfyUI/models/SwiftVR/
  reae.safetensors
  prompt_embedding.safetensors
  transformer/
    ...
```

You can also store the checkpoint anywhere else and paste that folder path into `checkpoint_dir`.

Download the official SwiftVR model from:

- [H-oliday/SwiftVR on Hugging Face](https://huggingface.co/H-oliday/SwiftVR)

Example download with the Hugging Face CLI:

```bash
huggingface-cli download H-oliday/SwiftVR --local-dir models/SwiftVR
```

If you run that command from the `ComfyUI` directory, use this loader path:

```text
models/SwiftVR
```

## Expected Checkpoint Layout

The default loader inputs expect this structure:

```text
checkpoints/
  reae.safetensors
  prompt_embedding.safetensors
  transformer/
    ...
```

If your files use different names, change these inputs on **SwiftVR Model Loader**:

- `reae_filename`
- `prompt_embedding_filename`
- `transformer_subfolder`

## Nodes

### SwiftVR Model Loader

Loads a SwiftVR checkpoint and outputs a reusable `SWIFTVR_PIPE` object. Loaded pipelines are cached by checkpoint path, device, dtype, attention backend, compile setting, upscale mode, and checkpoint filenames.

Main inputs:

- `checkpoint_dir`: path to the SwiftVR checkpoint directory.
- `device`: `cuda` or `cpu`.
- `dtype`: `bfloat16`, `float16`, or `float32`.
- `attention_backend`: `auto`, `sdpa`, `flash_attn_3`, `flash_attn_2`, `sageattention`, or `xformers`.
- `torch_compile`: enable only after confirming your environment supports it.

### SwiftVR Restore Video Path

Restores a video file or image-folder path and writes the result to disk.

- `resolution`: leave empty for upscale mode, or enter a value such as `1920x1080` or `3840x2160`.
- `fps`: set to `0` to keep the source fps.
- `png_save`: write PNG frames instead of only a video output.
- Outputs: `output_path` and formatted `stats_json`.

### SwiftVR Restore Image Batch

Restores a ComfyUI `IMAGE` batch by writing temporary PNG frames, running SwiftVR, and reading the restored PNG frames back into ComfyUI.

- Input tensor format is the normal ComfyUI image format: `[B,H,W,C]`, float `0..1`.
- `keep_temp=false` removes temporary folders after loading frames back into ComfyUI.
- `keep_temp=true` leaves temporary files on disk for debugging.
- The node returns all PNG frames produced by SwiftVR.

### SwiftVR Clear Cache

Clears the global SwiftVR pipeline cache, runs Python garbage collection, and empties the CUDA cache when available.

## Example Workflows

### Video or image-folder restoration

```text
SwiftVR Model Loader -> SwiftVR Restore Video Path
```

1. Set `checkpoint_dir` in **SwiftVR Model Loader**.
2. Connect `swiftvr_pipe` to **SwiftVR Restore Video Path**.
3. Set `input_path` to a video file or image folder.
4. Set `output_path` to an `.mp4` file path or an output folder.
5. Queue the workflow.

### ComfyUI image-batch restoration

```text
Load Image / video frames -> SwiftVR Restore Image Batch
SwiftVR Model Loader -----> SwiftVR Restore Image Batch
```

Use `resolution` for an exact output size, or leave it empty and set `upscale`.

## Known Limitations

- SwiftVR is very VRAM-heavy at 1080p and especially at 4K.
- `clip_len` must be a multiple of 4.
- Output frame counts may follow SwiftVR's internal `4k+1` chunk handling.
- Optional attention backends require separate installation.
- x265/MP4 writing requires ffmpeg support; `imageio-ffmpeg` usually provides a bundled ffmpeg binary.
- CPU execution is useful for testing wiring but is expected to be very slow.

## Troubleshooting

- **Checkpoint missing:** verify that `reae.safetensors`, `prompt_embedding.safetensors`, and `transformer/` exist under `checkpoint_dir`.
- **Import errors:** install `requirements.txt` with the same Python executable that launches ComfyUI.
- **Optional backend errors:** switch `attention_backend` to `auto` or `sdpa`.
- **Out of memory:** lower `resolution`, reduce `clip_len`, close other GPU processes, or try `float16`/`bfloat16` on CUDA.
