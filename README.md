# ComfyUI-Hi3D (Sparc3D Refinement)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hi3Dv1.0-blue)](https://huggingface.co/hi3d-official/Hi3Dv1.0)
[![Official Website](https://img.shields.io/badge/Website-hi3d.ai-success)](http://hi3d.ai/)
[![Paper](https://img.shields.io/badge/arXiv-2505.14521-b31b1b)](https://arxiv.org/abs/2505.14521)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)
[![License: Hi3D 1.0](https://img.shields.io/badge/License-Hi3D_Open_1.0-lightgrey.svg)](LICENSE-HI3D)

A ComfyUI custom node that brings **[Hi3Dv1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0)** high-resolution 3D geometry refinement into ComfyUI. 

It takes a coarse 3D mesh (e.g., from **TRELLIS**, **Hunyuan3D**, or any 3D generator) + a reference image and carves out fine surface details, sharp bevels, relief, and micro-geometry at up to **1536³** sparse lattice resolution.

* 🌐 **Hugging Face Model Page:** [huggingface.co/hi3d-official/Hi3Dv1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0)
* 🏠 **Project Homepage:** [hi3d.ai](http://hi3d.ai/)
* 📄 **Technical Report (Sparc3D):** [arxiv.org/abs/2505.14521](https://arxiv.org/abs/2505.14521)

---

## 💻 Installation

### 1. Clone this repository
Navigate to your ComfyUI `custom_nodes` directory and clone this repo:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wadimh/ComfyUI-Hi3D-Refinement.git
cd ComfyUI-Hi3D-Refinement
```

### 2. Install requirements
Install the required packages using your ComfyUI python environment:

**For Portable / Embedded ComfyUI (Windows):**
```bash
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

**For Standard Python / Virtualenv:**
```bash
pip install -r requirements.txt
```

> **Note on CUDA Kernels:** `spconv-cu120`, `flash-attn`, and `triton` require an NVIDIA GPU with compute capability >= 8.0 (RTX 30xx, 40xx, 50xx, A-series). If your main ComfyUI environment uses fixed/locked PyTorch versions, you can point to an isolated python venv by setting:
> ```bash
> set HI3D_PYTHON=C:\path\to\your\venv\Scripts\python.exe
> ```

---

## 📦 Model Download & Placement

Download the weights from the official [Hi3Dv1.0 Hugging Face repository](https://huggingface.co/hi3d-official/Hi3Dv1.0) and place the 3 safetensors files into your standard ComfyUI model directories:

| Model File | Source on Hugging Face | Place in ComfyUI Directory |
| :--- | :--- | :--- |
| **`Hi3D_v1.0_transformer_512.safetensors`** | `transformer_512/model.safetensors` | `ComfyUI/models/diffusion_models/` |
| **`Hi3D_v1.0_transformer_1024.safetensors`** | `transformer_1024/model.safetensors` | `ComfyUI/models/diffusion_models/` |
| **`Hi3D_v1.0_vae.safetensors`** | `vae/model.safetensors` | `ComfyUI/models/vae/` |

*(No extra folders, configs, or licenses needed inside your `models/` directory).*

---

## 🚀 Node Overview & Usage

Find the node under `Hi3D` $\rightarrow$ **`Hi3D Refine Mesh (Sparc3D)`**.

### Inputs
* **`image`**: ComfyUI `IMAGE` conditioning tensor.
* **`mesh`**: Coarse 3D mesh (`TRIMESH`, `MESH`, or `MESHWITHVOXEL` from Trellis, Trellis2, WTiVo, etc.).
* **`transformer_512`**: Model picker dropdown for the 512 pass (selects from `models/diffusion_models/`).
* **`transformer_1024`**: Model picker dropdown for the 1024/1536 detail pass (selects from `models/diffusion_models/`).
* **`vae`**: Model picker dropdown for the sparse VAE (selects from `models/vae/`).
* **`resolution`**: `512` (Pass 1), `1024` (Passes 1+2), or `1536` (All 3 passes, recommended for micro-detail).
* **`steps_512`**: Inference steps for the 512 structural pass (e.g. `24`).
* **`steps_1024`**: Inference steps for the 1024 detail pass (e.g. `16`, active when resolution is 1024 or 1536).
* **`steps_1536`**: Inference steps for the 1536 micro-detail pass (e.g. `16`, active when resolution is 1536).
* **`cpu_offload`**: CPU offloading mode:
  * `auto` *(default)*: Only enables sequential CPU offload on 1536 resolution, keeping 512/1024 fully in VRAM for maximum speed.
  * `enable`: Forcibly enables sequential CPU offload (via Hugging Face Accelerate hooks) to run on lower-VRAM GPUs (e.g., 8–16 GB).
  * `disable`: Forcibly keeps all models resident on the GPU for maximum speed without host RAM swapping (ideal for GPUs with 24 GB+ VRAM like RTX 3090 / 4090 / A5000 / A6000).
* **`seed`**: Random seed.
* **`remove_background`**: Toggle to automatically run `rembg` background cutout before conditioning.
* **`save_to_output`**: Save the final `.glb` directly to `ComfyUI/output/`.

### Outputs
* **`trimesh`**: Standard `trimesh.Trimesh` object (compatible with Trellis2 / 3D preview nodes).
* **`mesh`**: Native ComfyUI `MESH` object with vertex/face tensors.
* **`glb_path`**: File path string to the exported `.glb` mesh.

---

## 🎨 Example Workflow: TRELLIS.2 + Hi3D (Textured) / Tested with 24GB VRAM and 32GB System RAM

An end-to-end example workflow is included in the [`examples/`](file:///X:/5_AI/ComfyUI-trellis/ComfyUI/custom_nodes/ComfyUI-Hi3D/examples) folder:
* **`examples/trellis2_hi3d_textured.json`**

Drag and drop this JSON file directly into ComfyUI. It connects:
1. **Background Removal**: `LoadBackgroundRemovalModel` (BiRefNet) + `ImageCropToMask` (centers subject).
2. **TRELLIS.2 Shape**: Generates the coarse geometry and sparse subdivides.
3. **Hi3D Geometry Refinement**: Refines the raw mesh to carve sharp edge reliefs, engravings, and details.
4. **TRELLIS.2 Texture Stage**: Generates PBR voxel colors from the latent code.
5. **UV Unwrap & Bake**: Unwraps UV charts with `UnwrapMesh` (pec) and bakes `base_color`, `metallic`, and `roughness` using `BakeTextureFromVoxel`.
6. **3D Preview & Save**: Combines texture onto the refined geometry for interactive `Preview3DAdvanced` and `Save3DAdvanced` GLB export.

---

## 📜 Acknowledgments & Licensing

* **ComfyUI Integration Code**: Released under the **[Apache License 2.0](LICENSE)**.
* **Hi3D Materials & Model Weights**: Governed by the **[Hi3D Open License Agreement Version 1.0](LICENSE-HI3D)** Copyright © 2026 OCEANO HK HOLDING LIMITED.
  * Free for research and commercial use up to 200k monthly active users or $1M monthly revenue.
  * For full terms and notices, see [NOTICE](NOTICE) and [LICENSE-HI3D](LICENSE-HI3D).
* Official Hugging Face Landing Page: [https://huggingface.co/hi3d-official/Hi3Dv1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0)
* Official Website: [http://hi3d.ai/](http://hi3d.ai/)
* Based on [Hi3Dv1.0](https://github.com/hi3d-official/Hi3Dv1.0) and [Sparc3D](https://arxiv.org/abs/2505.14521).
* *Note: This is an unofficial community custom node, not affiliated with or endorsed by OCEANO HK HOLDING LIMITED.*
