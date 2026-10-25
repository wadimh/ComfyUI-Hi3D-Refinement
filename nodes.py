import os
import sys
import subprocess
import uuid
from pathlib import Path
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None

NODE_DIR = Path(__file__).resolve().parent


class AnyType(str):
    """ComfyUI wildcard type to allow connecting MESH, TRIMESH, or file paths."""
    def __ne__(self, __value: object) -> bool:
        return False

any_mesh_type = AnyType("*")


def tensor2pil(image_tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor [B, H, W, C] to PIL Image."""
    if torch.is_tensor(image_tensor):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        i = 255.0 * image_tensor.cpu().numpy()
        img_np = np.clip(i, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    elif isinstance(image_tensor, Image.Image):
        return image_tensor
    elif isinstance(image_tensor, np.ndarray):
        return Image.fromarray(image_tensor)
    raise TypeError(f"Unsupported image type: {type(image_tensor)}")


def extract_or_save_mesh(mesh_input, temp_mesh_path: Path):
    """Extract mesh from ComfyUI MESH, TRIMESH, or MESHWITHVOXEL to a GLB file."""
    import trimesh

    if isinstance(mesh_input, (str, Path)):
        p = Path(mesh_input)
        if p.exists():
            return p

    # If it is a trimesh.Trimesh or trimesh.Scene
    if hasattr(mesh_input, "export") and callable(mesh_input.export):
        mesh_input.export(str(temp_mesh_path))
        return temp_mesh_path

    # If it's MESHWITHVOXEL or dict-like
    if isinstance(mesh_input, dict):
        if "mesh" in mesh_input:
            return extract_or_save_mesh(mesh_input["mesh"], temp_mesh_path)
        if "trimesh" in mesh_input:
            return extract_or_save_mesh(mesh_input["trimesh"], temp_mesh_path)

    # Check for ComfyUI native MESH object with .vertices and .faces
    vertices = getattr(mesh_input, "vertices", None)
    faces = getattr(mesh_input, "faces", None)

    if vertices is not None and faces is not None:
        if torch.is_tensor(vertices):
            if vertices.ndim == 3:
                vertices = vertices[0]
            v_np = vertices.detach().cpu().numpy()
        else:
            v_np = np.asarray(vertices)

        if torch.is_tensor(faces):
            if faces.ndim == 3:
                faces = faces[0]
            f_np = faces.detach().cpu().numpy()
        else:
            f_np = np.asarray(faces)

        tm = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
        tm.export(str(temp_mesh_path))
        return temp_mesh_path

    raise TypeError(f"Could not extract 3D mesh from input of type: {type(mesh_input)}")


def resolve_python_interpreter() -> Path:
    """Find the Python interpreter with Hi3D dependencies installed."""
    if "HI3D_PYTHON" in os.environ and Path(os.environ["HI3D_PYTHON"]).exists():
        return Path(os.environ["HI3D_PYTHON"])

    for local_cand in [
        NODE_DIR / ".venv" / "Scripts" / "python.exe",
        NODE_DIR / ".venv" / "bin" / "python",
    ]:
        if local_cand.exists():
            return local_cand

    # Check for relative sister virtualenv if cloned alongside Hi3D
    try:
        # e.g., <workspace>/Hi3D/.venv
        workspace_dir = NODE_DIR.parents[2]
        for sister_cand in [
            workspace_dir / "Hi3D" / ".venv" / "Scripts" / "python.exe",
            workspace_dir / "Hi3D" / ".venv" / "bin" / "python",
        ]:
            if sister_cand.exists():
                return sister_cand
    except Exception:
        pass

    return Path(sys.executable)


class Hi3DRefineMesh:
    """ComfyUI Node for Hi3Dv1.0 3D Mesh Refinement (Sparc3D).

    Refines a coarse 3D mesh conditioned on an image into high-resolution geometry.
    Supports model pickers for all 3 model files:
      - transformer_512
      - transformer_1024
      - vae
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Query ComfyUI's model registry for available models
        diffusion_list = []
        vae_list = []
        if folder_paths:
            diffusion_list = folder_paths.get_filename_list("diffusion_models")
            vae_list = folder_paths.get_filename_list("vae")

        # Smart defaults matching Hi3D filenames
        def_tf_512 = next(
            (m for m in diffusion_list if "512" in m and "hi3d" in m.lower()),
            diffusion_list[0] if diffusion_list else "Hi3D_v1.0_transformer_512.safetensors"
        )
        def_tf_1024 = next(
            (m for m in diffusion_list if "1024" in m and "hi3d" in m.lower()),
            diffusion_list[0] if diffusion_list else "Hi3D_v1.0_transformer_1024.safetensors"
        )
        def_vae = next(
            (m for m in vae_list if "hi3d" in m.lower() or "vae" in m.lower()),
            vae_list[0] if vae_list else "Hi3D_v1.0_vae.safetensors"
        )

        return {
            "required": {
                "image": ("IMAGE",),
                "mesh": (any_mesh_type, {"tooltip": "Coarse 3D mesh (TRIMESH or MESH from Trellis, WTiVo, etc.)"}),
                "transformer_512": (diffusion_list, {"default": def_tf_512, "tooltip": "Pass 1 structural transformer (512)"}),
                "transformer_1024": (diffusion_list, {"default": def_tf_1024, "tooltip": "Passes 2 & 3 detail transformer (1024 / 1536)"}),
                "vae": (vae_list, {"default": def_vae, "tooltip": "Sparse VAE decoder"}),
                "resolution": (["512", "1024", "1536"], {"default": "1024"}),
                "steps_512": ("INT", {"default": 24, "min": 1, "max": 100, "step": 1, "tooltip": "Inference steps for 512 pass"}),
                "steps_1024": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1, "tooltip": "Inference steps for 1024 pass (active for 1024 and 1536)"}),
                "steps_1536": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1, "tooltip": "Inference steps for 1536 pass (active for 1536)"}),
                "cpu_offload": (["auto", "enable", "disable"], {
                    "default": "auto",
                    "tooltip": "Sequential CPU offload mode. 'auto': offload only for 1536 resolution; 'enable': always offload to save VRAM; 'disable': keep all models on GPU for max speed (needs 24GB+ VRAM)."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "remove_background": ("BOOLEAN", {"default": False, "tooltip": "Remove background with rembg before conditioning"}),
            },
            "optional": {
                "save_to_output": ("BOOLEAN", {"default": True, "tooltip": "Save the refined GLB to the ComfyUI output folder"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESH", "STRING")
    RETURN_NAMES = ("trimesh", "mesh", "glb_path")
    FUNCTION = "refine"
    CATEGORY = "Hi3D"
    OUTPUT_NODE = True

    def refine(
        self,
        image,
        mesh,
        transformer_512,
        transformer_1024,
        vae,
        resolution="1024",
        steps_512=24,
        steps_1024=16,
        steps_1536=16,
        cpu_offload="auto",
        seed=0,
        remove_background=False,
        save_to_output=True,
    ):
        import trimesh

        # Determine temp & output directories dynamically
        if folder_paths:
            temp_dir = Path(folder_paths.get_temp_directory())
            out_dir = Path(folder_paths.get_output_directory())
            path_tf_512 = folder_paths.get_full_path("diffusion_models", transformer_512)
            path_tf_1024 = folder_paths.get_full_path("diffusion_models", transformer_1024)
            path_vae = folder_paths.get_full_path("vae", vae)
        else:
            temp_dir = NODE_DIR / "temp"
            out_dir = NODE_DIR / "output"
            path_tf_512 = None
            path_tf_1024 = None
            path_vae = None

        temp_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex[:8]
        temp_img_path = temp_dir / f"hi3d_in_{run_id}.png"
        temp_mesh_path = temp_dir / f"hi3d_coarse_{run_id}.glb"

        if save_to_output:
            final_out_path = out_dir / f"Hi3D_refined_{run_id}_{resolution}.glb"
        else:
            final_out_path = temp_dir / f"Hi3D_refined_{run_id}_{resolution}.glb"

        # 1. Save input image
        pil_img = tensor2pil(image)
        pil_img.save(temp_img_path)

        # 2. Extract input mesh
        actual_mesh_path = extract_or_save_mesh(mesh, temp_mesh_path)

        # 3. Locate worker script and python executable
        python_exe = resolve_python_interpreter()
        worker_script = NODE_DIR / "worker.py"
        configs_dir = NODE_DIR / "configs"

        cmd = [
            str(python_exe),
            str(worker_script),
            "--image", str(temp_img_path),
            "--coarse-mesh", str(actual_mesh_path),
            "--output", str(final_out_path),
            "--configs", str(configs_dir),
            "--resolution", str(resolution),
            "--steps-512", str(steps_512),
            "--steps-1024", str(steps_1024),
            "--steps-1536", str(steps_1536),
            "--cpu-offload", str(cpu_offload),
            "--seed", str(seed),
        ]

        if path_tf_512:
            cmd.extend(["--transformer-512", str(path_tf_512)])
        if path_tf_1024:
            cmd.extend(["--transformer-1024", str(path_tf_1024)])
        if path_vae:
            cmd.extend(["--vae", str(path_vae)])

        if remove_background:
            cmd.append("--remove-background")

        print(f"[Hi3D] Selected models:\n  - 512: {transformer_512}\n  - 1024: {transformer_1024}\n  - VAE: {vae}", flush=True)
        print(f"[Hi3D] Launching refinement: res={resolution}, steps=[512:{steps_512}, 1024:{steps_1024}, 1536:{steps_1536}], offload={cpu_offload}, seed={seed}...", flush=True)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(NODE_DIR),
        )

        for line in process.stdout:
            print(f"[Hi3D] {line.strip()}", flush=True)

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Hi3D refinement process failed with exit code {process.returncode}")

        if not final_out_path.exists():
            raise FileNotFoundError(f"Hi3D did not produce output at: {final_out_path}")

        # Load result
        refined_trimesh = trimesh.load(str(final_out_path), force="mesh", process=False)
        print(f"[Hi3D] Successfully refined: {len(refined_trimesh.vertices):,} vertices, {len(refined_trimesh.faces):,} faces", flush=True)

        # Construct ComfyUI native MESH output
        output_native_mesh = None
        try:
            v_t = torch.as_tensor(refined_trimesh.vertices, dtype=torch.float32).unsqueeze(0)
            f_t = torch.as_tensor(refined_trimesh.faces, dtype=torch.int32).unsqueeze(0)
            try:
                from comfy_api.latest import Types
                output_native_mesh = Types.MESH(vertices=v_t, faces=f_t)
            except Exception:
                mesh_class = type(mesh)
                if hasattr(mesh_class, "__init__") and hasattr(mesh, "vertices"):
                    output_native_mesh = mesh_class(vertices=v_t, faces=f_t)
                else:
                    output_native_mesh = {"vertices": v_t, "faces": f_t}
        except Exception as e:
            print(f"[Hi3D] Warning creating native MESH: {e}", flush=True)
            output_native_mesh = refined_trimesh

        # Cleanup temp inputs
        try:
            if temp_img_path.exists():
                temp_img_path.unlink()
            if temp_mesh_path.exists() and temp_mesh_path != actual_mesh_path:
                temp_mesh_path.unlink()
        except Exception:
            pass

        return (refined_trimesh, output_native_mesh, str(final_out_path))
