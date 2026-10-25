#!/usr/bin/env python3
"""Standalone inference worker for ComfyUI-Hi3D."""

import argparse
import sys
import os
import warnings
from pathlib import Path
import trimesh
from PIL import Image

# Suppress known PyTorch / spconv deprecation FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*spconv.*")
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.custom_.*")

# Add custom node directory to sys.path so sparc3d is imported locally
node_dir = Path(__file__).resolve().parent
if str(node_dir) not in sys.path:
    sys.path.insert(0, str(node_dir))

# Ensure CUDA device
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def parse_args():
    parser = argparse.ArgumentParser(description="ComfyUI-Hi3D Worker")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--coarse-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path, default=node_dir / "configs")
    parser.add_argument("--transformer-512", type=Path, default=None)
    parser.add_argument("--transformer-1024", type=Path, default=None)
    parser.add_argument("--vae", type=Path, default=None)
    parser.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=1536)
    parser.add_argument("--steps-512", type=int, default=24, help="Steps for 512 pass")
    parser.add_argument("--steps-1024", type=int, default=16, help="Steps for 1024 pass")
    parser.add_argument("--steps-1536", type=int, default=16, help="Steps for 1536 pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cpu-offload", type=str, choices=("auto", "enable", "disable"), default="auto", help="CPU offloading mode: auto, enable, or disable")
    parser.add_argument("--sequential-cpu-offload", action="store_true", help="Deprecated alias for --cpu-offload enable")
    parser.add_argument("--remove-background", action="store_true")
    return parser.parse_args()


def progress_tracker(event):
    print(f"  [Pass {event.pass_index + 1}/{event.num_passes}] Step {event.step}/{event.num_inference_steps}", flush=True)


def main():
    args = parse_args()

    # Pass explicit model picker selections via environment overrides
    if args.transformer_512 and args.transformer_512.is_file():
        os.environ["HI3D_TRANSFORMER_512"] = str(args.transformer_512)
    if args.transformer_1024 and args.transformer_1024.is_file():
        os.environ["HI3D_TRANSFORMER_1024"] = str(args.transformer_1024)
    if args.vae and args.vae.is_file():
        os.environ["HI3D_VAE"] = str(args.vae)

    from sparc3d import Sparc3DPipeline

    image = Image.open(args.image)
    if args.remove_background:
        from sparc3d.utils.image import remove_background
        image = remove_background(image)

    coarse_mesh = trimesh.load(args.coarse_mesh, force="mesh", process=False)

    pipeline = Sparc3DPipeline.from_pretrained(
        args.configs,
        local_files_only=True,
        progress_callback=progress_tracker,
    )

    if args.cpu_offload == "enable":
        should_offload = True
    elif args.cpu_offload == "disable":
        should_offload = False
    else:  # auto
        should_offload = (args.resolution >= 1536)

    if args.sequential_cpu_offload:
        should_offload = True

    if should_offload:
        print("[Hi3D] Sequential CPU offload: ENABLED", flush=True)
        pipeline.enable_sequential_cpu_offload(args.device)
    else:
        print("[Hi3D] Sequential CPU offload: DISABLED (full GPU residency)", flush=True)
        pipeline.to(args.device)

    result = pipeline(
        image,
        coarse_mesh,
        resolution=args.resolution,
        seed=args.seed,
        steps_512=args.steps_512,
        steps_1024=args.steps_1024,
        steps_1536=args.steps_1536,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(str(args.output))
    print(f"Refinement complete -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
