from __future__ import annotations
"""Safe local and Hugging Face artifact loading for the public pipeline."""

import json
import os
from pathlib import Path
from typing import Any, Mapping

_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/model.safetensors",
    "transformer_512/config.json",
    "transformer_512/model.safetensors",
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
)

_TRANSFORMER_1024_FILES = (
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
)

_INITIAL_REQUIRED_FILES = tuple(
    path for path in _REQUIRED_FILES if path not in _TRANSFORMER_1024_FILES
)

_UNSAFE_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"}
_DEFAULT_SCHEDULER_CLASS = "FlowMatchHeunScheduler"
_INDEX_PATHS = {
    "scheduler": "scheduler",
    "vae": "vae",
    "transformer_512": "transformer_512",
    "transformer_1024": "transformer_1024",
}


def _find_comfy_model(folder_type: str, filename: str) -> Path | None:
    try:
        import folder_paths
        for d in folder_paths.get_folder_paths(folder_type):
            p = Path(d) / filename
            if p.is_file():
                return p
    except Exception:
        pass
    try:
        base_models = Path(__file__).resolve().parents[4] / "models"
        p = base_models / folder_type / filename
        if p.is_file():
            return p
    except Exception:
        pass
    return None


def resolve_model_directory(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
    defer_transformer_1024: bool = False,
) -> Path:
    source = str(pretrained_model_name_or_path)
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        root = candidate.resolve()
        required_files = _REQUIRED_FILES
    elif candidate.exists():
        raise ValueError(f"model source must be a directory, got {candidate}")
    elif source.startswith("modelscope://"):
        raise ValueError("ModelScope network loading is unsupported; pass a downloaded local directory")
    elif candidate.is_absolute() or source.startswith((".", "~")):
        raise FileNotFoundError(f"local model directory does not exist: {candidate}")
    else:
        from huggingface_hub import snapshot_download

        required_files = (
            _INITIAL_REQUIRED_FILES if defer_transformer_1024 else _REQUIRED_FILES
        )
        root = Path(
            snapshot_download(
                repo_id=source,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
                allow_patterns=list(required_files),
            )
        ).resolve()

    _validate_repository(root, required_files=required_files)
    return root


def load_deferred_transformer_1024(
    root: Path,
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Any:
    env_tf1024 = os.environ.get("HI3D_TRANSFORMER_1024")
    if env_tf1024 and Path(env_tf1024).is_file():
        return load_transformer(root, "transformer_1024")

    has_local_1024 = (
        (root / "transformer_1024" / "config.json").is_file()
        and (
            (root / "transformer_1024" / "model.safetensors").is_file()
            or _find_comfy_model("diffusion_models", "Hi3D_v1.0_transformer_1024.safetensors")
            or _find_comfy_model("diffusion_models", "hi3d_transformer_1024.safetensors")
        )
    )
    if not has_local_1024:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=_resolved_snapshot_revision(root, revision),
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            allow_patterns=list(_TRANSFORMER_1024_FILES),
        )
    return load_transformer(root, "transformer_1024")


def load_scheduler(root: Path) -> Any:
    from sparc3d.schedulers import FlowMatchHeunScheduler

    schedulers = {
        "FlowMatchHeunScheduler": FlowMatchHeunScheduler,
    }
    values = _read_json(root / "scheduler" / "scheduler_config.json")
    class_name = values.pop("_class_name", _DEFAULT_SCHEDULER_CLASS)
    if class_name not in schedulers:
        raise ValueError(f"unsupported scheduler class {class_name!r}")
    allowed = {"num_inference_steps", "sigma_min", "time_rescale_factor"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scheduler config keys: {unknown}")
    return schedulers[class_name](**values)


def load_conditioner(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    dino_model_id: str | Path | None = None,
    clip_model_id: str | Path | None = None,
) -> Any:
    from sparc3d.models.conditioners.image import ImageConditioner

    overrides: dict[str, str | Path] = {}
    if dino_model_id is not None:
        overrides["dino_model_id"] = dino_model_id
    if clip_model_id is not None:
        overrides["clip_model_id"] = clip_model_id

    return (
        ImageConditioner(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **overrides,
        )
        .to("cpu")
        .eval()
    )


def load_transformer(root: Path, artifact_name: str) -> Any:
    if artifact_name not in {"transformer_512", "transformer_1024"}:
        raise ValueError(f"unsupported transformer artifact {artifact_name!r}")
    from safetensors.torch import load_file
    from sparc3d.models.transformers import (
        Sparc3DTransformer,
        Sparc3DTransformerConfig,
    )

    artifact = root / artifact_name
    config = Sparc3DTransformerConfig.from_dict(_read_json(artifact / "config.json"))
    model = Sparc3DTransformer(config)

    # Check environment override first (from model picker)
    env_override = os.environ.get(f"HI3D_{artifact_name.upper()}")
    if env_override and Path(env_override).is_file():
        weights_path = Path(env_override)
    else:
        weights_path = artifact / "model.safetensors"
        if not weights_path.is_file():
            found = (
                _find_comfy_model("diffusion_models", f"Hi3D_v1.0_{artifact_name}.safetensors")
                or _find_comfy_model("diffusion_models", f"hi3d_{artifact_name}.safetensors")
            )
            if found:
                weights_path = found

    model.load_state_dict(
        load_file(str(weights_path), device="cpu"), strict=True
    )
    return model.to("cpu").eval()


def load_vae(root: Path) -> Any:
    from sparc3d.models.autoencoders.config import load_sparconv_vae

    env_vae = os.environ.get("HI3D_VAE")
    if env_vae and Path(env_vae).is_file():
        weights_path = Path(env_vae)
    else:
        weights_path = root / "vae" / "model.safetensors"
        if not weights_path.is_file():
            found = (
                _find_comfy_model("vae", "Hi3D_v1.0_vae.safetensors")
                or _find_comfy_model("vae", "hi3d_vae.safetensors")
            )
            if found:
                weights_path = found

    config_path = root / "vae" / "config.json"
    return load_sparconv_vae(weights_path, config=config_path, device="cpu")


def _validate_repository(
    root: Path, *, required_files: tuple[str, ...] = _REQUIRED_FILES
) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    required_paths = {Path(relative) for relative in required_files}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink() and relative not in required_paths:
            raise ValueError(f"model repository has an unexpected symlink: {path}")
        if path.is_file() and path.suffix.lower() in _UNSAFE_SUFFIXES:
            raise ValueError(f"unsafe checkpoint file is not permitted: {path}")

    for relative in required_files:
        path = root / relative
        if not path.is_file():
            if "transformer_" in relative and relative.endswith(".safetensors"):
                art_name = relative.split("/")[0]
                if (
                    os.environ.get(f"HI3D_{art_name.upper()}")
                    or _find_comfy_model("diffusion_models", f"Hi3D_v1.0_{art_name}.safetensors")
                    or _find_comfy_model("diffusion_models", f"hi3d_{art_name}.safetensors")
                ):
                    continue
            elif "vae" in relative and relative.endswith(".safetensors"):
                if (
                    os.environ.get("HI3D_VAE")
                    or _find_comfy_model("vae", "Hi3D_v1.0_vae.safetensors")
                    or _find_comfy_model("vae", "hi3d_vae.safetensors")
                ):
                    continue
            raise FileNotFoundError(f"required model artifact is missing: {path}")

    index = _read_json(root / "model_index.json")
    allowed = {"_class_name", "_format_version", "_sparc3d_version", *_INDEX_PATHS}
    unknown = sorted(set(index) - allowed)
    if unknown:
        raise ValueError(f"unknown model index keys: {unknown}")
    class_name = index.get("_class_name", "Sparc3DPipeline")
    if class_name != "Sparc3DPipeline":
        raise ValueError(f"unsupported pipeline class {class_name!r}")
    for key, expected in _INDEX_PATHS.items():
        if key in index and index[key] != expected:
            raise ValueError(f"model index entry {key!r} must be the fixed path {expected!r}")
    for key in ("_format_version", "_sparc3d_version"):
        if key in index and not isinstance(index[key], (str, int, float)):
            raise ValueError(f"model index metadata {key!r} must be scalar")


def _resolved_snapshot_revision(root: Path, requested_revision: str | None) -> str | None:
    parent = root.parent
    if parent.name == "snapshots":
        return root.name
    return requested_revision


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(values, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return dict(values)
