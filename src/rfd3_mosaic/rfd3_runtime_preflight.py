"""Checkpoint-free execution of the native RFD3 input feature pipeline.

This validates software/feature compatibility, not a network forward pass,
memory requirements, secondary structure, or protein designability.
"""

import copy
import hashlib
import json
import os
import random
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def isolated_preflight_random_state():
    """Do not let diagnostic feature/noise generation alter a task's RNG."""

    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    seed_environment = {
        key: os.environ.get(key) for key in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS")
    }
    try:
        # CPU-only preflight must not initialize CUDA just to snapshot its RNG.
        with torch.random.fork_rng(devices=[]):
            random.seed(0)
            np.random.seed(0)
            torch.random.default_generator.manual_seed(0)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for key, value in seed_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def source_inference_training_config():
    """Read shipped inference settings, without instantiating a model.

    Use the actual validation transform rather than passing all training
    transform settings into inference (which silently changes augmentation).
    The source checkout and the built wheel keep configs in different places.
    """

    import rfd3
    import yaml
    from omegaconf import OmegaConf

    package_root = Path(rfd3.__file__).resolve().parent
    candidates = (package_root / "configs", package_root.parent.parent / "configs")
    config_root = next(
        (path for path in candidates if (path / "datasets/design_base.yaml").is_file()),
        None,
    )
    if config_root is None:
        raise FileNotFoundError(
            "RFD3 inference configs are unavailable; reinstall the package or "
            "supply resolved_training_config for runtime feature preflight"
        )

    def read(relative):
        return yaml.safe_load((config_root / relative).read_text(encoding="utf-8"))

    datasets = read("datasets/design_base.yaml")
    datasets["val"] = {
        "source_inference": read("datasets/val/design_validation_base.yaml")
    }
    return OmegaConf.create(
        {
            "datasets": datasets,
            "model": {"net": read("model/components/rfd3_net.yaml")},
            "paths": {"data": read("paths/data/default.yaml")},
        }
    )


def resolve_preflight_pipeline(resolved_training_config=None):
    """Resolve transforms using the same helper as BaseInferenceEngine."""

    import hydra
    from omegaconf import OmegaConf

    from foundry.inference_engines.base import resolve_inference_transform_config

    if resolved_training_config is None:
        config = source_inference_training_config()
        source = "shipped_source_configuration"
    else:
        config = (
            OmegaConf.load(resolved_training_config)
            if isinstance(resolved_training_config, (str, Path))
            else resolved_training_config
        )
        source = "provided_training_configuration"
    dataset_name, transform = resolve_inference_transform_config(
        config, {"diffusion_batch_size": 1}
    )
    resolved = OmegaConf.to_container(transform, resolve=True)
    if not isinstance(resolved, dict) or resolved.get("is_inference") is not True:
        raise ValueError("Runtime preflight requires an inference transform")
    if (
        resolved.get("_target_")
        != "rfd3.transforms.pipelines.build_atom14_base_pipeline"
    ):
        raise ValueError(
            "Runtime preflight currently supports the native RFD3 Atom14 pipeline"
        )
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hydra.utils.instantiate(transform), {
        "configuration_source": source,
        "validation_dataset": dataset_name,
        "transform_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "resolved_transform": resolved,
        "diffusion_batch_size": 1,
        "checkpoint_loaded": False,
        "checkpoint_compatibility_validated": False,
        "model_forward_validated": False,
    }


def default_preflight_sampler(raw_spec: dict[str, Any]) -> dict[str, Any]:
    """The public exact-symmetry workflow, inferred only when none is supplied.

    These defaults are a dispatch check, not a claim to have inspected a future
    invocation's flags. A worker/caller with effective overrides can supply them.
    """

    extra = raw_spec.get("extra") or {}
    return {
        "kind": "symmetry",
        "symmetry_state_mode": "orbit_average",
        "symmetry_noise_mode": "coupled",
        "symmetry_execution_backend": "explicit_all_copy",
        "preserve_fixed_motif_during_symmetry": True,
        "enable_orbit_rigid_motif_mobility": any(
            orbit.get("mobility_mode") == "orbit_rigid"
            for orbit in extra.get("motif_constraint_orbits", [])
        ),
    }


def audit_native_feature_pipeline(atom_array, metadata, example_id, pipeline):
    """Run the native post-build parser and all inference transforms once.

    ``DesignInputSpecification.to_pipeline_input`` uses this same parser.
    Reuse the already-built array so a ranged contig is not sampled twice and
    the semantic audit and feature audit refer to exactly the same structure.
    """

    import numpy as np
    import torch
    from rfd3.inference.input_parsing import prepare_pipeline_input_from_atom_array

    data = prepare_pipeline_input_from_atom_array(atom_array.copy())
    data["example_id"] = example_id
    data["specification"] = copy.deepcopy(metadata)
    data["specification"].setdefault("extra", {})["example_id"] = example_id
    data = pipeline(data)
    features = data.get("feats")
    if not isinstance(features, Mapping) or not features:
        raise ValueError("Native inference pipeline produced no model features")
    nonfinite = []

    def inspect(value, path):
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()
            ):
                nonfinite.append(path)
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if not np.isfinite(value).all():
                nonfinite.append(path)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                inspect(item, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                inspect(item, f"{path}[{index}]")

    for key in ("feats", "coord_atom_lvl_to_be_noised", "noise", "t"):
        if key not in data:
            raise ValueError(f"Native inference pipeline omitted {key}")
        inspect(data[key], key)
    if nonfinite:
        raise ValueError(
            "Native inference features contain NaN or Inf: " + ", ".join(nonfinite)
        )
    return {
        "passed": True,
        "padded_atom_count": len(data["atom_array"]),
        "feature_names": sorted(features),
        "tensor_shapes": {
            name: list(value.shape)
            for name, value in features.items()
            if isinstance(value, (torch.Tensor, np.ndarray))
        },
        "symmetry_orbit_slots_verified": bool(
            features.get("sym_orbit_slot_verified", False)
        ),
        "fixed_coordinate_atom_count": int(
            data["atom_array"].is_motif_atom_with_fixed_coord.astype(bool).sum()
        ),
    }
