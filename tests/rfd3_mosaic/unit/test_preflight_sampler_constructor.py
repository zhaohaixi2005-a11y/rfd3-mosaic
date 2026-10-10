"""Actual sampler construction must fail on CPU before weights or GPU are used."""
import json
from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf
from rfd3.model.inference_sampler import ConditionalDiffusionSampler

from rfd3_mosaic.experiment_worker import (
    _sampler_dispatch_overrides,
    _sampler_inference_arguments,
)
from rfd3_mosaic.output import compile_rfd3_input
from rfd3_mosaic.rfd3_prevalidate import (
    _resolve_sampler_constructor_config,
    prevalidate_rfd3_input,
)
from rfd3_mosaic.rfd3_runtime_preflight import source_inference_training_config

ROOT = Path(__file__).resolve().parents[3]
EXACT = {
    "kind": "symmetry",
    "symmetry_state_mode": "orbit_average",
    "symmetry_noise_mode": "coupled",
    "preserve_fixed_motif_during_symmetry": True,
}


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    return compile_rfd3_input(
        ROOT / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml",
        tmp_path_factory.mktemp("sampler-constructor"),
        base_directory=ROOT,
    ).input_path


@pytest.mark.parametrize("overrides,message", [
    ({"enable_orbit_rigid_motif_mobility": True,
      "motif_mobility_proposal_source": "denoiser",
      "enable_assembly_robust_capture": True, "assembly_capture_weight": 1.0},
     "Assembly robust capture requires motif_mobility_proposal_source=scaffold_boundary"),
    ({"enable_orbit_rigid_motif_mobility": True,
      "motif_mobility_proposal_source": "denoiser"},
     "no learned pose signal"),
    ({"enable_orbit_rigid_motif_mobility": True,
      "motif_mobility_proposal_source": "scaffold_boundary",
      "symmetry_execution_backend": "local_neighbourhood"},
     "local_neighbourhood does not yet support dynamic motif mobility"),
    ({"allow_realignment": True}, "allow_realignment=True is incompatible"),
    ({"enable_graph_interface_guidance": True, "enable_symmetric_scaffold_packing": True},
     "Declare graph interface guidance or automatic symmetric scaffold packing"),
])
def test_native_constructor_conflicts_are_persisted_before_feature_pipeline(
    compiled, tmp_path, monkeypatch, overrides, message
):
    def unexpected_feature_pipeline(*args, **kwargs):
        pytest.fail("invalid sampler reached feature pipeline")

    monkeypatch.setattr("rfd3_mosaic.rfd3_prevalidate.resolve_preflight_pipeline", unexpected_feature_pipeline)
    report_path = tmp_path / "failed.json"
    with pytest.raises(ValueError, match=message):
        prevalidate_rfd3_input(compiled, inference_sampler={**EXACT, **overrides}, report_path=report_path)
    report = json.loads(report_path.read_text())
    assert report["atom_array_validated"]
    assert report["sampler_dispatch_validated"]
    assert not report["sampler_constructor_validated"]
    assert not report["sampler_compatibility_validated"]
    assert not report["runtime_features_validated"]
    assert report["sampler_constructor_audit"]["constructor_attempted"]
    assert message in report["sampler_constructor_audit"]["error"]
    assert not report["checkpoint_compatibility_validated"]
    assert not report["model_forward_validated"]


def test_constructor_resolution_uses_native_preset_and_training_interpolation():
    training = source_inference_training_config()
    training.model.net.diffusion_module.sigma_data = 23
    training.model.net.inference_sampler = {
        "gamma_0": 0.8, "s_max": 123, "sigma_data": "${model.net.diffusion_module.sigma_data}"
    }
    parameters, audit = _resolve_sampler_constructor_config(EXACT, training)
    sampler = ConditionalDiffusionSampler(**parameters).sampler
    assert sampler.gamma_0 == 0.6  # Native CLI preset overrides training value.
    assert sampler.sigma_data == 23  # Training-only value resolves in its namespace.
    assert sampler.s_max == 123
    assert audit["training_sampler_source"] == "provided_training_configuration"
    assert not audit["checkpoint_loaded"]
    assert training.model.net.inference_sampler.gamma_0 == 0.8  # No caller mutation.
    parameters, _ = _resolve_sampler_constructor_config({**EXACT, "gamma_0": 0.55}, training)
    assert ConditionalDiffusionSampler(**parameters).sampler.gamma_0 == 0.55


@pytest.mark.parametrize("legacy", [False, True])
def test_worker_preflight_and_native_hydra_command_bind_same_sampler(tmp_path, legacy):
    example = {"extra": {
        "generated_polymer_continuity_guidance": {
            "enabled": True, "target_ca_distance": 3.9, "tolerance": 0.4, "projection_iterations": 27,
        },
        "generated_cross_chain_topology_guidance": {
            "enabled": True, "routing_ownership_weight": 0.7, "routing_clearance": 3.3,
        },
    }}
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"design": example}))
    sampling = {"sampler": {**EXACT, "allow_realignment": False}, "timesteps": 50,
                "execution_backend": "explicit_all_copy", "neighbour_radius": 2}
    if legacy:
        sampling["sampler"].update(symmetry_state_mode="legacy_asu", symmetry_noise_mode="independent")
    checked = _sampler_dispatch_overrides(sampling, path)
    resolved, _ = _resolve_sampler_constructor_config(checked)
    preflight_sampler = ConditionalDiffusionSampler(**resolved).sampler
    with hydra.initialize_config_dir(config_dir=str(ROOT / "models/rfd3/configs"), version_base="1.3"):
        configuration = hydra.compose(config_name="inference", overrides=_sampler_inference_arguments(sampling, path))
    actual = OmegaConf.to_container(configuration.inference_sampler, resolve=True)
    runtime_sampler = ConditionalDiffusionSampler(**actual).sampler
    assert runtime_sampler.num_timesteps == preflight_sampler.num_timesteps == 50
    assert runtime_sampler.symmetry_neighbour_radius == preflight_sampler.symmetry_neighbour_radius == 2
    assert runtime_sampler.enable_generated_polymer_continuity_guidance is (not legacy)
    assert runtime_sampler.enable_generated_cross_chain_topology_guidance is (not legacy)
    for key in checked:
        if key == "kind":
            continue  # The dispatcher consumes kind rather than forwarding it.
        assert getattr(runtime_sampler, key) == getattr(preflight_sampler, key)


def test_default_preflight_binds_compiled_guidance_not_only_symmetry_kind(compiled, tmp_path):
    payload = json.loads(compiled.read_text())
    example = next(iter(payload.values()))
    # Keep real compiled geometry unchanged, but reproduce contradictory
    # compiler preferences: the old input-only dispatch guard accepted this.
    example["extra"]["resolved_design_preferences"] = {
        "packing": "balanced", "cavity": "auto", "diversity": "medium",
        "interface_area": "auto", "component_motion": "free",
        "mobility_subspace": "bounded_se3", "initial_radius_scale": 1.0,
        "diversity_plan": {"global_pose_samples": 1},
        "sampler_overrides": {"enable_assembly_robust_capture": True, "assembly_capture_weight": 1.0},
    }
    if not Path(example["input"]).is_absolute():
        example["input"] = str(compiled.parent / example["input"])
    copied = tmp_path / "input.json"
    copied.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Assembly robust capture requires"):
        prevalidate_rfd3_input(copied, report_path=tmp_path / "failed.json")
    report = json.loads((tmp_path / "failed.json").read_text())
    assert report["sampler_dispatch_validated"]
    assert not report["sampler_constructor_validated"]
