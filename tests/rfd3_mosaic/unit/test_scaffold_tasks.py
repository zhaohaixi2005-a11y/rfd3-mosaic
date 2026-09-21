"""CPU task preparation and frozen-input replay, without learned inference."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_scaffold_builder import _reference

from rfd3_mosaic.assembly_compiler import compile_experiment_assembly
from rfd3_mosaic.output import compile_rfd3_input
from rfd3_mosaic.scaffold_input import _compiled_contig_chains
from rfd3_mosaic.scaffold_tasks import _remap_extra, _write_cif, prepare_scaffold_task

ROOT = Path(__file__).resolve().parents[3]


def _fixture(directory):
    # Ideal peptide with two helical arms and a return turn. The seed is the
    # first and last two residues; template coordinates have a nonzero origin.
    backbone = _reference(20, compact=True) + [35.0, 0.0, 10.0]
    chains = [
        [
            {
                "name": "ALA",
                "atoms": dict(zip(("N", "CA", "C", "O"), residue, strict=True)),
            }
            for residue in backbone @ rotation.T
        ]
        for rotation in (np.eye(3), np.diag([-1.0, -1.0, 1.0]))
    ]
    template = directory / "template.cif"
    _write_cif(template, chains)
    # Keep source numbering contiguous; the public selectors identify the
    # immutable spans and the compiler removes the reference-generated middle.
    seed = directory / "source.cif"
    _write_cif(seed, chains[:1])
    design = {
        "schema_version": 1,
        "name": "complete_scaffold_replay",
        "input": str(seed),
        "symmetry": "C2",
        "task": "preserve_supplied_geometry",
        "fixed_arrangement": "locked",
        "generation": [
            {
                "kind": "between",
                "from_selector": "A1-2",
                "to_selector": "A21-22",
                "length": 18,
                "orbit_offset": 0,
            }
        ],
        "constraints": [
            {
                "kind": "fixed_xyz",
                "selector": selection,
                "atoms": "all",
                "coupling_group": "seed",
            }
            for selection in ("A1-2", "A21-22")
        ],
        "sampling": {
            "initial_pose": {
                "radius": {"minimum": 35.0, "maximum": 35.0},
                "orientation": {"method": "fixed", "rotation_deg": [0.0, 0.0, 0.0]},
                "seed": 0,
            },
            "designs": 1,
            "seed": 17,
            "timesteps": 50,
            "scaffold_packing": "off",
        },
        "output": {"root": str(directory / "runs")},
    }
    blueprint = {
        "schema_version": 1,
        "template": str(template),
        "chains": [["A", "B"]],
        "partial_t": 2.0,
        "maximum_template_seed_rmsd": 0.01,
        "maximum_template_symmetry_rmsd": 0.01,
        "limits": {
            "maximum_ca_deviation": 1.0,
            "contact_distance": 8.0,
            "minimum_helix_contact_fraction": 0.2,
            "minimum_interchain_segment_distance": 1.0,
            "minimum_ca_bond_distance": 3.3,
            "maximum_ca_bond_distance": 4.3,
            "fixed_ca_tolerance": 0.01,
            "geometry_tolerance": 0.001,
            "maximum_unsupported_run": 8,
        },
        "helix_blocks": [
            {"entity": 0, "id": "left", "start": 2, "end": 9},
            {"entity": 0, "id": "right", "start": 14, "end": 21},
        ],
        "support_edges": [{"left": "left", "right": "right"}],
        "closure": {
            "attempts": 1,
            "max_nfev": 150,
            "reference_shape_weight": 1.0,
            "maximum_reference_ca_deviation": 1.0,
        },
    }
    config = directory / "source_design.yaml"
    config.write_text(yaml.safe_dump(design))
    blueprint_path = directory / "blueprint.yaml"
    blueprint_path.write_text(yaml.safe_dump(blueprint))
    return config, blueprint_path


def _compile_prepared(design, directory):
    return compile_experiment_assembly(
        {"kind": "user_design", "config": str(design), "example_id": "replay"},
        directory,
        project_directory=design.parent,
        experiment_name="replay",
    )


def test_complete_peptide_preparation_recompiles_same_frozen_scaffold(tmp_path, capsys):
    from rfd3_mosaic.cli import main
    from rfd3_mosaic.experiment_worker import _require_compiled_pose_feasibility

    config, blueprint = _fixture(tmp_path)
    main(
        [
            "prepare-scaffold",
            str(config),
            "--blueprint",
            str(blueprint),
            "--output-dir",
            str(tmp_path / "prepared"),
        ]
    )
    report = json.loads((tmp_path / "prepared" / "preparation_report.json").read_text())
    assert "Prepared complete-scaffold task:" in capsys.readouterr().out
    assert report["status"] == "prepared"
    assert report["gpu_jobs_submitted"] == 0
    assert report["full_backbone_clearance"]["passed"]
    prepared = tmp_path / "prepared"
    prepared_design = yaml.safe_load((prepared / "design.yaml").read_text())
    assert not Path(prepared_design["input"]).is_absolute()
    assert (prepared / prepared_design["input"]).is_file()
    replay_blueprint = yaml.safe_load((prepared / "blueprint.yaml").read_text())
    assert not Path(replay_blueprint["template"]).is_absolute()
    assert (prepared / replay_blueprint["template"]).is_file()
    original_artifact = json.loads((prepared / "scaffold_artifact.json").read_text())
    # Recompilation is independent of the original authoring seed file.
    Path(yaml.safe_load(config.read_text())["input"]).unlink()
    outputs = _compile_prepared(prepared / "design.yaml", tmp_path / "replay")
    example = next(iter(json.loads(outputs.input_path.read_text()).values()))
    assert example["partial_t"] == 2.0
    assert example["contig"] == "A1-22,/0,B1-22"
    assert (
        example["extra"]["scaffold_input"]["structure_sha256"]
        == original_artifact["structure_sha256"]
    )
    assert example["extra"]["scaffold_input"]["input_contract_audit"]["passed"]
    # A curved, verified complete scaffold must use its actual geometry, not
    # the earlier straight-chord pose proxy retained in a compiler manifest.
    manifest = tmp_path / "legacy_failed_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "validation": {
                    "assembly_pose_feasibility": {
                        "evaluated": True,
                        "passed": False,
                        "failure_reasons": ["straight chord intersects competitor"],
                    }
                }
            }
        )
    )
    preflight = _require_compiled_pose_feasibility(outputs.input_path, manifest)
    assert preflight["passed"]
    assert preflight["measurement"] == "complete_scaffold_input"
    native_path = outputs.input_path.parent / example["input"]
    original_structure = native_path.read_bytes()
    native_path.write_bytes(original_structure + b"\n# changed after compilation\n")
    with pytest.raises(ValueError, match="changed before pose preflight"):
        _require_compiled_pose_feasibility(outputs.input_path, manifest)
    native_path.write_bytes(original_structure)
    original_payload = json.loads(outputs.input_path.read_text())
    unbound_payload = copy.deepcopy(original_payload)
    del next(iter(unbound_payload.values()))["extra"]["scaffold_input"]
    outputs.input_path.write_text(json.dumps(unbound_payload))
    with pytest.raises(
        ValueError, match="validated complete partial-diffusion artifact"
    ):
        _require_compiled_pose_feasibility(outputs.input_path, manifest)
    outputs.input_path.write_text(json.dumps(original_payload))
    # The marker alone is not provenance: both the frozen artifact and the
    # runtime contract must still be the exact bound versions.
    frozen_artifact = (
        outputs.input_path.parent / example["extra"]["scaffold_input"]["artifact_path"]
    )
    original_artifact_bytes = frozen_artifact.read_bytes()
    frozen_artifact.write_bytes(original_artifact_bytes + b" ")
    with pytest.raises(ValueError):
        _require_compiled_pose_feasibility(outputs.input_path, manifest)
    frozen_artifact.write_bytes(original_artifact_bytes)
    relaxed_payload = copy.deepcopy(original_payload)
    next(iter(relaxed_payload.values()))["extra"]["mosaic_scaffold_contract"]["limits"][
        "maximum_ca_deviation"
    ] += 1.0
    outputs.input_path.write_text(json.dumps(relaxed_payload))
    with pytest.raises(ValueError):
        _require_compiled_pose_feasibility(outputs.input_path, manifest)
    missing_contract = copy.deepcopy(original_payload)
    del next(iter(missing_contract.values()))["extra"]["mosaic_scaffold_contract"]
    outputs.input_path.write_text(json.dumps(missing_contract))
    manifest.write_text(
        json.dumps(
            {
                "validation": {
                    "assembly_pose_feasibility": {"evaluated": True, "passed": True}
                }
            }
        )
    )
    with pytest.raises(ValueError):
        _require_compiled_pose_feasibility(outputs.input_path, manifest)
    outputs.input_path.write_text(json.dumps(original_payload))
    with pytest.raises(FileExistsError, match="overwrite prepared"):
        prepare_scaffold_task(config, blueprint, prepared)
    artifact_path = prepared / "scaffold_artifact.json"
    original_artifact["compiled_contract_sha256"] = "0" * 64
    artifact_path.write_text(json.dumps(original_artifact))
    with pytest.raises(ValueError, match="compiler contract SHA256"):
        _compile_prepared(prepared / "design.yaml", tmp_path / "tampered_replay")


def test_actual_interface_relation_uses_physical_member_copy(tmp_path):
    compiled = compile_rfd3_input(
        ROOT / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml",
        tmp_path / "compiled",
        base_directory=ROOT,
    )
    example = json.loads(compiled.input_path.read_text())[compiled.example_id]
    remapped, _ = _remap_extra(
        example["extra"], _compiled_contig_chains(example["contig"]), 3
    )
    relation = remapped["assembly_interface_relations"][0]
    # Original source-copy-zero fragment A is absent from selected ASU B+C.
    # The correct full-chain member is C's end, not A's end or ASU C itself.
    assert relation["left_transform_index"] == 2
    assert relation["right_transform_index"] == 0
    assert relation["left_source_components"] == [
        ",".join(f"C{i}" for i in range(117, 147))
    ]
    assert relation["right_source_components"] == [
        ",".join(f"A{i}" for i in range(1, 32))
    ]
    assert relation["source_copy_index"] == 0
    assert relation["target_copy_index"] == 0


def test_mobile_complete_task_builds_native_input_with_frozen_transport_plan(tmp_path):
    from rfd3_mosaic.rfd3_prevalidate import prevalidate_rfd3_input

    config, blueprint = _fixture(tmp_path)
    design = yaml.safe_load(config.read_text())
    # Advanced per-component control retains the same task and initial pose.
    design.pop("task")
    design.pop("fixed_arrangement")
    for constraint in design["constraints"]:
        constraint["pose"] = {
            "mode": "bounded_mobile",
            "subspace": "bounded_se3",
            "proposal": "denoiser_fit",
            "max_translation": 2.0,
            "max_rotation_deg": 10.0,
        }
    config.write_text(yaml.safe_dump(design))
    prepared = tmp_path / "mobile"
    prepare_scaffold_task(config, blueprint, prepared)
    # Simulate libm/NumPy operator roundoff after moving an artifact to Linux.
    artifact_path = prepared / "scaffold_artifact.json"
    artifact = json.loads(artifact_path.read_text())
    native = artifact["native_input"]
    matrices = native["symmetry"]["declared_transform_matrices"]
    key = next(iter(matrices))
    matrices[key][0][0] = float(np.nextafter(matrices[key][0][0], np.inf))
    matrix = native["extra"]["mosaic_reference_transport"]["registry_transforms"][0]
    matrix[0][0] = float(np.nextafter(matrix[0][0], np.inf))
    artifact_path.write_text(json.dumps(artifact))
    outputs = _compile_prepared(prepared / "design.yaml", tmp_path / "native")
    payload = json.loads(outputs.input_path.read_text())
    example = next(iter(payload.values()))
    plan = example["extra"]["mosaic_reference_transport"]
    assert all(g["mobile"] for g in plan["groups"].values())
    assert len(plan["residue_transforms"]) == 44
    report = prevalidate_rfd3_input(outputs.input_path)
    assert report["status"] == "passed"
    # Exercise the final feature aggregation binding against native-built
    # atom identities as well; prevalidation alone runs the symmetry builder.
    import torch
    from rfd3.inference.input_parsing import (
        DesignInputSpecification,
        ensure_input_is_abspath,
    )
    from rfd3.transforms.util_transforms import (
        AggregateFeaturesLikeAF3WithoutMSA,
        assign_types_,
    )

    raw = ensure_input_is_abspath(copy.deepcopy(example), outputs.input_path)
    atoms = assign_types_(DesignInputSpecification.safe_init(**raw).build())
    atoms.set_annotation("coord_to_be_noised", atoms.coord.copy())
    if "chain_iid" not in atoms.get_annotation_categories():
        atoms.set_annotation("chain_iid", atoms.chain_id.copy())
    features = {
        "ref_atom_name_chars": torch.zeros((len(atoms), 4), dtype=torch.long),
        "ref_element": torch.zeros(len(atoms), dtype=torch.long),
        "ref_pos": torch.tensor(atoms.coord),
    }
    built = AggregateFeaturesLikeAF3WithoutMSA().forward(
        {
            "atom_array": atoms,
            "feats": features,
            "specification": raw,
            "is_inference": True,
        }
    )
    assert len(built["feats"]["mosaic_transport_fixed_atom_indices"]) == len(
        plan["fixed_atoms"]
    )
    assert built["feats"]["mosaic_scaffold_backbone_atom_indices"].shape == (44, 4)
    from rfd3.inference.symmetry.motif_mobility import OrbitRigidMotifController
    from rfd3.transforms.symmetry import AddSymmetryFeats
    built = AddSymmetryFeats().forward(built)
    controller = OrbitRigidMotifController.from_features(
        built["feats"], torch.tensor(atoms.coord)[None], start_fraction=.05,
        end_fraction=.75, response=.2, per_step_translation=.25,
        per_step_rotation_degrees=1.,
    )
    assert controller is not None
    isolated = copy.deepcopy(controller)
    assert isolated is not controller
    torch.testing.assert_close(isolated.base_target, controller.base_target)


def test_interrupted_preparation_publication_exposes_no_runnable_task(
    tmp_path, monkeypatch
):
    import shutil

    config, blueprint = _fixture(tmp_path)
    copyfile = shutil.copyfile

    def fail_seed_copy(source, destination, **kwargs):
        if Path(destination).name == "seed_input.cif":
            raise OSError("simulated publication interruption")
        return copyfile(source, destination, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", fail_seed_copy)
    destination = tmp_path / "prepared"
    with pytest.raises(OSError, match="publication interruption"):
        prepare_scaffold_task(config, blueprint, destination)
    assert not destination.exists()
    assert list(tmp_path.glob(".mosaic-prepare-*")) == []


def test_mobile_scaffold_passes_complete_native_feature_pipeline(tmp_path):
    from rfd3.transforms.pipelines import build_atom14_base_pipeline
    from rfd3.inference.datasets import ContigJsonDataset

    config, blueprint = _fixture(tmp_path)
    design = yaml.safe_load(config.read_text())
    design.pop('task')
    design.pop('fixed_arrangement')
    for constraint in design['constraints']:
        constraint['pose'] = {'mode':'bounded_mobile','subspace':'bounded_se3',
                              'proposal':'denoiser_fit','max_translation':2.0,
                              'max_rotation_deg':10.0}
    config.write_text(yaml.safe_dump(design))
    prepared = tmp_path/'prepared'
    prepare_scaffold_task(config, blueprint, prepared)
    outputs = _compile_prepared(prepared/'design.yaml',tmp_path/'compiled')
    repo = ROOT
    args = yaml.safe_load((repo/'models/rfd3/configs/datasets/design_base.yaml').read_text())['global_transform_args']
    net = yaml.safe_load((repo/'models/rfd3/configs/model/components/rfd3_net.yaml').read_text())['token_initializer']
    args.update(is_inference=True,sigma_data=16.0,diffusion_batch_size=1,
                atom_1d_features=net['atom_1d_features'],token_1d_features=net['token_1d_features'])
    data = ContigJsonDataset(data=str(outputs.input_path),cif_parser_args=None,
                             transform=build_atom14_base_pipeline(**args),
                             name='scaffold-regression',subset_to_keys=None,eval_every_n=1)[0]
    feats = data['feats']
    assert feats['sym_orbit_slot_verified']
    assert len(feats['mosaic_transport_fixed_atom_indices']) == len(feats['mosaic_reference_transport']['fixed_atoms'])
    assert tuple(feats['mosaic_scaffold_backbone_atom_indices'].shape)==(44,4)
