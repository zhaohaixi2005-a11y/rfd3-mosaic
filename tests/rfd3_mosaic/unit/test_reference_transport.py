"""Rigid seed/reference transactions and replay without learned inference."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_generated_backbone_contract import backbone_contract_fixture

from rfd3_mosaic.validation.reference_transport import (
    build_reference_transport_plan,
    interpolate_se3,
    replay_reference_transport,
    transport_fingerprint,
    transported_reference,
)


def transport_fixture():
    contract, xyz = backbone_contract_fixture()
    xyz[:22] += [35, 0, 10]
    rotation = np.diag([-1.0, -1.0, 1.0])
    xyz[22:] = xyz[:22] @ rotation.T
    for record, bb in zip(contract["residues"], xyz, strict=True):
        record["reference_ca"], record["reference_backbone"] = (
            bb[1].tolist(),
            bb.tolist(),
        )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    groups = [
        {
            "group_id": c,
            "members": [{"src_components": [f"{c}{i}" for i in (1, 2, 21, 22)]}],
        }
        for c in "AB"
    ]
    orbits = [
        {
            "constraint_orbit_id": "seed",
            "group_ids": ["A", "B"],
            "master_group_id": "A",
            "group_transform_ids": [0, 1],
            "mobility_mode": "orbit_rigid",
            "mobility_subspace": "bounded_se3",
            "max_translation": 2.0,
            "max_rotation_deg": 10.0,
        }
    ]
    atoms, indices = [], []
    for i, record in enumerate(contract["residues"]):
        if record["fixed"]:
            for j, name in enumerate(("N", "CA", "C", "O")):
                atoms.append(
                    {
                        "chain_id": record["chain_id"],
                        "residue_number": record["residue_number"],
                        "atom_name": name,
                        "coordinate": xyz[i, j].tolist(),
                    }
                )
                indices.append(i * 4 + j)
    plan = build_reference_transport_plan(
        contract, groups, orbits, ["e", "r1"], {"e": np.eye(4), "r1": transform}, atoms
    )
    features = {
        "atom_to_token_map": torch.arange(44).repeat_interleave(4),
        "asym_id": torch.arange(2).repeat_interleave(22),
        "residue_index": torch.arange(1, 23).repeat(2),
        "is_ca": torch.tensor([False, True, False, False]).repeat(44),
        "is_protein": torch.ones(44, dtype=torch.bool),
        "motif_pos": torch.tensor(xyz.reshape(-1, 3), dtype=torch.float64),
        "mosaic_scaffold_contract": contract,
        "mosaic_scaffold_ca_atom_indices": torch.arange(1, 176, 4),
        "mosaic_scaffold_backbone_atom_indices": torch.arange(176).reshape(44, 4),
        "mosaic_reference_transport": plan,
        "mosaic_transport_fixed_atom_indices": indices,
    }
    from rfd3.inference.symmetry.scaffold_core_guidance import (
        build_scaffold_core_topology,
    )

    fixed = torch.tensor([r["fixed"] for r in contract["residues"]]).repeat_interleave(
        4
    )
    topology = build_scaffold_core_topology(features, fixed)
    return contract, xyz, plan, features, topology


def test_screw_interpolation_is_equivariant_under_translated_rotated_frame():
    a, b, frame = (np.eye(4) for _ in range(3))
    a[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    b[:3, :3] = Rotation.from_rotvec([-0.1, 0.3, 0.2]).as_matrix()
    b[:3, 3] = [2.0, -1.0, 3.0]
    frame[:3, :3] = Rotation.from_rotvec([0.5, 0.1, 0.7]).as_matrix()
    frame[:3, 3] = [30.0, -20.0, 7.0]
    inverse = np.linalg.inv(frame)
    actual = interpolate_se3(frame @ a @ inverse, frame @ b @ inverse, 0.37)
    np.testing.assert_allclose(
        actual, frame @ interpolate_se3(a, b, 0.37) @ inverse, atol=1e-10
    )


def test_reference_transaction_preserves_seed_and_moves_generated_region_and_replays():
    from rfd3.inference.symmetry.scaffold_transport import ScaffoldReferenceTransport

    contract, _xyz, plan, features, topology = transport_fixture()
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features["motif_pos"][None].clone()
    moved = original.clone()
    moved[:, :88, 0] += 0.1
    moved[:, 88:, 0] -= 0.1
    proposal = SimpleNamespace(target=moved, coordinates=None)
    prepared = runtime.prepare(
        original, proposal, projector=lambda candidate, target: candidate
    )
    torch.testing.assert_close(prepared[0], moved)
    assert runtime.accepted == []  # prepare does not mutate
    assert runtime.reference.contract == contract
    runtime.commit(prepared, progress=0.2)
    final, _transforms = replay_reference_transport(
        contract, plan, runtime.diagnostics()
    )
    np.testing.assert_allclose(
        [r["reference_backbone"] for r in final["residues"]],
        moved.numpy().reshape(44, 4, 3),
        atol=1e-10,
    )
    assert len(runtime.accepted) == 1
    # Fitting uses every selected fixed atom; a deformed seed is not accepted.
    invalid = moved.clone()
    invalid[0, 0] += 1
    with pytest.raises(ValueError, match="one preserved rigid body"):
        runtime.prepare(
            moved,
            SimpleNamespace(target=invalid, coordinates=None),
            projector=lambda x, t: x,
        )
    assert len(runtime.accepted) == 1
    assert runtime.reference.contract == final
    tampered = copy.deepcopy(runtime.diagnostics())
    tampered["accepted_transitions"] = []
    with pytest.raises(ValueError, match="accepted transport history"):
        replay_reference_transport(contract, plan, tampered)


@pytest.mark.parametrize("kind", ["asymmetric", "out_of_bounds", "locked"])
def test_invalid_group_motions_are_rejected(kind):
    contract, _, plan, _, _ = transport_fixture()
    a, b = np.eye(4), np.eye(4)
    a[0, 3], b[0, 3] = 0.1, -0.1
    if kind == "asymmetric":
        b[0, 3] = 0.2
    elif kind == "out_of_bounds":
        a[0, 3], b[0, 3] = 3.0, -3.0
    else:
        plan["groups"]["A"]["mobile"] = False
    with pytest.raises(ValueError):
        transported_reference(contract, plan, {"A": a, "B": b})


def test_replay_requires_matching_plan_even_when_no_movement():
    contract, _, plan, _, _ = transport_fixture()
    with pytest.raises(ValueError, match="fingerprint"):
        replay_reference_transport(contract, plan, {})
    report = {
        "plan_sha256": transport_fingerprint(plan),
        "accepted_transitions": [],
        "final_group_transforms": {k: np.eye(4).tolist() for k in plan["groups"]},
    }
    replay_reference_transport(contract, plan, report)


@pytest.mark.parametrize("valid", [True, False])
def test_native_sampler_coupled_commit_or_rollback_and_final_audit(monkeypatch, valid):
    from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry
    from test_symmetry_motif_finalization import _RecordingScaffoldController

    contract, _xyz, plan, features, topology = transport_fixture()
    original = features["motif_pos"][None].clone()
    fixed = ~topology.generated_atom_mask
    features.update(
        {
            "symmetry_id": "C2",
            "sym_entity_id": torch.zeros(176, dtype=torch.long),
            "sym_transform_id": torch.arange(2).repeat_interleave(88),
            "is_sym_asu": torch.arange(176) < 88,
            "sym_orbit_slot": torch.arange(88).repeat(2),
            "sym_orbit_slot_verified": torch.tensor(True),
            "sym_transform": {
                str(i): (torch.tensor(m[:3, :3]), torch.tensor(m[:3, 3]))
                for i, m in enumerate(np.array(plan["registry_transforms"]))
            },
            "motif_constraint_group_membership": fixed[None, :],
            "motif_constraint_target_coordinates": original[:, None].clone(),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "is_motif_atom_with_fixed_coord": fixed,
            "ref_element": torch.zeros(176, dtype=torch.long),
            "partial_t": 2.0,
        }
    )
    proposed = original.clone()
    proposed[:, :88, 0] += 0.1
    proposed[:, 88:, 0] -= 0.1
    if not valid:
        proposed[0, 0] += 1.0  # deforms a fixed joint seed

    class Controller(_RecordingScaffoldController):
        def update(self, coordinates, *, progress):
            self.last_update_applied = not torch.allclose(self.fixed_target, proposed)
            self.fixed_target = proposed.clone()
            return self.fixed_target

    controller = Controller(original)
    controller.motifs[0].group_atom_indices = tuple(
        torch.tensor(features["mosaic_transport_fixed_atom_indices"])[g["atom_indices"]]
        for g in plan["groups"].values()
    )
    monkeypatch.setattr(
        "rfd3.model.inference_sampler.OrbitRigidMotifController.from_features",
        lambda *a, **k: controller,
    )
    observed = []

    class Denoiser(torch.nn.Module):
        def forward(self, X_noisy_L, f, **kwargs):
            observed.append(f["motif_pos"].clone())
            return {
                "X_L": torch.tensor(
                    [
                        r["reference_backbone"]
                        for r in f["mosaic_scaffold_contract"]["residues"]
                    ],
                    dtype=original.dtype,
                ).reshape(1, -1, 3)
            }

    sampler = SampleDiffusionWithSymmetry(
        gamma_0=0.6,
        num_timesteps=6,
        preserve_fixed_motif_during_symmetry=True,
        require_motif_constraint_groups=True,
        symmetry_state_mode="orbit_average",
        symmetry_noise_mode="coupled",
        enable_orbit_rigid_motif_mobility=True,
        motif_mobility_proposal_source="denoiser",
        motif_mobility_apply_updates=True,
        motif_mobility_target_update_count=0,
    )
    monkeypatch.setattr(
        sampler,
        "_construct_inference_noise_schedule",
        lambda **kw: torch.tensor([1.0, 0.5, 0.2, 0.0], dtype=original.dtype),
    )
    with torch.no_grad():
        result = sampler.sample_diffusion_like_af3(
            f=features,
            diffusion_module=Denoiser(),
            diffusion_batch_size=1,
            coord_atom_lvl_to_be_noised=original,
            initializer_outputs={"chunked_pairwise_embedder": object()},
            ref_initializer_outputs=None,
            f_ref=None,
        )
    diagnostic = result["scaffold_contract_diagnostics"]
    assert diagnostic["contract_met"]
    trace = diagnostic["reference_transport"]
    assert bool(trace["accepted_transitions"]) is valid, str(
        trace["rejected_proposals"]
    )
    assert bool(trace["rejected_proposals"]) is not valid
    expected = proposed if valid else original
    torch.testing.assert_close(result["X_L"][:, fixed], expected[:, fixed])
    torch.testing.assert_close(observed[-1][fixed], expected[0, fixed])
    # The input mapping remains immutable across independent designs.
    torch.testing.assert_close(features["motif_pos"], original[0])
    assert features["mosaic_scaffold_contract"] == contract
    replay_reference_transport(contract, plan, trace)
