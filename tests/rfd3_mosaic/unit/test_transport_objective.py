"""The searched full-scaffold objective must describe the committed state."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rfd3.inference.symmetry.differentiable_transport import interpolate_se3
from rfd3.inference.symmetry.motif_mobility import OrbitRigidMotifController
from rfd3.inference.symmetry.scaffold_guidance import (
    ScaffoldGuidanceConfig,
    _rotation_from_vector,
    build_boundary_topology,
    scaffold_orbit_energy,
)
from rfd3.inference.symmetry.scaffold_transport import ScaffoldReferenceTransport
from scipy.spatial.transform import Rotation
from test_reference_transport import transport_fixture

from rfd3_mosaic.validation.reference_transport import (
    build_reference_transport_plan,
    replay_reference_transport,
)
from rfd3_mosaic.validation.reference_transport import interpolate_se3 as numpy_screw


def _matrix(vector, translation):
    rotation = _rotation_from_vector(vector)
    return torch.cat(
        (
            torch.cat((rotation, translation[:, None]), dim=1),
            vector.new_tensor([[0.0, 0.0, 0.0, 1.0]]),
        ),
        dim=0,
    )


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.1, 2.8])
def test_screw_matches_commit_and_has_finite_correct_gradient(angle):
    parameters = torch.tensor(
        [angle, 0.0, 0.0, 0.2, -0.3, 0.7, 0.0, 0.0, 0.0, -0.1, 0.8, 0.2],
        dtype=torch.float64,
        requires_grad=True,
    )

    def evaluate(p):
        return interpolate_se3(_matrix(p[:3], p[3:6]), _matrix(p[6:9], p[9:12]), 0.37)

    observed = evaluate(parameters)
    expected = numpy_screw(
        _matrix(parameters[:3], parameters[3:6]).detach().numpy(),
        _matrix(parameters[6:9], parameters[9:12]).detach().numpy(),
        0.37,
    )
    np.testing.assert_allclose(observed.detach().numpy(), expected, atol=1e-10)
    assert torch.autograd.gradcheck(evaluate, (parameters,), eps=1e-6, atol=2e-6)


def test_screw_rejects_same_ambiguous_pi_branch_as_commit():
    left = torch.eye(4, dtype=torch.float64)
    right = left.clone()
    right[:3, :3] = torch.from_numpy(Rotation.from_rotvec([np.pi, 0, 0]).as_matrix())
    with pytest.raises(ValueError, match="ambiguous pi"):
        interpolate_se3(left, right, 0.4)


@pytest.mark.parametrize(
    "angle",
    [
        0.0,
        0.00011,
        0.0002,
        0.0005,
        0.001,
        0.009999,
        0.010001,
        0.4999,
        0.5001,
        1.3512,
        1.3515,
    ],
)
def test_float32_screw_jacobian_matches_float64_and_finite_difference(angle):
    # Construct a rotation analytically to isolate screw interpolation from
    # any separate rotation-vector conversion's small-angle numerics.
    def evaluate(parameters):
        theta, x, y, z = parameters.unbind()
        zero, one = theta * 0.0, theta * 0.0 + 1.0
        sine, cosine = torch.sin(theta), torch.cos(theta)
        right = torch.stack(
            (
                one,
                zero,
                zero,
                x,
                zero,
                cosine,
                -sine,
                y,
                zero,
                sine,
                cosine,
                z,
                zero,
                zero,
                zero,
                one,
            )
        ).reshape(4, 4)
        return interpolate_se3(torch.eye(4, dtype=parameters.dtype), right, 0.37)

    double = torch.tensor([angle, 0.2, -0.3, 0.7], dtype=torch.float64)
    single = double.float()
    reference = torch.autograd.functional.jacobian(evaluate, double)
    actual = torch.autograd.functional.jacobian(evaluate, single)
    torch.testing.assert_close(
        evaluate(single).double(), evaluate(double), atol=2e-7, rtol=2e-6
    )
    torch.testing.assert_close(actual.double(), reference, atol=3e-6, rtol=2e-5)
    finite_difference = torch.stack(
        [
            (evaluate(double + step) - evaluate(double - step)) / 2e-6
            for step in torch.eye(4, dtype=torch.float64) * 1e-6
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual.double(), finite_difference, atol=3e-6, rtol=2e-5)


def _controller_fixture():
    contract, xyz, plan, features, topology = transport_fixture()
    original = features["motif_pos"][None].clone()
    fixed_indices = torch.tensor(features["mosaic_transport_fixed_atom_indices"])
    groups = torch.stack(
        [fixed_indices[group["atom_indices"]] for group in plan["groups"].values()]
    )
    features.update(
        {
            "token_bonds": torch.zeros((44, 44)),
            "sym_transform": {
                str(i): (torch.tensor(m[:3, :3]), torch.tensor(m[:3, 3]))
                for i, m in enumerate(np.array(plan["registry_transforms"]))
            },
            "motif_constraint_group_orbit_index": torch.tensor([0, 0]),
            "motif_constraint_group_orbit_transform_id": torch.tensor([0, 1]),
            "motif_constraint_group_atom_indices": groups,
            "motif_constraint_group_atom_mask": torch.ones_like(
                groups, dtype=torch.bool
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor([0]),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "motif_constraint_orbit_bounds": torch.tensor([[2.0, 10.0]]),
        }
    )
    controller = OrbitRigidMotifController.from_features(
        features,
        original,
        start_fraction=0.0,
        end_fraction=1.0,
        response=1.0,
        per_step_translation=0.25,
        per_step_rotation_degrees=1.0,
    )
    boundary = build_boundary_topology(features, ~topology.generated_atom_mask)
    return original, features, topology, boundary, controller


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_scored_coordinates_equal_commit_including_second_increment(dtype):
    original, features, topology, _, controller = _controller_fixture()
    original = original.to(dtype)
    runtime = ScaffoldReferenceTransport(features, topology)
    motif = controller.motifs[0]
    for scale in (0.4, 1.0):
        motif.state.rotation[0] = _rotation_from_vector(
            torch.tensor([0.01, -0.02, 0.015], dtype=torch.float64) * scale
        )
        motif.state.translation[0] = torch.tensor([0.2, -0.1, 0.3]) * scale
        target = controller.materialize_target()[0].to(dtype)
        evaluated = runtime.candidate_coordinates(target, original[0])
        prepared = runtime.prepare(
            original,
            SimpleNamespace(target=target[None], coordinates=None),
            projector=lambda x, t: x,
        )
        torch.testing.assert_close(evaluated, prepared[0][0], atol=2e-5, rtol=2e-6)
        runtime.commit(prepared, progress=scale)
        original = prepared[0]


def test_coupled_rigid_motion_has_no_spurious_junction_gradient():
    original, features, topology, boundary, _ = _controller_fixture()
    runtime = ScaffoldReferenceTransport(features, topology)
    displacement = torch.tensor(0.2, dtype=original.dtype, requires_grad=True)
    target = original[0] + torch.stack(
        (displacement * 0.0, displacement * 0.0, displacement)
    )
    config = ScaffoldGuidanceConfig(clash_weight=0.0, tilt_weight=0.0, prior_weight=0.0)
    frozen = scaffold_orbit_energy(
        target, original[0], boundary, None, config=config
    ).total
    coupled = scaffold_orbit_energy(
        target,
        runtime.candidate_coordinates(target, original[0]),
        boundary,
        None,
        config=config,
    ).total
    frozen_gradient = torch.autograd.grad(frozen, displacement, retain_graph=True)[0]
    coupled_gradient = torch.autograd.grad(coupled, displacement)[0]
    assert abs(float(frozen_gradient)) > 1e-3
    assert abs(float(coupled_gradient)) < 1e-10


def test_two_different_anchors_transport_gradients_and_committed_coordinates_match():
    contract, _, plan, features, topology = transport_fixture()
    groups = [
        {
            "group_id": f"{chain}{end}",
            "members": [
                {"src_components": [f"{chain}{residue}" for residue in residues]}
            ],
        }
        for chain in "AB"
        for end, residues in (("start", (1, 2)), ("end", (21, 22)))
    ]
    orbits = [
        {
            "constraint_orbit_id": end,
            "group_ids": [f"A{end}", f"B{end}"],
            "master_group_id": f"A{end}",
            "group_transform_ids": [0, 1],
            "mobility_mode": "orbit_rigid",
            "mobility_subspace": "bounded_se3",
            "max_translation": 2.0,
            "max_rotation_deg": 10.0,
        }
        for end in ("start", "end")
    ]
    registry = {str(i): np.array(t) for i, t in enumerate(plan["registry_transforms"])}
    plan = build_reference_transport_plan(
        contract, groups, orbits, ["0", "1"], registry, plan["fixed_atoms"]
    )
    features["mosaic_reference_transport"] = plan
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features["motif_pos"].clone()
    fixed = torch.tensor(features["mosaic_transport_fixed_atom_indices"])

    def candidate(p):
        target = original.clone()
        for name, group in plan["groups"].items():
            indices = fixed[group["atom_indices"]]
            base = original[indices]
            offset = 0 if name.endswith("start") else 6
            rotation = _rotation_from_vector(p[offset : offset + 3])
            translation = p[offset + 3 : offset + 6]
            if name.startswith("B"):
                symmetry = original.new_tensor(np.diag([-1.0, -1.0, 1.0]))
                rotation = symmetry @ rotation @ symmetry.T
                translation = symmetry @ translation
            target[indices] = (
                (base - base.mean(dim=0)) @ rotation.T + base.mean(dim=0) + translation
            )
        return target

    parameters = torch.tensor(
        [
            0.0001,
            -0.0002,
            0.0001,
            0.002,
            -0.001,
            0.003,
            -0.0001,
            0.0001,
            -0.0002,
            -0.001,
            0.002,
            0.004,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    def evaluate(p):
        return runtime.candidate_coordinates(candidate(p), original)

    assert torch.autograd.gradcheck(evaluate, (parameters,), eps=1e-6, atol=2e-6)
    target = candidate(parameters).detach()
    prepared = runtime.prepare(
        original[None],
        SimpleNamespace(target=target[None], coordinates=None),
        projector=lambda x, t: x,
    )
    torch.testing.assert_close(
        evaluate(parameters).detach(), prepared[0][0], atol=1e-10, rtol=1e-10
    )


def test_real_controller_can_take_feasible_transport_move_without_frozen_penalty():
    original, features, topology, boundary, controller = _controller_fixture()
    frozen_controller = copy.deepcopy(controller)
    runtime = ScaffoldReferenceTransport(features, topology)
    config = ScaffoldGuidanceConfig(
        junction_weight=1.0,
        clash_weight=0.0,
        tilt_weight=0.0,
        prior_weight=0.0,
    )

    # A weak, prescribed axial preference isolates the junction term. This
    # whole-assembly rigid move preserves every distance, so it must not
    # compete with an invented junction strain in the proposal objective.
    def preference(target):
        return -0.001 * (
            target[boundary.fixed_ca_atom_indices, 2].mean()
            - original[0, boundary.fixed_ca_atom_indices, 2].mean()
        )

    kwargs = dict(
        progress=0.6,
        topology=boundary,
        axis=None,
        principal_axes=(None,),
        config=config,
        apply_update=True,
        pose_energy=preference,
        candidate_validator=runtime.candidate_validator(
            original,
            projector=lambda x, t: x,
            geometry_guard=lambda x: {"accepted": True},
        ),
    )
    frozen_controller.update_orbits_from_scaffold(original, **kwargs)
    target = controller.update_orbits_from_scaffold(
        original,
        **kwargs,
        candidate_state_resolver=runtime.candidate_coordinates,
    )
    old_motion = float(frozen_controller.motifs[0].state.translation[0, 2])
    new_motion = float(controller.motifs[0].state.translation[0, 2])
    assert new_motion > 0.1
    assert new_motion > 5 * abs(old_motion)
    prepared = runtime.prepare(
        original,
        SimpleNamespace(target=target, coordinates=None),
        projector=lambda x, t: x,
    )
    # The exact same state used by local and joint descent passes the full
    # independent reference/backbone certificate before it could be committed.
    torch.testing.assert_close(
        runtime.candidate_coordinates(target[0], original[0]),
        prepared[0][0],
        atol=1e-9,
        rtol=1e-9,
    )


def test_native_sampler_scores_commits_and_replays_real_controller_motion(monkeypatch):
    from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry

    original, features, topology, boundary, controller = _controller_fixture()
    fixed = ~topology.generated_atom_mask
    initial_contract = copy.deepcopy(features["mosaic_scaffold_contract"])
    features.update(
        {
            "symmetry_id": "C2",
            "sym_entity_id": torch.zeros(176, dtype=torch.long),
            "sym_transform_id": torch.arange(2).repeat_interleave(88),
            "is_sym_asu": torch.arange(176) < 88,
            "sym_orbit_slot": torch.arange(88).repeat(2),
            "sym_orbit_slot_verified": torch.tensor(True),
            "motif_constraint_group_membership": fixed[None, :],
            "motif_constraint_target_coordinates": original[:, None].clone(),
            "is_motif_atom_with_fixed_coord": fixed,
            "ref_element": torch.zeros(176, dtype=torch.long),
            "partial_t": 2.0,
        }
    )
    monkeypatch.setattr(
        OrbitRigidMotifController, "from_features", lambda *a, **k: controller
    )
    update = OrbitRigidMotifController.update_orbits_from_scaffold

    def with_prescribed_preference(self, coordinates, **kwargs):
        core = kwargs["pose_energy"]
        kwargs["pose_energy"] = (
            lambda target: core(target)
            - 0.001 * target[boundary.fixed_ca_atom_indices, 2].mean()
        )
        assert kwargs["candidate_state_resolver"] is not None
        return update(self, coordinates, **kwargs)

    monkeypatch.setattr(
        OrbitRigidMotifController,
        "update_orbits_from_scaffold",
        with_prescribed_preference,
    )
    observed_conditioning = []

    class Denoiser(torch.nn.Module):
        def forward(self, X_noisy_L, f, **kwargs):
            observed_conditioning.append(f["motif_pos"].clone())
            return {
                "X_L": torch.tensor(
                    [
                        residue["reference_backbone"]
                        for residue in f["mosaic_scaffold_contract"]["residues"]
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
        motif_mobility_proposal_source="scaffold_boundary",
        motif_mobility_apply_updates=True,
        motif_mobility_target_update_count=0,
        motif_mobility_update_interval=1,
        motif_mobility_clash_weight=0.0,
        motif_mobility_tilt_weight=0.0,
        motif_mobility_prior_weight=0.0,
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
    trace = diagnostic["reference_transport"]
    assert diagnostic["contract_met"]
    assert len(trace["accepted_transitions"]) >= 1
    assert float((result["X_L"][0, fixed, 2] - original[0, fixed, 2]).mean()) > 0.1
    assert not torch.equal(observed_conditioning[-1][fixed], original[0, fixed])
    torch.testing.assert_close(features["motif_pos"], original[0])
    final, _ = replay_reference_transport(
        initial_contract, features["mosaic_reference_transport"], trace
    )
    torch.testing.assert_close(
        result["X_L"][0, fixed],
        torch.tensor(
            [record["reference_backbone"] for record in final["residues"]]
        ).reshape(-1, 3)[fixed],
        check_dtype=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_joint_packing_reports_energy_of_coupled_commit_and_keeps_raw_proposal():
    from rfd3.inference.symmetry.graph_interface_guidance import (
        GraphInterfaceGuidanceConfig,
        GraphInterfacePatchState,
        build_symmetric_scaffold_interface_topology,
        graph_interface_energy,
    )
    from rfd3.inference.symmetry.scaffold_core_guidance import (
        ScaffoldCoreGuidanceConfig,
        scaffold_geometry_guard,
    )

    original, features, topology, boundary, controller = _controller_fixture()
    features["symmetry_id"] = "C2"
    interface = build_symmetric_scaffold_interface_topology(
        features, ~topology.generated_atom_mask
    )
    interface = replace(
        interface,
        edges=tuple(
            replace(
                edge,
                automatic_quality=False,
                requested_contact_count=2,
                requested_residues_per_side=2,
                requested_contiguous_residues_per_side=2,
            )
            for edge in interface.edges
        ),
    )
    config = GraphInterfaceGuidanceConfig(
        weight=1.0,
        coverage_weight=0.0,
        continuity_weight=0.0,
        orientation_weight=0.0,
        shape_weight=0.0,
        backbone_weight=0.0,
        interface_balance_weight=0.0,
        patch_exclusivity_weight=0.0,
        clash_weight=0.0,
        distance_weight=0.0,
        target_ca_distance=8.0,
        capture_ca_distance=20.0,
        pairs_per_edge=2,
        start_fraction=0.0,
        end_fraction=1.0,
        terminal_weight_floor=1.0,
        maximum_token_step=0.1,
        token_smoothing_weight=0.0,
        patch_blend_radius=0,
        maximum_patch_rotation_degrees=0.0,
        final_polish_steps=0,
    )
    patch = GraphInterfacePatchState(assignments={})
    runtime = ScaffoldReferenceTransport(features, topology)
    validate = runtime.candidate_validator(
        original,
        projector=lambda x, t: x,
        geometry_guard=scaffold_geometry_guard(
            original, topology, ScaffoldCoreGuidanceConfig()
        ),
    )
    frozen_controller = copy.deepcopy(controller)
    kwargs = dict(
        progress=0.6,
        topology=boundary,
        axis=None,
        principal_axes=(None,),
        scaffold_config=ScaffoldGuidanceConfig(
            clash_weight=0.0,
            tilt_weight=0.0,
            prior_weight=0.0,
        ),
        interface_topology=interface,
        interface_config=config,
        projector=lambda x: x,
        apply_update=True,
        candidate_validator=validate,
    )
    frozen_controller.update_orbits_with_interface_packing(
        original,
        features,
        **kwargs,
        patch_state=GraphInterfacePatchState(assignments={}),
    )
    target, raw, report = controller.update_orbits_with_interface_packing(
        original,
        features,
        **kwargs,
        patch_state=patch,
        candidate_state_resolver=runtime.candidate_coordinates,
    )
    assert report["committed"], report
    assert report["motif_pose_changed"]
    assert float(controller.motifs[0].state.translation.norm()) > 2 * float(
        frozen_controller.motifs[0].state.translation.norm()
    )
    prepared = runtime.prepare(
        original,
        SimpleNamespace(target=target, coordinates=raw),
        projector=lambda x, t: x,
    )
    scored = runtime.candidate_coordinates(target[0], raw[0])
    torch.testing.assert_close(scored, prepared[0][0], atol=1e-9, rtol=1e-9)
    # A prematurely transported return value would be transported a second
    # time by prepare(), which this coordinate/energy comparison would catch.
    assert not torch.allclose(
        raw[0, topology.generated_atom_mask], scored[topology.generated_atom_mask]
    )
    graph = graph_interface_energy(
        scored[None],
        interface,
        GraphInterfaceGuidanceConfig(**report["packing_objective"]["effective_config"]),
        target_ca_distance_override=report["packing_objective"]["target_ca_distance"],
        patch_assignments=patch.assignments,
    )
    assert report["candidate_packing"] == pytest.approx(float(graph.total), abs=1e-8)
