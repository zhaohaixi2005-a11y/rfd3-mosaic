import json
import math
import unittest
from unittest.mock import patch

import torch
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceEdge,
    GraphInterfaceGuidanceConfig,
    GraphInterfacePatchState,
    GraphInterfaceTopology,
    graph_interface_energy,
)
from rfd3.inference.symmetry.motif_mobility import (
    OrbitRigidMotifController,
    fit_centered_rigid_pose,
    mobility_window_weight,
    rigid_mobility_phase,
    rigid_mobility_response,
)
from rfd3.inference.symmetry.scaffold_guidance import (
    BoundaryTopology,
    CyclicAxis,
    ScaffoldGuidanceConfig,
)


def _z_rotation(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _apply(points, rotation, translation):
    return points @ rotation.T + translation


class MotifMobilityTestCase(unittest.TestCase):
    def test_schedule_freezes_before_and_after_window(self) -> None:
        self.assertEqual(
            mobility_window_weight(
                0.05,
                start_fraction=0.10,
                end_fraction=0.85,
            ),
            0.0,
        )
        self.assertGreater(
            mobility_window_weight(
                0.50,
                start_fraction=0.10,
                end_fraction=0.85,
            ),
            0.0,
        )
        self.assertEqual(
            mobility_window_weight(
                0.90,
                start_fraction=0.10,
                end_fraction=0.85,
            ),
            0.0,
        )

    def test_rigid_mobility_uses_active_window_percentages(self) -> None:
        expected = (
            (0.20, "capture", 5.0),
            (0.60, "settle", 2.5),
            (0.90, "polish", 1.0),
        )
        for progress, name, response_scale in expected:
            with self.subTest(progress=progress):
                phase = rigid_mobility_phase(
                    progress,
                    start_fraction=0.0,
                    end_fraction=1.0,
                )
                self.assertEqual(phase.name, name)
                self.assertEqual(phase.response_scale, response_scale)

        self.assertEqual(
            rigid_mobility_phase(
                0.05,
                start_fraction=0.05,
                end_fraction=0.85,
            ).name,
            "frozen",
        )
        self.assertEqual(
            rigid_mobility_phase(
                0.85,
                start_fraction=0.05,
                end_fraction=0.85,
            ).name,
            "frozen",
        )

    def test_phase_schedule_rejects_missing_polish_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero polish"):
            rigid_mobility_phase(
                0.5,
                start_fraction=0.0,
                end_fraction=1.0,
                capture_fraction=0.5,
                settle_fraction=0.5,
            )

    def test_phase_response_ratios_survive_larger_base_response(self) -> None:
        effective = []
        for progress in (0.20, 0.60, 0.90):
            phase = rigid_mobility_phase(
                progress,
                start_fraction=0.0,
                end_fraction=1.0,
            )
            effective.append(
                rigid_mobility_response(
                    0.4,
                    phase,
                    capture_response_scale=5.0,
                )
            )
        self.assertEqual(effective, [1.0, 0.5, 0.2])

    def test_rigid_fit_recovers_known_pose_without_reflection(self) -> None:
        template = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        rotation = _z_rotation(17.0)
        center = template.mean(dim=1, keepdim=True)
        translation = torch.tensor(
            [[[1.0, -0.5, 0.25]]],
            dtype=torch.float64,
        )
        proposal = (template - center) @ rotation.T + center + translation

        fitted_rotation, fitted_translation, rmsd = fit_centered_rigid_pose(
            template, proposal
        )

        self.assertTrue(
            torch.allclose(
                fitted_rotation[0],
                rotation,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                fitted_translation,
                translation[:, 0, :],
                atol=1e-6,
            )
        )
        self.assertLess(float(rmsd.max()), 1e-8)
        self.assertGreater(float(torch.linalg.det(fitted_rotation[0])), 0.0)

    @staticmethod
    def _controller_case(*, with_orbit_schedule: bool = False):
        template = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        transforms = {
            str(transform_id): (
                _z_rotation(120.0 * transform_id),
                torch.zeros(3, dtype=torch.float64),
            )
            for transform_id in range(3)
        }
        target = torch.empty((1, 12, 3), dtype=torch.float64)
        for transform_id in range(3):
            target[:, 4 * transform_id : 4 * (transform_id + 1)] = _apply(
                template,
                transforms[str(transform_id)][0],
                transforms[str(transform_id)][1],
            )
        features = {
            "sym_transform": transforms,
            "motif_constraint_group_orbit_index": torch.tensor([0, 0, 0]),
            "motif_constraint_group_orbit_transform_id": torch.tensor([0, 1, 2]),
            "motif_constraint_group_atom_indices": torch.tensor(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                ]
            ),
            "motif_constraint_group_atom_mask": torch.ones(
                (3, 4),
                dtype=torch.bool,
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor([0]),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "motif_constraint_orbit_bounds": torch.tensor([[2.0, 10.0]]),
        }
        if with_orbit_schedule:
            features.update(
                {
                    "motif_constraint_orbit_subspace": torch.tensor([4]),
                    "motif_constraint_orbit_proposal": torch.tensor([1]),
                    "motif_constraint_orbit_schedule": torch.tensor(
                        [[0.0, 1.0, 1.0, 0.10, 0.50]]
                    ),
                    "motif_constraint_orbit_objective_ids": ((),),
                }
            )
        controller = OrbitRigidMotifController.from_features(
            features,
            target,
            start_fraction=0.0,
            end_fraction=1.0,
            response=1.0,
            per_step_translation=0.25,
            per_step_rotation_degrees=1.0,
        )
        assert controller is not None
        return template, target, transforms, controller

    def test_orbit_schedule_overrides_legacy_global_step_bounds(self) -> None:
        template, _, transforms, controller = self._controller_case(
            with_orbit_schedule=True
        )
        desired_rotation = _z_rotation(30.0)
        center = template.mean(dim=1, keepdim=True)
        desired_master = (
            (template - center) @ desired_rotation.T
            + center
            + torch.tensor([[[5.0, 0.0, 0.0]]], dtype=torch.float64)
        )
        raw = torch.empty((1, 12, 3), dtype=torch.float64)
        for transform_id in range(3):
            raw[:, 4 * transform_id : 4 * (transform_id + 1)] = _apply(
                desired_master,
                transforms[str(transform_id)][0],
                transforms[str(transform_id)][1],
            )

        controller.update(raw, progress=0.5)

        motif = controller.motifs[0]
        rotation_angle = torch.acos(
            torch.clamp(
                (torch.trace(motif.state.rotation[0]) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
        self.assertAlmostEqual(
            math.degrees(float(rotation_angle)),
            0.50,
            places=5,
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(motif.state.translation[0])),
            0.10,
            places=6,
        )
        diagnostics = controller.diagnostics()["orbits"][0]
        self.assertEqual(diagnostics["mobility_subspace"], "bounded_se3")
        self.assertEqual(diagnostics["proposal_source"], "denoiser_fit")
        self.assertAlmostEqual(
            diagnostics["schedule"]["max_step_rotation_degrees"],
            0.50,
        )

    @staticmethod
    def _scaffold_guidance_case():
        template, target, transforms, controller = (
            MotifMobilityTestCase._controller_case()
        )
        scaffold = target.clone()
        scaffold[0, 1] = torch.tensor(
            [6.0, 0.0, 0.0],
            dtype=scaffold.dtype,
        )
        generated_mask = torch.zeros(12, dtype=torch.bool)
        generated_mask[1] = True
        topology = BoundaryTopology(
            junction_pairs=torch.tensor([[0, 1]]),
            fixed_ca_atom_indices=torch.tensor([0]),
            generated_ca_atom_indices=torch.tensor([1]),
            generated_atom_mask=generated_mask,
        )
        axis = CyclicAxis(
            point=torch.zeros(3, dtype=torch.float64),
            direction=torch.tensor(
                [0.0, 0.0, 1.0],
                dtype=torch.float64,
            ),
            transform_ids=(0, 1, 2),
        )
        config = ScaffoldGuidanceConfig(
            junction_weight=1.0,
            clash_weight=0.0,
            tilt_weight=0.0,
            prior_weight=0.0,
        )
        return (
            template,
            target,
            transforms,
            controller,
            scaffold,
            topology,
            axis,
            config,
        )

    @staticmethod
    def _joint_packing_mobility_case():
        """One mobile C3 motif plus two generated residues per copy."""

        template = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [5.0, 1.0, 0.0],
                    [5.0, 0.0, 1.0],
                    [6.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        generated_template = torch.tensor(
            [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        transforms = {
            str(transform_id): (
                _z_rotation(120.0 * transform_id),
                torch.zeros(3, dtype=torch.float64),
            )
            for transform_id in range(3)
        }
        target = torch.empty((1, 18, 3), dtype=torch.float64)
        fixed_groups = []
        generated_indices = []
        for transform_id in range(3):
            start = transform_id * 6
            rotation, translation = transforms[str(transform_id)]
            target[:, start : start + 4] = _apply(
                template,
                rotation,
                translation,
            )
            target[0, start + 4 : start + 6] = _apply(
                generated_template,
                rotation,
                translation,
            )
            fixed_groups.append(list(range(start, start + 4)))
            generated_indices.extend((start + 4, start + 5))

        controller_features = {
            "sym_transform": transforms,
            "motif_constraint_group_orbit_index": torch.tensor([0, 0, 0]),
            "motif_constraint_group_orbit_transform_id": torch.tensor([0, 1, 2]),
            "motif_constraint_group_atom_indices": torch.tensor(fixed_groups),
            "motif_constraint_group_atom_mask": torch.ones((3, 4), dtype=torch.bool),
            "motif_constraint_orbit_master_group_index": torch.tensor([0]),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "motif_constraint_orbit_bounds": torch.tensor([[2.0, 10.0]]),
            "motif_constraint_orbit_subspace": torch.tensor([4]),
            "motif_constraint_orbit_proposal": torch.tensor([2]),
            "motif_constraint_orbit_schedule": torch.tensor(
                [[0.0, 1.0, 1.0, 0.25, 1.0]]
            ),
            "motif_constraint_orbit_objective_ids": ((),),
        }
        controller = OrbitRigidMotifController.from_features(
            controller_features,
            target,
            start_fraction=0.0,
            end_fraction=1.0,
            response=1.0,
            per_step_translation=0.25,
            per_step_rotation_degrees=1.0,
        )
        assert controller is not None

        generated_mask = torch.zeros(18, dtype=torch.bool)
        generated_mask[generated_indices] = True
        junction_pairs = torch.tensor([[3, 4], [9, 10], [15, 16]], dtype=torch.long)
        boundary_topology = BoundaryTopology(
            junction_pairs=junction_pairs,
            fixed_ca_atom_indices=junction_pairs[:, 0],
            generated_ca_atom_indices=junction_pairs[:, 1],
            generated_atom_mask=generated_mask,
        )
        axis = CyclicAxis(
            point=torch.zeros(3, dtype=torch.float64),
            direction=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            transform_ids=(0, 1, 2),
        )
        scaffold_config = ScaffoldGuidanceConfig(
            junction_weight=1.0,
            clash_weight=0.0,
            tilt_weight=0.0,
            prior_weight=0.01,
        )

        def edge(edge_index, left_indices, right_indices):
            left = torch.zeros(18, dtype=torch.bool)
            right = torch.zeros(18, dtype=torch.bool)
            left[left_indices] = True
            right[right_indices] = True
            return GraphInterfaceEdge(
                edge_id=f"packing@{edge_index}",
                source_interface_id="packing",
                left_generated_ca_mask=left,
                right_generated_ca_mask=right,
                left_generated_token_ids=torch.tensor(left_indices),
                right_generated_token_ids=torch.tensor(right_indices),
                requested_contact_count=2,
                requested_residues_per_side=2,
                requested_contiguous_residues_per_side=2,
                automatic_quality=False,
                contact_cutoff=5.5,
                distance_target=None,
                distance_tolerance=None,
            )

        interface_topology = GraphInterfaceTopology(
            edges=(
                edge(0, [4, 5], [10, 11]),
                edge(1, [10, 11], [16, 17]),
                edge(2, [16, 17], [4, 5]),
            ),
            generated_atom_mask=generated_mask,
            junction_ca_pairs=junction_pairs,
        )
        interface_config = GraphInterfaceGuidanceConfig(
            weight=5.0,
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
            maximum_token_step=0.5,
            token_smoothing_weight=0.0,
            patch_blend_radius=0,
            maximum_patch_rotation_degrees=0.0,
            final_polish_steps=0,
        )
        runtime_features = {
            "atom_to_token_map": torch.arange(18),
            "asym_id": torch.tensor([0] * 6 + [1] * 6 + [2] * 6),
            "residue_index": torch.tensor(list(range(6)) * 3),
            "is_ca": torch.ones(18, dtype=torch.bool),
            "is_virtual": torch.zeros(18, dtype=torch.bool),
        }
        return (
            target,
            transforms,
            controller,
            boundary_topology,
            axis,
            scaffold_config,
            interface_topology,
            interface_config,
            runtime_features,
        )

    def test_joint_packing_mobility_commits_one_atomic_transaction(self):
        (
            coordinates,
            transforms,
            controller,
            boundary_topology,
            axis,
            scaffold_config,
            interface_topology,
            interface_config,
            features,
        ) = self._joint_packing_mobility_case()
        baseline = graph_interface_energy(
            coordinates, interface_topology, interface_config
        )
        patch_state = GraphInterfacePatchState(assignments={})
        # Public create-interface designs declare response=0.2.  Interface
        # quality still requests capture, while the time-based schedule caps
        # this mid-window proposal at the settle response.  A late poor patch
        # therefore cannot trigger an unsafe capture-sized rigid jump.
        controller.motifs[0].response = 0.2

        target, observed, diagnostics = controller.update_orbits_with_interface_packing(
            coordinates,
            features,
            progress=0.5,
            topology=boundary_topology,
            axis=axis,
            principal_axes=(axis.direction,),
            scaffold_config=scaffold_config,
            interface_topology=interface_topology,
            interface_config=interface_config,
            patch_state=patch_state,
            projector=lambda candidate: candidate,
            apply_update=True,
            capture_response_scale=4.0,
            expand_response_scale=3.0,
            polish_response_scale=1.0,
        )
        final = graph_interface_energy(
            observed,
            interface_topology,
            interface_config,
            patch_assignments=patch_state.assignments,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertTrue(diagnostics["committed"])
        self.assertEqual(diagnostics["adaptive_phase"], "capture")
        self.assertEqual(diagnostics["motif_pose_response_scale"], 4.0)
        trajectory = controller.diagnostics()["trajectory"][-1]
        self.assertAlmostEqual(
            trajectory["orbit_proposals"][0]["effective_response"],
            0.5,
        )
        self.assertEqual(
            trajectory["orbit_proposals"][0]["temporal_phase"],
            "settle",
        )
        self.assertTrue(controller.last_joint_transaction_applied)
        self.assertLess(float(final.total), float(baseline.total))
        self.assertFalse(torch.equal(observed, coordinates))
        master = target[:, :4]
        for transform_id, start in enumerate((0, 6, 12)):
            rotation, translation = transforms[str(transform_id)]
            canonical = (target[:, start : start + 4] - translation) @ rotation
            self.assertTrue(torch.allclose(canonical, master, atol=1e-6))

    def test_joint_packing_mobility_proposal_only_rolls_back_everything(self):
        (
            coordinates,
            _,
            controller,
            boundary_topology,
            axis,
            scaffold_config,
            interface_topology,
            interface_config,
            features,
        ) = self._joint_packing_mobility_case()
        initial_rotation = controller.motifs[0].state.rotation.clone()
        initial_translation = controller.motifs[0].state.translation.clone()
        patch_state = GraphInterfacePatchState(assignments={})

        target, observed, diagnostics = controller.update_orbits_with_interface_packing(
            coordinates,
            features,
            progress=0.5,
            topology=boundary_topology,
            axis=axis,
            principal_axes=(axis.direction,),
            scaffold_config=scaffold_config,
            interface_topology=interface_topology,
            interface_config=interface_config,
            patch_state=patch_state,
            projector=lambda candidate: candidate,
            apply_update=False,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertFalse(diagnostics["committed"])
        self.assertTrue(torch.equal(target, coordinates))
        self.assertTrue(torch.equal(observed, coordinates))
        self.assertTrue(
            torch.equal(controller.motifs[0].state.rotation, initial_rotation)
        )
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.translation,
                initial_translation,
            )
        )
        self.assertEqual(patch_state.assignments, {})
        self.assertFalse(controller.last_joint_transaction_applied)

    def test_joint_packing_mobility_rolls_back_on_proposal_error(self):
        (
            coordinates,
            _,
            controller,
            boundary_topology,
            axis,
            scaffold_config,
            interface_topology,
            interface_config,
            features,
        ) = self._joint_packing_mobility_case()
        initial_rotation = controller.motifs[0].state.rotation.clone()
        initial_translation = controller.motifs[0].state.translation.clone()
        patch_state = GraphInterfacePatchState(assignments={})

        def fail_after_mutation(*args, **kwargs):
            patch_state.locked = True
            raise RuntimeError("synthetic packing failure")

        with (
            patch(
                "rfd3.inference.symmetry.motif_mobility."
                "apply_graph_interface_guidance",
                side_effect=fail_after_mutation,
            ),
            self.assertRaisesRegex(RuntimeError, "synthetic packing failure"),
        ):
            controller.update_orbits_with_interface_packing(
                coordinates,
                features,
                progress=0.5,
                topology=boundary_topology,
                axis=axis,
                principal_axes=(axis.direction,),
                scaffold_config=scaffold_config,
                interface_topology=interface_topology,
                interface_config=interface_config,
                patch_state=patch_state,
                projector=lambda candidate: candidate,
                apply_update=True,
            )

        self.assertFalse(patch_state.locked)
        self.assertEqual(patch_state.assignments, {})
        self.assertTrue(
            torch.equal(controller.motifs[0].state.rotation, initial_rotation)
        )
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.translation,
                initial_translation,
            )
        )

    def test_controller_moves_one_master_pose_with_bounded_c3_copies(
        self,
    ) -> None:
        template, _, transforms, controller = self._controller_case()
        desired_rotation = _z_rotation(30.0)
        center = template.mean(dim=1, keepdim=True)
        desired_master = (
            (template - center) @ desired_rotation.T
            + center
            + torch.tensor(
                [[[5.0, 0.0, 0.0]]],
                dtype=torch.float64,
            )
        )
        raw = torch.empty((1, 12, 3), dtype=torch.float64)
        for transform_id in range(3):
            raw[:, 4 * transform_id : 4 * (transform_id + 1)] = _apply(
                desired_master,
                transforms[str(transform_id)][0],
                transforms[str(transform_id)][1],
            )

        target = controller.update(raw, progress=0.5)
        state = controller.motifs[0].state
        angle = torch.acos(
            torch.clamp(
                (torch.trace(state.rotation[0]) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
        self.assertLessEqual(
            math.degrees(float(angle)),
            1.0 + 1e-5,
        )
        self.assertAlmostEqual(
            math.degrees(float(angle)),
            1.0,
            places=5,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(state.translation[0])),
            0.25 + 1e-6,
        )
        self.assertTrue(
            torch.allclose(
                state.translation[0],
                torch.tensor(
                    [0.25, 0.0, 0.0],
                    dtype=torch.float64,
                ),
                atol=1e-6,
            )
        )

        master = target[:, :4]
        reference_distances = torch.cdist(master, master)
        for transform_id in range(3):
            group = target[
                :,
                4 * transform_id : 4 * (transform_id + 1),
            ]
            inverse = (group - transforms[str(transform_id)][1]) @ transforms[
                str(transform_id)
            ][0]
            self.assertTrue(torch.allclose(inverse, master, atol=1e-6))
            self.assertTrue(
                torch.allclose(
                    torch.cdist(group, group),
                    reference_distances,
                    atol=1e-6,
                )
            )

        for _ in range(20):
            target = controller.update(raw, progress=0.5)
        state = controller.motifs[0].state
        cumulative_angle = torch.acos(
            torch.clamp(
                (torch.trace(state.rotation[0]) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
        self.assertLessEqual(
            math.degrees(float(cumulative_angle)),
            10.0 + 1e-5,
        )
        self.assertAlmostEqual(
            math.degrees(float(cumulative_angle)),
            10.0,
            places=5,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(state.translation[0])),
            2.0 + 1e-6,
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(state.translation[0])),
            2.0,
            places=6,
        )

        state.rotation[0] = _z_rotation(3.0)
        state.translation[0] = torch.tensor(
            [0.5, 0.0, 0.0],
            dtype=torch.float64,
        )
        frozen_rotation = state.rotation.clone()
        frozen_translation = state.translation.clone()
        controller.update(raw, progress=1.0)
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.rotation,
                frozen_rotation,
            )
        )
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.translation,
                frozen_translation,
            )
        )

    def test_controller_diagnostics_are_json_serializable(self) -> None:
        template, _, transforms, controller = self._controller_case()
        desired_master = template + torch.tensor(
            [[[1.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        raw = torch.empty((1, 12, 3), dtype=torch.float64)
        for transform_id in range(3):
            raw[:, 4 * transform_id : 4 * (transform_id + 1)] = _apply(
                desired_master,
                transforms[str(transform_id)][0],
                transforms[str(transform_id)][1],
            )

        controller.update(raw, progress=0.5)
        diagnostics = controller.diagnostics()

        self.assertEqual(diagnostics["update_calls"], 1)
        self.assertEqual(diagnostics["active_window_calls"], 1)
        self.assertEqual(len(diagnostics["orbits"]), 1)
        self.assertEqual(len(diagnostics["trajectory"]), 1)
        self.assertGreater(
            diagnostics["orbits"][0]["translation_norms"][0],
            0.0,
        )
        json.dumps(diagnostics)

    def test_scaffold_proposal_only_does_not_change_target(self) -> None:
        (
            _,
            target,
            _,
            controller,
            scaffold,
            topology,
            axis,
            config,
        ) = self._scaffold_guidance_case()
        initial_rotation = controller.motifs[0].state.rotation.clone()
        initial_translation = controller.motifs[0].state.translation.clone()

        observed = controller.update_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axis=axis.direction,
            config=config,
            apply_update=False,
        )

        self.assertTrue(torch.equal(observed, target))
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.rotation,
                initial_rotation,
            )
        )
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.translation,
                initial_translation,
            )
        )
        self.assertFalse(controller.last_update_applied)
        snapshot = controller.diagnostics()["trajectory"][-1]
        self.assertTrue(snapshot["proposal_only"])
        self.assertTrue(snapshot["accepted"])
        self.assertFalse(snapshot["applied"])
        self.assertLess(
            snapshot["proposed_energy"]["total"],
            snapshot["initial_energy"]["total"],
        )

    def test_scaffold_rigid_steps_follow_capture_settle_polish_schedule(
        self,
    ) -> None:
        expected = (
            (0.20, "capture", 1.0),
            (0.60, "settle", 0.5),
            (0.90, "polish", 0.2),
        )
        for progress, phase, response in expected:
            with self.subTest(progress=progress):
                (
                    _,
                    _,
                    _,
                    controller,
                    scaffold,
                    topology,
                    axis,
                    config,
                ) = self._scaffold_guidance_case()
                controller.motifs[0].response = 0.2
                controller.update_from_scaffold(
                    scaffold,
                    progress=progress,
                    topology=topology,
                    axis=axis,
                    principal_axis=axis.direction,
                    config=config,
                    apply_update=False,
                )
                proposal = controller.diagnostics()["trajectory"][-1][
                    "orbit_proposals"
                ][0]
                self.assertEqual(proposal["temporal_phase"], phase)
                self.assertAlmostEqual(
                    proposal["effective_response"],
                    response,
                )

    def test_scaffold_update_lowers_energy_and_preserves_exact_c3(
        self,
    ) -> None:
        (
            _,
            target,
            transforms,
            controller,
            scaffold,
            topology,
            axis,
            config,
        ) = self._scaffold_guidance_case()

        observed = controller.update_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axis=axis.direction,
            config=config,
            apply_update=True,
        )

        snapshot = controller.diagnostics()["trajectory"][-1]
        self.assertTrue(snapshot["accepted"])
        self.assertTrue(snapshot["applied"])
        self.assertTrue(controller.last_update_applied)
        proposal = snapshot["orbit_proposals"][0]
        self.assertIn("rotation_projected", proposal["gradient_norms"])
        self.assertIn("translation_projected", proposal["gradient_norms"])
        self.assertGreaterEqual(len(proposal["line_search_trials"]), 1)
        accepted_trial = proposal["line_search_trials"][-1]
        self.assertTrue(accepted_trial["finite"])
        self.assertTrue(accepted_trial["improves"])
        self.assertLess(
            accepted_trial["energy"],
            snapshot["initial_energy"]["total"],
        )
        self.assertLess(
            snapshot["proposed_energy"]["total"],
            snapshot["initial_energy"]["total"],
        )
        self.assertFalse(torch.allclose(observed[:, :4], target[:, :4]))

        master = observed[:, :4]
        for transform_id in range(3):
            group = observed[
                :,
                4 * transform_id : 4 * (transform_id + 1),
            ]
            rotation, translation = transforms[str(transform_id)]
            canonical = (group - translation) @ rotation
            self.assertTrue(torch.allclose(canonical, master, atol=1e-6))

    def test_axis_free_bounded_se3_scaffold_update_executes(self) -> None:
        (
            _,
            target,
            transforms,
            controller,
            scaffold,
            topology,
            _,
            config,
        ) = self._scaffold_guidance_case()

        observed = controller.update_orbits_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=None,
            principal_axes=(None,),
            config=config,
            apply_update=True,
        )

        self.assertTrue(controller.last_update_applied)
        self.assertFalse(torch.allclose(observed[:, :4], target[:, :4]))
        master = observed[:, :4]
        for transform_id in range(3):
            group = observed[
                :,
                4 * transform_id : 4 * (transform_id + 1),
            ]
            rotation, translation = transforms[str(transform_id)]
            canonical = (group - translation) @ rotation
            self.assertTrue(torch.allclose(canonical, master, atol=1e-6))

    def test_scaffold_pose_prior_scales_with_declared_orbit_bounds(
        self,
    ) -> None:
        (
            _,
            _,
            _,
            controller,
            scaffold,
            topology,
            axis,
            _,
        ) = self._scaffold_guidance_case()
        motif = controller.motifs[0]
        motif.maximum_translation = 15.0
        motif.maximum_rotation_degrees = 45.0
        config = ScaffoldGuidanceConfig(
            junction_weight=1.0,
            clash_weight=0.0,
            tilt_weight=0.0,
            prior_weight=0.05,
            translation_prior_scale=1.0,
            rotation_prior_scale_degrees=5.0,
        )

        controller.update_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axis=axis.direction,
            config=config,
            apply_update=False,
        )

        prior = controller.diagnostics()["orbits"][0]["effective_pose_prior"]
        self.assertEqual(prior["translation_scale"], 5.0)
        self.assertEqual(prior["rotation_scale_degrees"], 15.0)
        self.assertEqual(
            prior["normalization"],
            "at_least_one_third_of_hard_bound",
        )

    def test_joint_scaffold_acceptance_includes_additional_pose_energy(
        self,
    ) -> None:
        (
            _,
            target,
            _,
            controller,
            scaffold,
            topology,
            axis,
            config,
        ) = self._scaffold_guidance_case()
        baseline = target[0].clone()

        def packing_energy(candidate_target):
            # Strongly prefer positive master-copy x translation.  A C3-wide
            # center would cancel by symmetry, so use one corresponding
            # master atom exactly as a real per-interface objective does.
            return -100.0 * candidate_target[0, 0]

        def scaffold_only_joint_energy(
            candidate_target,
            _scaffold_coordinates,
            **_kwargs,
        ):
            # Any pose change is slightly worse under the scaffold-only
            # objective.  The combined packing improvement must nevertheless
            # be allowed to win the joint acceptance decision.
            penalty = torch.mean(torch.square(candidate_target - baseline))
            return penalty, {"total": float(penalty.detach().cpu().item())}

        with patch.object(
            controller,
            "_joint_scaffold_energy",
            side_effect=scaffold_only_joint_energy,
        ):
            observed = controller.update_orbits_from_scaffold(
                scaffold,
                progress=0.5,
                topology=topology,
                axis=axis,
                principal_axes=(axis.direction,),
                config=config,
                apply_update=True,
                pose_energy=packing_energy,
            )

        snapshot = controller.diagnostics()["trajectory"][-1]
        self.assertTrue(snapshot["accepted"])
        self.assertTrue(snapshot["applied"])
        self.assertGreater(
            snapshot["proposed_energy"]["total"],
            snapshot["initial_energy"]["total"],
        )
        self.assertLess(
            snapshot["additional_pose_energy"]["delta"],
            0.0,
        )
        self.assertLess(snapshot["joint_energy_delta"], 0.0)
        self.assertFalse(torch.allclose(observed[0], baseline))

    def test_controller_materializes_complete_d3_orbit(self) -> None:
        template = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        secondary = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float64,
        )
        transforms = {}
        for transform_id in range(3):
            transforms[str(transform_id)] = (
                _z_rotation(120.0 * transform_id),
                torch.zeros(3, dtype=torch.float64),
            )
            transforms[str(3 + transform_id)] = (
                _z_rotation(120.0 * transform_id) @ secondary,
                torch.zeros(3, dtype=torch.float64),
            )
        target = torch.empty((1, 24, 3), dtype=torch.float64)
        for transform_id in range(6):
            rotation, translation = transforms[str(transform_id)]
            target[:, 4 * transform_id : 4 * (transform_id + 1)] = _apply(
                template, rotation, translation
            )
        features = {
            "sym_transform": transforms,
            "motif_constraint_group_orbit_index": torch.zeros(
                6,
                dtype=torch.long,
            ),
            "motif_constraint_group_orbit_transform_id": torch.arange(6),
            "motif_constraint_group_atom_indices": torch.arange(24).reshape(
                6,
                4,
            ),
            "motif_constraint_group_atom_mask": torch.ones(
                (6, 4),
                dtype=torch.bool,
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor([0]),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "motif_constraint_orbit_bounds": torch.tensor([[2.0, 10.0]]),
        }
        controller = OrbitRigidMotifController.from_features(
            features,
            target,
            start_fraction=0.0,
            end_fraction=1.0,
            response=1.0,
            per_step_translation=0.25,
            per_step_rotation_degrees=1.0,
        )
        assert controller is not None
        desired_master = template + torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        proposal = torch.empty_like(target)
        for transform_id in range(6):
            rotation, translation = transforms[str(transform_id)]
            proposal[
                :,
                4 * transform_id : 4 * (transform_id + 1),
            ] = _apply(desired_master, rotation, translation)

        observed = controller.update(proposal, progress=0.5)

        master = observed[:, :4]
        self.assertFalse(torch.equal(master, target[:, :4]))
        for transform_id in range(6):
            group = observed[
                :,
                4 * transform_id : 4 * (transform_id + 1),
            ]
            rotation, translation = transforms[str(transform_id)]
            canonical = (group - translation) @ rotation
            self.assertTrue(torch.allclose(canonical, master, atol=1e-6))

    def test_scaffold_radial_subspace_moves_only_radially(self) -> None:
        (
            _,
            _,
            _,
            controller,
            scaffold,
            topology,
            axis,
            config,
        ) = self._scaffold_guidance_case()
        motif = controller.motifs[0]
        motif.mobility_subspace = "radial"
        motif.maximum_rotation_degrees = 0.0
        initial_center = motif.template_master[0].mean(dim=0)
        radial = initial_center.clone()
        radial[2] = 0.0
        radial = radial / torch.linalg.vector_norm(radial)

        controller.update_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axis=axis.direction,
            config=config,
            apply_update=True,
        )

        translation = motif.state.translation[0]
        tangential = torch.tensor(
            [-radial[1], radial[0], 0.0],
            dtype=translation.dtype,
        )
        self.assertTrue(controller.last_update_applied)
        self.assertGreater(float(torch.linalg.vector_norm(translation)), 0.0)
        self.assertAlmostEqual(
            float(torch.dot(translation, axis.direction)),
            0.0,
            places=8,
        )
        self.assertAlmostEqual(
            float(torch.dot(translation, tangential)),
            0.0,
            places=8,
        )
        self.assertTrue(
            torch.allclose(
                motif.state.rotation[0],
                torch.eye(3, dtype=translation.dtype),
                atol=1e-8,
            )
        )

    def test_scaffold_radial_rotation_blocks_translation_leakage(self) -> None:
        (
            _,
            _,
            _,
            controller,
            scaffold,
            topology,
            axis,
            config,
        ) = self._scaffold_guidance_case()
        motif = controller.motifs[0]
        motif.mobility_subspace = "radial_rotation"
        initial_center = motif.template_master[0].mean(dim=0)
        radial = initial_center.clone()
        radial[2] = 0.0
        radial = radial / torch.linalg.vector_norm(radial)

        controller.update_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axis=axis.direction,
            config=config,
            apply_update=True,
        )

        translation = motif.state.translation[0]
        tangential = torch.tensor(
            [-radial[1], radial[0], 0.0],
            dtype=translation.dtype,
        )
        self.assertTrue(controller.last_update_applied)
        self.assertAlmostEqual(
            float(torch.dot(translation, axis.direction)),
            0.0,
            places=8,
        )
        self.assertAlmostEqual(
            float(torch.dot(translation, tangential)),
            0.0,
            places=8,
        )
        self.assertFalse(
            torch.allclose(
                motif.state.rotation[0],
                torch.eye(3, dtype=translation.dtype),
                atol=1e-8,
            )
        )

    def test_scaffold_tilt_only_rotates_without_translation_or_twist(self) -> None:
        (
            _,
            _,
            _,
            controller,
            scaffold,
            topology,
            axis,
            _,
        ) = self._scaffold_guidance_case()
        motif = controller.motifs[0]
        motif.mobility_subspace = "tilt_only"
        config = ScaffoldGuidanceConfig(
            junction_weight=0.0,
            clash_weight=0.0,
            tilt_weight=1.0,
            prior_weight=0.0,
            maximum_tilt_degrees=5.0,
        )

        controller.update_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axis=torch.tensor(
                [1.0, 0.0, 0.2],
                dtype=scaffold.dtype,
            ),
            config=config,
            apply_update=True,
        )

        rotation = motif.state.rotation[0]
        translation = motif.state.translation[0]
        self.assertTrue(controller.last_update_applied)
        self.assertTrue(torch.equal(translation, torch.zeros_like(translation)))
        self.assertFalse(
            torch.allclose(
                rotation,
                torch.eye(3, dtype=rotation.dtype),
                atol=1e-8,
            )
        )
        # The infinitesimal tilt proposal has no component about the cyclic
        # z axis.  For the small bounded step this appears as equal diagonal
        # x/y rotation terms and zero in-plane skew (no axial twist).
        self.assertAlmostEqual(float(rotation[1, 0] - rotation[0, 1]), 0.0, places=8)

    @staticmethod
    def _two_orbit_scaffold_guidance_case(*, dihedral: bool = False):
        template = torch.tensor(
            [
                [
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        templates = (
            template,
            template
            + torch.tensor(
                [[[0.0, 0.0, 4.0]]],
                dtype=torch.float64,
            ),
        )
        transforms = {
            str(transform_id): (
                _z_rotation(120.0 * transform_id),
                torch.zeros(3, dtype=torch.float64),
            )
            for transform_id in range(3)
        }
        if dihedral:
            secondary = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=torch.float64,
            )
            for transform_id in range(3):
                transforms[str(3 + transform_id)] = (
                    _z_rotation(120.0 * transform_id) @ secondary,
                    torch.zeros(3, dtype=torch.float64),
                )
        group_action_count = len(transforms)
        atoms_per_orbit = group_action_count * 4
        atom_count = 2 * atoms_per_orbit
        target = torch.empty((1, atom_count, 3), dtype=torch.float64)
        for orbit_index, orbit_template in enumerate(templates):
            for transform_id in range(group_action_count):
                start = orbit_index * atoms_per_orbit + transform_id * 4
                target[:, start : start + 4] = _apply(
                    orbit_template,
                    transforms[str(transform_id)][0],
                    transforms[str(transform_id)][1],
                )
        features = {
            "sym_transform": transforms,
            "motif_constraint_group_orbit_index": torch.tensor(
                [0] * group_action_count + [1] * group_action_count
            ),
            "motif_constraint_group_orbit_transform_id": torch.tensor(
                list(range(group_action_count)) * 2
            ),
            "motif_constraint_group_atom_indices": torch.arange(atom_count).reshape(
                2 * group_action_count, 4
            ),
            "motif_constraint_group_atom_mask": torch.ones(
                (2 * group_action_count, 4),
                dtype=torch.bool,
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor(
                [0, group_action_count]
            ),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1, 1]),
            "motif_constraint_orbit_bounds": torch.tensor([[2.0, 10.0], [2.0, 10.0]]),
            "motif_constraint_orbit_subspace": torch.tensor([4, 4]),
            "motif_constraint_orbit_proposal": torch.tensor([2, 2]),
            "motif_constraint_orbit_schedule": torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.25, 1.0],
                    [0.0, 1.0, 1.0, 0.25, 1.0],
                ]
            ),
            "motif_constraint_orbit_objective_ids": ((), ()),
            "motif_constraint_orbit_ids": (
                "mobile_orbit_alpha",
                "mobile_orbit_beta",
            ),
            "motif_constraint_orbit_component_ids": (
                "mobile_component_alpha",
                "mobile_component_beta",
            ),
        }
        controller = OrbitRigidMotifController.from_features(
            features,
            target,
        )
        assert controller is not None
        scaffold = target.clone()
        scaffold[0, 1] = torch.tensor(
            [7.0, 0.0, 0.0],
            dtype=scaffold.dtype,
        )
        second_generated = atoms_per_orbit + 1
        scaffold[0, second_generated] = torch.tensor(
            [2.0, 5.0, 4.0],
            dtype=scaffold.dtype,
        )
        generated_mask = torch.zeros(atom_count, dtype=torch.bool)
        generated_mask[[1, second_generated]] = True
        topology = BoundaryTopology(
            junction_pairs=torch.tensor([[0, 1], [atoms_per_orbit, second_generated]]),
            fixed_ca_atom_indices=torch.tensor([0, atoms_per_orbit]),
            generated_ca_atom_indices=torch.tensor([1, second_generated]),
            generated_atom_mask=generated_mask,
        )
        axis = CyclicAxis(
            point=torch.zeros(3, dtype=torch.float64),
            direction=torch.tensor(
                [0.0, 0.0, 1.0],
                dtype=torch.float64,
            ),
            transform_ids=(0, 1, 2),
        )
        config = ScaffoldGuidanceConfig(
            junction_weight=1.0,
            clash_weight=0.0,
            tilt_weight=0.0,
            prior_weight=0.0,
        )
        return target, scaffold, topology, axis, config, controller

    def test_d3_orbits_update_atomically_and_close_all_actions(
        self,
    ) -> None:
        (
            _,
            scaffold,
            topology,
            axis,
            config,
            controller_a,
        ) = self._two_orbit_scaffold_guidance_case(dihedral=True)
        (
            _,
            _,
            _,
            _,
            _,
            controller_b,
        ) = self._two_orbit_scaffold_guidance_case(dihedral=True)
        controller_b.motifs.reverse()
        principal_axes_a = (
            axis.direction,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        principal_axes_b = tuple(reversed(principal_axes_a))

        observed_a = controller_a.update_orbits_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axes=principal_axes_a,
            config=config,
            apply_update=True,
        )
        observed_b = controller_b.update_orbits_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axes=principal_axes_b,
            config=config,
            apply_update=True,
        )

        self.assertTrue(controller_a.last_update_applied)
        self.assertTrue(controller_b.last_update_applied)
        self.assertTrue(torch.allclose(observed_a, observed_b, atol=1e-8))
        for motif in controller_a.motifs:
            self.assertEqual(motif.group_transform_ids.numel(), 6)
            master = observed_a[:, motif.master_atom_indices]
            for group_row, transform_id_tensor in enumerate(motif.group_transform_ids):
                transform_id = int(transform_id_tensor.item())
                rotation, translation = controller_a.sym_transforms[transform_id]
                group = observed_a[
                    :,
                    motif.group_atom_indices[group_row],
                ]
                canonical = (group - translation) @ rotation
                self.assertTrue(torch.allclose(canonical, master, atol=1e-6))
        diagnostics = controller_a.diagnostics()
        self.assertEqual(
            [orbit["group_action_count"] for orbit in diagnostics["orbits"]],
            [6, 6],
        )

    def test_joint_energy_penalizes_inter_orbit_ca_clashes(self) -> None:
        (
            _,
            scaffold,
            topology,
            axis,
            config,
            controller,
        ) = self._two_orbit_scaffold_guidance_case()
        rotations = tuple(motif.state.rotation[0] for motif in controller.motifs)
        translations = tuple(motif.state.translation[0] for motif in controller.motifs)
        target = controller.materialize_target()[0]
        _, terms = controller._joint_scaffold_energy(
            target,
            scaffold[0],
            topology=topology,
            axis=axis,
            principal_axes=(axis.direction, axis.direction),
            rotations=rotations,
            translations=translations,
            config=ScaffoldGuidanceConfig(
                junction_weight=0.0,
                clash_weight=1.0,
                tilt_weight=0.0,
                prior_weight=0.0,
                clash_distance=5.0,
            ),
        )

        self.assertGreater(terms["inter_orbit_clash"], 0.0)
        self.assertLess(terms["minimum_inter_orbit_distance"], 5.0)

    def test_multiple_scaffold_orbits_update_atomically_and_order_independently(
        self,
    ) -> None:
        (
            _,
            scaffold,
            topology,
            axis,
            config,
            controller_a,
        ) = self._two_orbit_scaffold_guidance_case()
        (
            _,
            _,
            _,
            _,
            _,
            controller_b,
        ) = self._two_orbit_scaffold_guidance_case()
        controller_b.motifs.reverse()
        principal_axes = (axis.direction, axis.direction)

        observed_a = controller_a.update_orbits_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axes=principal_axes,
            config=config,
            apply_update=True,
        )
        observed_b = controller_b.update_orbits_from_scaffold(
            scaffold,
            progress=0.5,
            topology=topology,
            axis=axis,
            principal_axes=principal_axes,
            config=config,
            apply_update=True,
        )

        self.assertTrue(controller_a.last_update_applied)
        self.assertTrue(controller_b.last_update_applied)
        self.assertTrue(torch.allclose(observed_a, observed_b, atol=1e-8))
        for motif in controller_a.motifs:
            self.assertGreater(
                float(torch.linalg.vector_norm(motif.state.translation[0])),
                0.0,
            )
        self.assertFalse(
            torch.allclose(
                controller_a.motifs[0].state.translation,
                controller_a.motifs[1].state.translation,
                atol=1e-8,
            )
        )
        snapshot = controller_a.diagnostics()["trajectory"][-1]
        self.assertTrue(snapshot["atomic_joint_acceptance"])
        self.assertEqual(len(snapshot["orbit_proposals"]), 2)
        self.assertTrue(snapshot["accepted"])
        self.assertTrue(snapshot["applied"])
        self.assertLess(
            snapshot["proposed_energy"]["total"],
            snapshot["initial_energy"]["total"],
        )
        proposal_ids = {
            proposal["constraint_orbit_id"] for proposal in snapshot["orbit_proposals"]
        }
        self.assertEqual(
            proposal_ids,
            {"mobile_orbit_alpha", "mobile_orbit_beta"},
        )
        for proposal in snapshot["orbit_proposals"]:
            self.assertTrue(proposal["committed"])
            self.assertLess(
                proposal["objective"]["delta"]["total"],
                0.0,
            )

        for motif in controller_a.motifs:
            master = observed_a[:, motif.master_atom_indices]
            for group_row, transform_id_tensor in enumerate(motif.group_transform_ids):
                transform_id = int(transform_id_tensor.item())
                rotation, translation = controller_a.sym_transforms[transform_id]
                group = observed_a[
                    :,
                    motif.group_atom_indices[group_row],
                ]
                canonical = (group - translation) @ rotation
                self.assertTrue(torch.allclose(canonical, master, atol=1e-6))

    def test_joint_rejection_rolls_back_every_scaffold_orbit(self) -> None:
        (
            target,
            scaffold,
            topology,
            axis,
            config,
            controller,
        ) = self._two_orbit_scaffold_guidance_case()
        principal_axes = (axis.direction, axis.direction)

        def joint_result(total: float):
            return (
                torch.tensor(total, dtype=target.dtype),
                {
                    "total": total,
                    "junction": total,
                    "weighted_junction": total,
                    "clash": 0.0,
                    "weighted_clash": 0.0,
                    "maximum_junction_error": 0.0,
                    "minimum_clash_distance": 3.0,
                    "orbits": [],
                },
            )

        with patch.object(
            controller,
            "_joint_scaffold_energy",
            side_effect=(joint_result(1.0), joint_result(2.0)),
        ):
            observed = controller.update_orbits_from_scaffold(
                scaffold,
                progress=0.5,
                topology=topology,
                axis=axis,
                principal_axes=principal_axes,
                config=config,
                apply_update=True,
            )

        self.assertFalse(controller.last_update_applied)
        self.assertTrue(torch.allclose(observed, target, atol=1e-8))
        snapshot = controller.diagnostics()["trajectory"][-1]
        self.assertEqual(snapshot["joint_decision"], "rejected")
        self.assertFalse(snapshot["applied"])
        self.assertTrue(
            any(proposal["accepted"] for proposal in snapshot["orbit_proposals"])
        )
        self.assertTrue(
            all(not proposal["committed"] for proposal in snapshot["orbit_proposals"])
        )

    def test_controller_rejects_nonfinite_proposal(self) -> None:
        _, target, _, controller = self._controller_case()
        target[:, 0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            controller.update(target, progress=0.5)


if __name__ == "__main__":
    unittest.main()
