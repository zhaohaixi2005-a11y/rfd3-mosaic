import json
import math
import unittest
from unittest.mock import patch

import torch

from rfd3.inference.symmetry.motif_mobility import (
    OrbitRigidMotifController,
    fit_centered_rigid_pose,
    mobility_window_weight,
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
        proposal = (
            (template - center) @ rotation.T
            + center
            + translation
        )

        fitted_rotation, fitted_translation, rmsd = (
            fit_centered_rigid_pose(template, proposal)
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
            target[:, 4 * transform_id : 4 * (transform_id + 1)] = (
                _apply(
                    template,
                    transforms[str(transform_id)][0],
                    transforms[str(transform_id)][1],
                )
            )
        features = {
            "sym_transform": transforms,
            "motif_constraint_group_orbit_index": torch.tensor(
                [0, 0, 0]
            ),
            "motif_constraint_group_orbit_transform_id": torch.tensor(
                [0, 1, 2]
            ),
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
            "motif_constraint_orbit_master_group_index": torch.tensor(
                [0]
            ),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "motif_constraint_orbit_bounds": torch.tensor(
                [[2.0, 10.0]]
            ),
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
            raw[:, 4 * transform_id : 4 * (transform_id + 1)] = (
                _apply(
                    desired_master,
                    transforms[str(transform_id)][0],
                    transforms[str(transform_id)][1],
                )
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
            inverse = (
                group - transforms[str(transform_id)][1]
            ) @ transforms[str(transform_id)][0]
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
            raw[:, 4 * transform_id : 4 * (transform_id + 1)] = (
                _apply(
                    desired_master,
                    transforms[str(transform_id)][0],
                    transforms[str(transform_id)][1],
                )
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
        initial_rotation = (
            controller.motifs[0].state.rotation.clone()
        )
        initial_translation = (
            controller.motifs[0].state.translation.clone()
        )

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
            self.assertTrue(
                torch.allclose(canonical, master, atol=1e-6)
            )

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
            target[:, 4 * transform_id : 4 * (transform_id + 1)] = (
                _apply(template, rotation, translation)
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
            "motif_constraint_orbit_bounds": torch.tensor(
                [[2.0, 10.0]]
            ),
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
            self.assertTrue(
                torch.allclose(canonical, master, atol=1e-6)
            )

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
            template + torch.tensor(
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
            "motif_constraint_group_atom_indices": torch.arange(
                atom_count
            ).reshape(2 * group_action_count, 4),
            "motif_constraint_group_atom_mask": torch.ones(
                (2 * group_action_count, 4),
                dtype=torch.bool,
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor(
                [0, group_action_count]
            ),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1, 1]),
            "motif_constraint_orbit_bounds": torch.tensor(
                [[2.0, 10.0], [2.0, 10.0]]
            ),
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
            junction_pairs=torch.tensor(
                [[0, 1], [atoms_per_orbit, second_generated]]
            ),
            fixed_ca_atom_indices=torch.tensor(
                [0, atoms_per_orbit]
            ),
            generated_ca_atom_indices=torch.tensor(
                [1, second_generated]
            ),
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
            for group_row, transform_id_tensor in enumerate(
                motif.group_transform_ids
            ):
                transform_id = int(transform_id_tensor.item())
                rotation, translation = controller_a.sym_transforms[
                    transform_id
                ]
                group = observed_a[
                    :,
                    motif.group_atom_indices[group_row],
                ]
                canonical = (group - translation) @ rotation
                self.assertTrue(
                    torch.allclose(canonical, master, atol=1e-6)
                )
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
        rotations = tuple(
            motif.state.rotation[0] for motif in controller.motifs
        )
        translations = tuple(
            motif.state.translation[0] for motif in controller.motifs
        )
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
            proposal["constraint_orbit_id"]
            for proposal in snapshot["orbit_proposals"]
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
            for group_row, transform_id_tensor in enumerate(
                motif.group_transform_ids
            ):
                transform_id = int(transform_id_tensor.item())
                rotation, translation = controller_a.sym_transforms[
                    transform_id
                ]
                group = observed_a[
                    :,
                    motif.group_atom_indices[group_row],
                ]
                canonical = (group - translation) @ rotation
                self.assertTrue(
                    torch.allclose(canonical, master, atol=1e-6)
                )

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
            any(
                proposal["accepted"]
                for proposal in snapshot["orbit_proposals"]
            )
        )
        self.assertTrue(
            all(
                not proposal["committed"]
                for proposal in snapshot["orbit_proposals"]
            )
        )

    def test_controller_rejects_nonfinite_proposal(self) -> None:
        _, target, _, controller = self._controller_case()
        target[:, 0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            controller.update(target, progress=0.5)


if __name__ == "__main__":
    unittest.main()
