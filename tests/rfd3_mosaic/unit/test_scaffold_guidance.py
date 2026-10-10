import math
import unittest
from dataclasses import replace

import torch
from rfd3.inference.symmetry.scaffold_guidance import (
    ScaffoldGuidanceConfig,
    _rotation_axis_angle,
    _rotation_from_vector,
    build_boundary_topology,
    expand_master_orbit,
    extract_cyclic_axis,
    extract_symmetry_primary_axis,
    propose_bounded_se3_step,
    scaffold_orbit_energy,
)


def _z_rotation(
    angle_degrees: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def _rotation_angle_degrees(rotation: torch.Tensor) -> float:
    cosine = torch.clamp(
        (torch.trace(rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return math.degrees(float(torch.acos(cosine)))


def _y_rotation(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    return torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=torch.float64,
    )


def _cyclic_transforms(
    order: int,
    *,
    point: torch.Tensor | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if point is None:
        point = torch.zeros(3, dtype=torch.float64)
    transforms = {}
    for transform_id in range(order):
        rotation = _z_rotation(360.0 * transform_id / order)
        translation = point - rotation @ point
        transforms[str(transform_id)] = (rotation, translation)
    return transforms


def _dihedral_transforms(
    order: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    transforms = _cyclic_transforms(order)
    secondary = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    zero = torch.zeros(3, dtype=torch.float64)
    for copy_index in range(order):
        rotation = _z_rotation(360.0 * copy_index / order) @ secondary
        transforms[str(order + copy_index)] = (rotation, zero)
    return transforms


def _boundary_features(
    order: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build Cn copies of fixed-generated-generated-fixed token chains."""

    tokens_per_chain = 4
    token_count = order * tokens_per_chain
    token_bonds = torch.zeros(
        (token_count, token_count),
        dtype=torch.bool,
    )
    fixed_mask = torch.zeros(token_count, dtype=torch.bool)
    for copy_index in range(order):
        start = copy_index * tokens_per_chain
        for offset in range(tokens_per_chain - 1):
            left = start + offset
            right = left + 1
            token_bonds[left, right] = True
            token_bonds[right, left] = True
        fixed_mask[start] = True
        fixed_mask[start + tokens_per_chain - 1] = True

    features = {
        "atom_to_token_map": torch.arange(token_count),
        "asym_id": torch.repeat_interleave(
            torch.arange(order),
            tokens_per_chain,
        ),
        "is_ca": torch.ones(token_count, dtype=torch.bool),
        "token_bonds": token_bonds,
    }
    return features, fixed_mask


class ScaffoldBoundaryTopologyTestCase(unittest.TestCase):
    def test_c3_and_c5_boundaries_are_found_per_copy(self) -> None:
        for order in (3, 5):
            with self.subTest(order=order):
                features, fixed_mask = _boundary_features(order)
                topology = build_boundary_topology(features, fixed_mask)

                observed = {
                    tuple(pair) for pair in topology.junction_pairs.cpu().tolist()
                }
                expected = set()
                for copy_index in range(order):
                    start = 4 * copy_index
                    expected.add((start, start + 1))
                    expected.add((start + 3, start + 2))

                self.assertEqual(observed, expected)
                self.assertEqual(
                    topology.generated_atom_mask.cpu().tolist(),
                    (~fixed_mask).tolist(),
                )
                self.assertEqual(
                    topology.fixed_ca_atom_indices.cpu().tolist(),
                    torch.nonzero(
                        fixed_mask,
                        as_tuple=False,
                    )
                    .flatten()
                    .tolist(),
                )
                self.assertEqual(
                    topology.generated_ca_atom_indices.cpu().tolist(),
                    torch.nonzero(
                        ~fixed_mask,
                        as_tuple=False,
                    )
                    .flatten()
                    .tolist(),
                )

    def test_sequence_adjacency_recovers_real_contig_boundaries(
        self,
    ) -> None:
        """Foundry may omit ordinary peptide edges from token_bonds."""

        order = 5
        tokens_per_chain = 4
        token_count = order * tokens_per_chain
        features = {
            "atom_to_token_map": torch.arange(token_count),
            "asym_id": torch.repeat_interleave(
                torch.arange(order),
                tokens_per_chain,
            ),
            "residue_index": torch.arange(tokens_per_chain).repeat(order),
            "is_ca": torch.ones(token_count, dtype=torch.bool),
            "is_protein": torch.ones(token_count, dtype=torch.bool),
            "token_bonds": torch.zeros(
                (token_count, token_count),
                dtype=torch.bool,
            ),
        }
        fixed_mask = torch.zeros(token_count, dtype=torch.bool)
        for copy_index in range(order):
            start = copy_index * tokens_per_chain
            fixed_mask[start] = True
            fixed_mask[start + tokens_per_chain - 1] = True

        topology = build_boundary_topology(features, fixed_mask)

        observed = {tuple(pair) for pair in topology.junction_pairs.cpu().tolist()}
        expected = set()
        for copy_index in range(order):
            start = copy_index * tokens_per_chain
            expected.add((start, start + 1))
            expected.add((start + 3, start + 2))
        self.assertEqual(observed, expected)

    def test_sequence_adjacency_does_not_cross_chain_or_residue_gap(
        self,
    ) -> None:
        features = {
            "atom_to_token_map": torch.arange(4),
            "asym_id": torch.tensor([0, 0, 1, 1]),
            "residue_index": torch.tensor([0, 2, 0, 1]),
            "is_ca": torch.ones(4, dtype=torch.bool),
            "is_protein": torch.ones(4, dtype=torch.bool),
            "token_bonds": torch.zeros((4, 4), dtype=torch.bool),
        }
        fixed_mask = torch.tensor([True, False, True, False])

        topology = build_boundary_topology(features, fixed_mask)

        self.assertEqual(
            topology.junction_pairs.cpu().tolist(),
            [[2, 3]],
        )

    def test_non_ca_atoms_are_not_used_as_boundary_representatives(
        self,
    ) -> None:
        token_bonds = torch.tensor(
            [
                [False, True],
                [True, False],
            ]
        )
        features = {
            "atom_to_token_map": torch.tensor([0, 0, 1, 1]),
            "asym_id": torch.tensor([0, 0]),
            "is_ca": torch.tensor([True, False, True, False]),
            "token_bonds": token_bonds,
        }
        fixed_mask = torch.tensor([True, True, False, False])

        topology = build_boundary_topology(features, fixed_mask)

        self.assertEqual(
            topology.junction_pairs.cpu().tolist(),
            [[0, 2]],
        )
        self.assertEqual(
            topology.generated_atom_mask.cpu().tolist(),
            [False, False, True, True],
        )

    def test_fixed_backbone_with_redesignable_sidechain_is_one_fixed_token(
        self,
    ) -> None:
        features = {
            "atom_to_token_map": torch.tensor([0, 0, 1, 1]),
            "asym_id": torch.tensor([0, 0]),
            "residue_index": torch.tensor([0, 1]),
            "is_ca": torch.tensor([True, False, True, False]),
            "is_protein": torch.tensor([True, True]),
            "token_bonds": torch.tensor(
                [
                    [False, True],
                    [True, False],
                ]
            ),
        }
        # Token 0 has a fixed CA/backbone but a redesignable side-chain atom.
        fixed_mask = torch.tensor([True, False, False, False])

        topology = build_boundary_topology(features, fixed_mask)

        self.assertEqual(topology.junction_pairs.tolist(), [[0, 2]])
        self.assertEqual(topology.fixed_ca_atom_indices.tolist(), [0])
        self.assertEqual(topology.generated_ca_atom_indices.tolist(), [2])
        self.assertEqual(
            topology.generated_atom_mask.tolist(),
            [False, False, True, True],
        )


class CyclicAxisTestCase(unittest.TestCase):
    def test_extracts_axis_and_off_origin_center_for_c3_and_c5(
        self,
    ) -> None:
        point = torch.tensor([2.5, -1.25, 0.75], dtype=torch.float64)
        for order in (3, 5):
            with self.subTest(order=order):
                axis = extract_cyclic_axis(_cyclic_transforms(order, point=point))

                self.assertAlmostEqual(
                    abs(
                        float(
                            torch.dot(
                                axis.direction,
                                torch.tensor(
                                    [0.0, 0.0, 1.0],
                                    dtype=torch.float64,
                                ),
                            )
                        )
                    ),
                    1.0,
                    places=6,
                )
                self.assertTrue(
                    torch.allclose(
                        axis.point[:2],
                        point[:2],
                        atol=1e-6,
                    )
                )
                self.assertEqual(
                    set(axis.transform_ids),
                    set(range(order)),
                )

    def test_rejects_transforms_without_a_nonidentity_rotation(self) -> None:
        identity = torch.eye(3, dtype=torch.float64)
        transforms = {
            "0": (identity, torch.zeros(3, dtype=torch.float64)),
            "1": (identity, torch.ones(3, dtype=torch.float64)),
        }

        with self.assertRaisesRegex(ValueError, "cyclic|rotation|axis"):
            extract_cyclic_axis(transforms)

    def test_extracts_d3_primary_axis_without_discarding_dihedral_group(
        self,
    ) -> None:
        transforms = _dihedral_transforms(3)

        axis = extract_symmetry_primary_axis(
            transforms,
            symmetry_id="D3",
        )

        self.assertEqual(axis.transform_ids, (0, 1, 2))
        self.assertAlmostEqual(
            abs(float(axis.direction[2])),
            1.0,
            places=6,
        )
        expanded = expand_master_orbit(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
            transforms,
        )
        self.assertEqual(tuple(expanded.shape), (6, 1, 1, 3))

    def test_d3_full_registry_is_not_misread_as_one_common_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "share one cyclic axis"):
            extract_cyclic_axis(_dihedral_transforms(3))

    def test_primary_axis_uses_native_dihedral_actions_not_transform_ids(self):
        from rfd3.inference.symmetry.frames import get_dihedral_frames

        frame_rotation = _y_rotation(37.0) @ _z_rotation(19.0)
        center = torch.tensor([4.2, -3.1, 2.7], dtype=torch.float64)
        for order in (3, 4, 10):
            native = get_dihedral_frames(order)
            for dtype in (torch.float32, torch.float64):
                # Moving identity away from zero and interleaving both cosets
                # must not change the physical axis or the selected subgroup.
                permutation = tuple(range(1, 2 * order)) + (0,)
                transforms = {}
                for index, source in enumerate(permutation):
                    rotation = frame_rotation @ torch.as_tensor(
                        native[source][0], dtype=torch.float64
                    ) @ frame_rotation.T
                    transforms[str(index)] = (
                        rotation.to(dtype), (center - rotation @ center).to(dtype)
                    )
                with self.subTest(order=order, dtype=dtype):
                    axis = extract_symmetry_primary_axis(
                        transforms, symmetry_id=f"D{order}"
                    )
                    self.assertEqual(
                        {permutation[i] for i in axis.transform_ids},
                        set(range(0, 2 * order, 2)),
                    )
                    expected_direction = frame_rotation[:, 2].to(dtype)
                    self.assertAlmostEqual(
                        abs(float(torch.dot(axis.direction, expected_direction))),
                        1.0, places=5,
                    )
                    offset = axis.point - center.to(dtype)
                    self.assertLess(float(torch.linalg.vector_norm(
                        offset - torch.dot(offset, axis.direction) * axis.direction
                    )), 1e-4)

    def test_d2_retains_declared_primary_axis_convention(self):
        axis = extract_symmetry_primary_axis(_dihedral_transforms(2), symmetry_id="D2")
        self.assertEqual(axis.transform_ids, (0, 1))
        self.assertAlmostEqual(abs(float(axis.direction[2])), 1.0)

    def test_axis_rejects_nonfinite_tolerance(self):
        for tolerance in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    extract_cyclic_axis(_cyclic_transforms(3), tolerance=tolerance)
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    extract_symmetry_primary_axis(
                        _dihedral_transforms(3), symmetry_id="D3", tolerance=tolerance
                    )


class CyclicMasterExpansionTestCase(unittest.TestCase):
    def test_master_is_expanded_by_each_c3_and_c5_group_action(
        self,
    ) -> None:
        master = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.5],
                    [1.5, -0.5, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        point = torch.tensor([0.25, -0.75, 0.0], dtype=torch.float64)
        for order in (3, 5):
            with self.subTest(order=order):
                transforms = _cyclic_transforms(order, point=point)
                expanded = expand_master_orbit(master, transforms)

                self.assertEqual(
                    tuple(expanded.shape),
                    (order, 1, 3, 3),
                )
                for transform_id in range(order):
                    rotation, translation = transforms[str(transform_id)]
                    expected = master @ rotation.T + translation
                    self.assertTrue(
                        torch.allclose(
                            expanded[transform_id],
                            expected,
                            atol=1e-7,
                        )
                    )
                    canonical = (expanded[transform_id] - translation) @ rotation
                    self.assertTrue(torch.allclose(canonical, master, atol=1e-7))


class ScaffoldOrbitEnergyTestCase(unittest.TestCase):
    @staticmethod
    def _energy_case():
        features, fixed_mask = _boundary_features(1)
        topology = build_boundary_topology(features, fixed_mask)
        axis = extract_cyclic_axis(_cyclic_transforms(3))
        motif_coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        scaffold_coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [3.8, 0.0, 0.0],
                    [6.2, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        return topology, axis, motif_coordinates, scaffold_coordinates

    def test_junction_energy_prefers_reachable_boundary_geometry(
        self,
    ) -> None:
        topology, axis, motif, scaffold = self._energy_case()
        good = scaffold_orbit_energy(
            motif,
            scaffold,
            topology,
            axis,
        )
        stretched = scaffold.clone()
        stretched[:, 1] = torch.tensor(
            [8.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        bad = scaffold_orbit_energy(
            motif,
            stretched,
            topology,
            axis,
        )

        self.assertTrue(torch.isfinite(good.total).all())
        self.assertLess(float(good.junction), float(bad.junction))
        self.assertEqual(good.junction_distances.numel(), 2)

    def test_clash_energy_detects_a_generated_atom_at_other_motif_atom(
        self,
    ) -> None:
        topology, axis, motif, scaffold = self._energy_case()
        separated = scaffold_orbit_energy(
            motif,
            scaffold,
            topology,
            axis,
        )
        clashing_scaffold = scaffold.clone()
        clashing_scaffold[:, 1] = torch.tensor(
            [9.5, 0.0, 0.0],
            dtype=torch.float64,
        )
        clashing = scaffold_orbit_energy(
            motif,
            clashing_scaffold,
            topology,
            axis,
        )

        self.assertLess(float(separated.clash), float(clashing.clash))
        self.assertLess(
            float(clashing.minimum_clash_distances.min()),
            1.0,
        )

    def test_tilt_energy_penalizes_a_principal_axis_past_the_limit(
        self,
    ) -> None:
        topology, axis, motif, scaffold = self._energy_case()
        principal_axis = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float64,
        )
        aligned = scaffold_orbit_energy(
            motif,
            scaffold,
            topology,
            axis,
            principal_axis=principal_axis,
            pose_rotation=torch.eye(3, dtype=torch.float64),
        )
        tilted = scaffold_orbit_energy(
            motif,
            scaffold,
            topology,
            axis,
            principal_axis=principal_axis,
            pose_rotation=_y_rotation(70.0),
        )

        self.assertAlmostEqual(float(aligned.tilt_degrees), 0.0, places=6)
        self.assertGreater(float(tilted.tilt_degrees), 60.0)
        self.assertLess(float(aligned.tilt), float(tilted.tilt))

    def test_axis_free_energy_keeps_non_tilt_terms(self) -> None:
        topology, _, motif, scaffold = self._energy_case()
        energy = scaffold_orbit_energy(
            motif,
            scaffold,
            topology,
            None,
        )

        self.assertTrue(torch.isfinite(energy.total))
        self.assertEqual(float(energy.tilt), 0.0)
        self.assertEqual(float(energy.tilt_degrees), 0.0)

    def test_principal_axis_without_symmetry_axis_is_rejected(self) -> None:
        topology, _, motif, scaffold = self._energy_case()
        with self.assertRaisesRegex(ValueError, "requires a Cn/Dn"):
            scaffold_orbit_energy(
                motif,
                scaffold,
                topology,
                None,
                principal_axis=torch.tensor(
                    [0.0, 0.0, 1.0],
                    dtype=torch.float64,
                ),
            )


class BoundedSE3ProposalTestCase(unittest.TestCase):
    def test_total_rotation_projection_cannot_jump_across_step_bound(self):
        current, target = _z_rotation(170.0), _z_rotation(190.0)
        proposal = propose_bounded_se3_step(
            current,
            torch.zeros(3, dtype=torch.float64),
            lambda r, t: (r - target).square().sum(),
            maximum_step_translation=0.0,
            maximum_step_rotation_degrees=15.0,
            maximum_total_translation=0.0,
            maximum_total_rotation_degrees=170.0,
        )
        actual = math.degrees(
            float(_rotation_axis_angle(proposal.rotation @ current.T)[1])
        )
        self.assertLessEqual(actual, 15.0 + 1e-5)

    def test_initial_pose_outside_total_bound_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "current pose.*bound"):
            propose_bounded_se3_step(
                torch.eye(3, dtype=torch.float64),
                torch.tensor([5.0, 0.0, 0.0]),
                lambda r, t: t.square().sum(),
                maximum_step_translation=0.1,
                maximum_step_rotation_degrees=1.0,
                maximum_total_translation=2.0,
                maximum_total_rotation_degrees=10.0,
            )

    def test_rotation_step_measurement_preserves_zero_and_small_float32_angles(self):
        rotation = _rotation_from_vector(torch.tensor([0.05, 0.025, 0.05 / 3]))
        self.assertLess(float(_rotation_axis_angle(rotation @ rotation.T)[1]), 1e-7)
        for angle in (0.0001, 0.03, math.pi):
            with self.subTest(angle=angle):
                value = _rotation_from_vector(torch.tensor([angle, 0.0, 0.0]))
                self.assertAlmostEqual(
                    float(_rotation_axis_angle(value)[1]), angle, places=6
                )

    def test_invalid_current_pose_is_not_returned_as_a_valid_noop(self):
        identity = torch.eye(3, dtype=torch.float64)
        zero = torch.zeros(3, dtype=torch.float64)
        for rotation, translation in (
            (identity * float("nan"), zero),
            (identity, zero + float("inf")),
            (-identity, zero),
            (2.0 * identity, zero),
        ):
            with self.subTest(rotation=rotation, translation=translation):
                with self.assertRaises(ValueError):
                    propose_bounded_se3_step(
                        rotation,
                        translation,
                        lambda r, t: torch.tensor(0.0),
                        maximum_step_translation=0.1,
                        maximum_step_rotation_degrees=1.0,
                        maximum_total_translation=2.0,
                        maximum_total_rotation_degrees=10.0,
                    )

    def test_weight_diagnostics_expose_opposing_forces_at_a_stalled_pose(self):
        topology, axis, motif, scaffold = ScaffoldOrbitEnergyTestCase._energy_case()
        template = scaffold_orbit_energy(motif, scaffold, topology, axis)

        def objective(rotation, translation):
            terms = {
                "left": (translation[0] + 1.0).square(),
                "right": (translation[0] - 1.0).square(),
            }
            return replace(template, total=sum(terms.values()), weighted_terms=terms)

        proposal = propose_bounded_se3_step(
            torch.eye(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            objective,
            maximum_step_translation=0.1,
            maximum_step_rotation_degrees=1.0,
            maximum_total_translation=2.0,
            maximum_total_rotation_degrees=10.0,
        )
        self.assertFalse(proposal.accepted)
        self.assertEqual(proposal.translation_gradient_norm, 0.0)
        terms = proposal.objective_gradients
        self.assertEqual(
            terms["left"]["allowed_translation_gradient_vector"], [2.0, 0.0, 0.0]
        )
        self.assertEqual(
            terms["right"]["allowed_translation_gradient_vector"], [-2.0, 0.0, 0.0]
        )

    def test_constant_objective_is_a_valid_noop(self):
        rotation = torch.eye(3, dtype=torch.float64)
        translation = torch.zeros(3, dtype=torch.float64)
        proposal = propose_bounded_se3_step(
            rotation,
            translation,
            lambda r, t: torch.tensor(4.0, dtype=t.dtype),
            maximum_step_translation=0.1,
            maximum_step_rotation_degrees=1.0,
            maximum_total_translation=2.0,
            maximum_total_rotation_degrees=10.0,
        )
        self.assertFalse(proposal.accepted)
        self.assertTrue(torch.equal(proposal.rotation, rotation))
        self.assertTrue(torch.equal(proposal.translation, translation))
        self.assertEqual(proposal.translation_gradient_norm, 0.0)

    def test_weight_diagnostics_distinguish_large_constants_from_allowed_force(self):
        topology, axis, motif, scaffold = ScaffoldOrbitEnergyTestCase._energy_case()
        template = scaffold_orbit_energy(motif, scaffold, topology, axis)

        def objective(rotation, translation):
            terms = {
                "constant": translation.sum() * 0.0 + 1000.0,
                "drive": 2.0 * (translation[0] - 0.2).square(),
                "blocked": 5.0 * (translation[1] - 1.0).square(),
            }
            return replace(template, total=sum(terms.values()), weighted_terms=terms)

        proposal = propose_bounded_se3_step(
            torch.eye(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            objective,
            maximum_step_translation=1.0,
            maximum_step_rotation_degrees=0.0,
            maximum_total_translation=2.0,
            maximum_total_rotation_degrees=0.0,
            translation_basis=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        )
        self.assertTrue(proposal.accepted)
        report = proposal.objective_gradients
        self.assertEqual(report["constant"]["weighted_energy"], 1000.0)
        self.assertEqual(report["constant"]["linear_step_sensitivity"], 0.0)
        self.assertEqual(report["blocked"]["linear_step_sensitivity"], 0.0)
        self.assertAlmostEqual(
            report["drive"]["allowed_translation_gradient_per_angstrom"], 0.8
        )
        for trial in proposal.line_search_trials:
            terms = trial["objective_terms"]
            self.assertAlmostEqual(
                sum(v for k, v in terms.items() if k.startswith("weighted_")),
                trial["energy"],
            )

    def test_small_feasible_descent_is_not_lost_below_quarter_step(self):
        for multistart in (False, True):
            with self.subTest(multistart=multistart):
                proposal = propose_bounded_se3_step(
                    torch.eye(3, dtype=torch.float64),
                    torch.zeros(3, dtype=torch.float64),
                    lambda rotation, translation: (translation[0] - 0.04).square(),
                    maximum_step_translation=1.0,
                    maximum_step_rotation_degrees=0.0,
                    maximum_total_translation=2.0,
                    maximum_total_rotation_degrees=0.0,
                    deterministic_multistart=multistart,
                    candidate_validator=lambda rotation, translation: {
                        "accepted": float(translation[0]) <= 0.05,
                    },
                )
                self.assertTrue(proposal.accepted)
                self.assertGreater(float(proposal.translation[0]), 0.0)
                self.assertLessEqual(float(proposal.translation[0]), 0.05)
                self.assertLess(
                    float(proposal.proposed_energy), float(proposal.initial_energy)
                )

    def test_nonfinite_penalty_coefficients_are_rejected(self):
        for field in ("junction_weight", "prior_weight", "translation_prior_scale"):
            for value in (float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        ScaffoldGuidanceConfig(**{field: value})

    def test_unsafe_full_step_backtracks_to_safe_improving_pose(self):
        proposal = propose_bounded_se3_step(
            torch.eye(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            lambda rotation, translation: (translation[0] - 2).square(),
            maximum_step_translation=1.0,
            maximum_step_rotation_degrees=1.0,
            maximum_total_translation=10.0,
            maximum_total_rotation_degrees=10.0,
            candidate_validator=lambda rotation, translation: {
                "accepted": float(translation[0]) <= 0.6,
            },
        )
        self.assertTrue(proposal.accepted)
        self.assertAlmostEqual(float(proposal.translation[0]), 0.5)
        self.assertFalse(proposal.line_search_trials[0]["geometry_guard"]["accepted"])
        self.assertTrue(proposal.line_search_trials[1]["geometry_guard"]["accepted"])

    def test_seeded_capture_samples_reproducibly_from_near_optimal_pool(self) -> None:
        identity = torch.eye(3, dtype=torch.float64)

        def symmetric_double_well(_rotation, translation):
            return torch.square(torch.square(translation[0]) - 0.25**2)

        observed = {}
        for seed in range(16):
            proposal = propose_bounded_se3_step(
                identity,
                torch.zeros(3, dtype=torch.float64),
                symmetric_double_well,
                maximum_step_translation=0.25,
                maximum_step_rotation_degrees=0.0,
                maximum_total_translation=1.0,
                maximum_total_rotation_degrees=0.0,
                translation_step_size=0.25,
                rotation_step_size_degrees=0.0,
                deterministic_multistart=True,
                selection_seed=seed,
            )
            observed[seed] = float(proposal.translation[0])
            self.assertEqual(
                sum(bool(trial["selected"]) for trial in proposal.line_search_trials),
                1,
            )

        replay = propose_bounded_se3_step(
            identity,
            torch.zeros(3, dtype=torch.float64),
            symmetric_double_well,
            maximum_step_translation=0.25,
            maximum_step_rotation_degrees=0.0,
            maximum_total_translation=1.0,
            maximum_total_rotation_degrees=0.0,
            translation_step_size=0.25,
            rotation_step_size_degrees=0.0,
            deterministic_multistart=True,
            selection_seed=7,
        )
        self.assertEqual(float(replay.translation[0]), observed[7])
        self.assertEqual(set(observed.values()), {-0.25, 0.25})

        overflow_replay = propose_bounded_se3_step(
            identity,
            torch.zeros(3, dtype=torch.float64),
            symmetric_double_well,
            maximum_step_translation=0.25,
            maximum_step_rotation_degrees=0.0,
            maximum_total_translation=1.0,
            maximum_total_rotation_degrees=0.0,
            deterministic_multistart=True,
            selection_seed=(1 << 64) + 7,
        )
        self.assertEqual(float(overflow_replay.translation[0]), observed[7])

    def test_capture_multistart_escapes_zero_gradient_pose(self) -> None:
        identity = torch.eye(3, dtype=torch.float64)

        def double_well(_rotation, translation):
            return torch.square(torch.square(translation[0]) - 0.25**2)

        local = propose_bounded_se3_step(
            identity,
            torch.zeros(3, dtype=torch.float64),
            double_well,
            maximum_step_translation=0.25,
            maximum_step_rotation_degrees=0.0,
            maximum_total_translation=1.0,
            maximum_total_rotation_degrees=0.0,
            translation_step_size=0.25,
            rotation_step_size_degrees=0.0,
        )
        capture = propose_bounded_se3_step(
            identity,
            torch.zeros(3, dtype=torch.float64),
            double_well,
            maximum_step_translation=0.25,
            maximum_step_rotation_degrees=0.0,
            maximum_total_translation=1.0,
            maximum_total_rotation_degrees=0.0,
            translation_step_size=0.25,
            rotation_step_size_degrees=0.0,
            deterministic_multistart=True,
        )

        self.assertFalse(local.accepted)
        self.assertTrue(capture.accepted)
        self.assertAlmostEqual(abs(float(capture.translation[0])), 0.25)
        self.assertLess(float(capture.proposed_energy), float(capture.initial_energy))

    def test_proposal_lowers_energy_and_respects_step_and_total_bounds(
        self,
    ) -> None:
        current_rotation = _z_rotation(4.5)
        current_translation = torch.tensor(
            [0.45, 0.0, 0.0],
            dtype=torch.float64,
        )
        target_rotation = _z_rotation(20.0)
        target_translation = torch.tensor(
            [2.0, 1.0, 0.0],
            dtype=torch.float64,
        )

        def energy_function(rotation, translation):
            return torch.sum(
                torch.square(translation - target_translation)
            ) + torch.sum(torch.square(rotation - target_rotation))

        proposal = propose_bounded_se3_step(
            current_rotation,
            current_translation,
            energy_function,
            maximum_step_translation=0.20,
            maximum_step_rotation_degrees=2.0,
            maximum_total_translation=0.50,
            maximum_total_rotation_degrees=5.0,
            translation_step_size=1.0,
            rotation_step_size_degrees=10.0,
        )

        self.assertTrue(proposal.accepted)
        self.assertLess(
            float(proposal.proposed_energy),
            float(proposal.initial_energy),
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(proposal.delta_translation)),
            0.20 + 1e-8,
        )
        self.assertLessEqual(
            _rotation_angle_degrees(proposal.delta_rotation),
            2.0 + 1e-6,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(proposal.translation)),
            0.50 + 1e-8,
        )
        self.assertLessEqual(
            _rotation_angle_degrees(proposal.rotation),
            5.0 + 1e-6,
        )

    def test_proposal_respects_translation_and_rotation_bases(self) -> None:
        identity = torch.eye(3, dtype=torch.float64)
        target_translation = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float64,
        )

        proposal = propose_bounded_se3_step(
            identity,
            torch.zeros(3, dtype=torch.float64),
            lambda rotation, translation: (
                torch.sum(torch.square(translation - target_translation))
                + torch.sum(torch.square(rotation - _z_rotation(20.0)))
            ),
            maximum_step_translation=0.25,
            maximum_step_rotation_degrees=0.0,
            maximum_total_translation=1.0,
            maximum_total_rotation_degrees=0.0,
            translation_step_size=0.25,
            rotation_step_size_degrees=0.0,
            translation_basis=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            rotation_basis=torch.empty((0, 3), dtype=torch.float64),
        )

        self.assertTrue(proposal.accepted)
        self.assertAlmostEqual(float(proposal.translation[1]), 0.0)
        self.assertTrue(torch.equal(proposal.rotation, identity))

    def test_rejects_a_nonfinite_objective(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite|NaN|Inf"):
            propose_bounded_se3_step(
                torch.eye(3, dtype=torch.float64),
                torch.zeros(3, dtype=torch.float64),
                lambda _rotation, _translation: torch.tensor(float("nan")),
                maximum_step_translation=0.20,
                maximum_step_rotation_degrees=2.0,
                maximum_total_translation=0.50,
                maximum_total_rotation_degrees=5.0,
            )


if __name__ == "__main__":
    unittest.main()
