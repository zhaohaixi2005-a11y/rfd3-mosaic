import unittest
from unittest.mock import patch

import numpy as np
import torch

from rfd3.engine import (
    RFD3InferenceEngine,
    _requires_true_precision,
    _restore_inference_geometry_precision,
    _snapshot_inference_geometry,
)
from rfd3.inference.symmetry.symmetry_utils import (
    apply_symmetry_to_xyz_atomwise,
    build_symmetry_orbit_layout,
    expand_symmetry_coupled_displacements,
    project_symmetry_orbit_average,
    symmetry_orbit_mask_mismatch_count,
    symmetry_orbit_residual,
)
from rfd3.transforms.symmetry import AddSymmetryFeats


class _AnnotationArray:
    def __init__(self, annotations):
        self.annotations = annotations
        self.shape = (len(next(iter(annotations.values()))),)

    def get_annotation(self, name):
        return self.annotations[name]

    def get_annotation_categories(self):
        return tuple(self.annotations)


class SymmetryFeatureGroupsTestCase(unittest.TestCase):
    def test_interface_relation_expands_rfd3_selector_ranges(self) -> None:
        atom_array = _AnnotationArray(
            {
                "src_component": np.asarray(
                    ["A1", "A2", "B1", "B2", "A1", "A2", "B1", "B2"]
                ),
                "sym_transform_id": np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            }
        )
        features = AddSymmetryFeats.make_assembly_interface_relation_features(
            atom_array,
            [
                {
                    "edge_instance_id": "edge@0",
                    "left_source_components": ["A1-2"],
                    "right_source_components": ["B1-2"],
                    "source_copy_index": 0,
                    "target_copy_index": 1,
                    "required": True,
                    "satisfaction_stage": "output",
                    "target_geometry": {
                        "mode": "geometric_constraints",
                        "contacts": {
                            "min_heavy_atom_contacts": 2,
                            "cutoff": 5.0,
                        },
                        "coverage": {"mode": "auto"},
                    },
                }
            ],
        )

        self.assertEqual(
            features["assembly_interface_left_membership"].tolist(),
            [[True, True, False, False, False, False, False, False]],
        )
        self.assertEqual(
            features["assembly_interface_right_membership"].tolist(),
            [[False, False, False, False, False, False, True, True]],
        )
        self.assertEqual(
            features["assembly_interface_automatic_quality"].tolist(),
            [True],
        )

    def test_exact_orbit_sampler_requires_true_fabric_precision(
        self,
    ) -> None:
        self.assertTrue(
            _requires_true_precision(
                {"symmetry_state_mode": "orbit_average"}
            )
        )
        self.assertFalse(
            _requires_true_precision(
                {"symmetry_state_mode": "legacy"}
            )
        )
        self.assertFalse(_requires_true_precision(None))

    def test_exact_engine_wires_true_precision_into_fabric_trainer(
        self,
    ) -> None:
        with patch(
            "rfd3.engine.BaseInferenceEngine.__init__",
            return_value=None,
        ) as base_init:
            RFD3InferenceEngine(
                skip_existing=False,
                json_keys_subset=None,
                prevalidate_inputs=True,
                diffusion_batch_size=1,
                inference_sampler={
                    "symmetry_state_mode": "orbit_average",
                },
                specification={},
                global_prefix="",
                cleanup_guideposts=True,
                cleanup_virtual_atoms=True,
                read_sequence_from_sequence_head=False,
                output_full_json=False,
                dump_prediction_metadata_json=False,
                dump_trajectories=False,
                align_trajectory_structures=False,
                low_memory_mode=True,
                ckpt_path="unused.ckpt",
            )

        trainer_overrides = base_init.call_args.kwargs[
            "trainer_overrides"
        ]
        self.assertEqual(trainer_overrides["precision"], "32-true")

    @staticmethod
    def _c3_features(atoms_per_copy=2):
        angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        transforms = {}
        for transform_id, angle in enumerate(angles):
            rotation = torch.tensor(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )
            transforms[str(transform_id)] = (
                rotation,
                torch.zeros(3, dtype=torch.float64),
            )
        return {
            "sym_entity_id": torch.zeros(
                3 * atoms_per_copy,
                dtype=torch.long,
            ),
            "sym_transform_id": torch.repeat_interleave(
                torch.arange(3),
                atoms_per_copy,
            ),
            "is_sym_asu": torch.tensor(
                [True] * atoms_per_copy
                + [False] * (2 * atoms_per_copy)
            ),
            "sym_orbit_slot": torch.arange(
                atoms_per_copy,
                dtype=torch.long,
            ).repeat(3),
            "sym_orbit_slot_verified": torch.tensor(True),
            "sym_transform": transforms,
        }

    def test_transform_frames_do_not_depend_on_atom_order(self) -> None:
        vector_dtype = np.dtype([("x", np.float64, (3,))])
        origins = np.zeros(4, dtype=vector_dtype)
        x_axes = np.zeros(4, dtype=vector_dtype)
        y_axes = np.zeros(4, dtype=vector_dtype)
        transform_ids = np.asarray([0, 1, 0, 1])
        for index, transform_id in enumerate(transform_ids):
            translation = np.asarray(
                [10.0 * transform_id, 0.0, 0.0]
            )
            origins["x"][index] = translation
            x_axes["x"][index] = translation + np.asarray(
                [1.0, 0.0, 0.0]
            )
            y_axes["x"][index] = translation + np.asarray(
                [0.0, 1.0, 0.0]
            )
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": transform_ids,
                "sym_transform_Ori": origins,
                "sym_transform_X": x_axes,
                "sym_transform_Y": y_axes,
            }
        )

        transforms = AddSymmetryFeats().make_transforms_dict(atom_array)

        self.assertEqual(set(transforms), {"0", "1"})
        self.assertEqual(transforms["0"][0].shape, (3, 3))
        self.assertEqual(transforms["0"][1].shape, (3,))
        self.assertTrue(
            torch.allclose(
                transforms["0"][0],
                torch.eye(3, dtype=transforms["0"][0].dtype),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                transforms["1"][1],
                torch.tensor(
                    [10.0, 0.0, 0.0],
                    dtype=transforms["1"][1].dtype,
                ),
            )
        )

    def test_frame_roundtrip_rotations_are_normalized_for_exact_orbits(
        self,
    ) -> None:
        vector_dtype = np.dtype([("x", np.float64, (3,))])
        origins = np.zeros(3, dtype=vector_dtype)
        x_axes = np.zeros(3, dtype=vector_dtype)
        y_axes = np.zeros(3, dtype=vector_dtype)
        transform_ids = np.arange(3)
        for transform_id, angle in enumerate(
            (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ):
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            # Match RFD3's virtual-frame serialization exactly.  The 1e-6
            # normalization in frames.py makes the recovered matrices only
            # approximately orthogonal.
            origins["x"][transform_id] = 0.0
            x_axes["x"][transform_id] = rotation[0] / (
                np.linalg.norm(rotation[0]) + 1e-6
            )
            y_axes["x"][transform_id] = rotation[1] / (
                np.linalg.norm(rotation[1]) + 1e-6
            )
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": transform_ids,
                "sym_transform_Ori": origins,
                "sym_transform_X": x_axes,
                "sym_transform_Y": y_axes,
            }
        )
        transforms = AddSymmetryFeats().make_transforms_dict(atom_array)
        features = self._c3_features(atoms_per_copy=128)
        features["sym_transform"] = transforms

        raw_rotation = transforms["1"][0]
        raw_error = torch.max(
            torch.abs(
                raw_rotation @ raw_rotation.T
                - torch.eye(3, dtype=raw_rotation.dtype)
            )
        )
        self.assertGreater(float(raw_error.item()), 1e-6)

        raw = torch.randn(
            (4, 3 * 128, 3),
            generator=torch.Generator().manual_seed(20260729),
            dtype=torch.float32,
        ) * 2560.0
        layout = build_symmetry_orbit_layout(features, like=raw)
        for rotation, _ in layout.sym_transforms.values():
            identity = torch.eye(3, dtype=rotation.dtype)
            self.assertLess(
                float(
                    torch.max(
                        torch.abs(rotation @ rotation.T - identity)
                    ).item()
                ),
                1e-5,
            )
            self.assertAlmostEqual(
                float(torch.linalg.det(rotation).item()),
                1.0,
                places=5,
            )

        projected = project_symmetry_orbit_average(
            raw,
            features,
            layout=layout,
        )
        _, maximum = symmetry_orbit_residual(
            projected,
            features,
            layout=layout,
        )
        # Float32 roundoff at this deliberately extreme coordinate scale is
        # non-zero, but the systematic ~0.04 A frame error must be gone.
        self.assertLess(float(maximum.max().item()), 0.01)

    def test_bfloat16_transport_of_c3_frames_is_normalized(
        self,
    ) -> None:
        features = self._c3_features(atoms_per_copy=2)
        features["sym_transform"] = {
            transform_id: (
                rotation.to(torch.bfloat16),
                translation.to(torch.bfloat16),
            )
            for transform_id, (rotation, translation)
            in features["sym_transform"].items()
        }
        coordinates = torch.zeros((1, 6, 3), dtype=torch.bfloat16)

        layout = build_symmetry_orbit_layout(
            features,
            like=coordinates,
        )

        for rotation, _ in layout.sym_transforms.values():
            identity = torch.eye(3, dtype=rotation.dtype)
            self.assertLess(
                float(
                    torch.max(
                        torch.abs(rotation @ rotation.T - identity)
                    ).item()
                ),
                1e-5,
            )
            self.assertAlmostEqual(
                float(torch.linalg.det(rotation).item()),
                1.0,
                places=5,
            )

    def test_exact_geometry_is_promoted_after_fabric_transfer(
        self,
    ) -> None:
        features = self._c3_features(atoms_per_copy=2)
        coordinates = torch.full((1, 6, 3), 12.34567)
        source = {
            "coord_atom_lvl_to_be_noised": coordinates,
            "noise": coordinates / 10.0,
            "feats": {
                **features,
                "motif_constraint_target_coordinates": torch.zeros(
                    (1, 1, 6, 3),
                ),
            },
        }
        batch = {
            "coord_atom_lvl_to_be_noised": coordinates.to(torch.bfloat16),
            "noise": source["noise"].to(torch.bfloat16),
            "feats": {
                **source["feats"],
                "sym_transform": {
                    transform_id: (
                        rotation.to(torch.bfloat16),
                        translation.to(torch.bfloat16),
                    )
                    for transform_id, (rotation, translation)
                    in features["sym_transform"].items()
                },
                "motif_constraint_target_coordinates": source["feats"][
                    "motif_constraint_target_coordinates"
                ].to(torch.bfloat16),
            },
        }

        promoted = _restore_inference_geometry_precision(
            batch,
            source=source,
        )

        self.assertEqual(
            promoted["coord_atom_lvl_to_be_noised"].dtype,
            torch.float32,
        )
        self.assertTrue(
            torch.equal(
                promoted["coord_atom_lvl_to_be_noised"],
                coordinates,
            )
        )
        self.assertEqual(promoted["noise"].dtype, torch.float32)
        for rotation, translation in promoted["feats"][
            "sym_transform"
        ].values():
            self.assertEqual(rotation.dtype, torch.float32)
            self.assertEqual(translation.dtype, torch.float32)
        self.assertEqual(
            promoted["feats"][
                "motif_constraint_target_coordinates"
            ].dtype,
            torch.float32,
        )

    def test_exact_geometry_snapshot_survives_nested_replacement(
        self,
    ) -> None:
        features = self._c3_features(atoms_per_copy=2)
        coordinates = torch.full((1, 6, 3), 12.34567)
        batch = {
            "coord_atom_lvl_to_be_noised": coordinates,
            "noise": coordinates / 10.0,
            "feats": features,
        }
        snapshot = _snapshot_inference_geometry(batch)

        # Emulate a precision plugin replacing tensors inside the same nested
        # batch object after a caller retained only a shallow reference.
        batch["coord_atom_lvl_to_be_noised"] = coordinates.to(
            torch.bfloat16
        )
        batch["noise"] = batch["noise"].to(torch.bfloat16)
        batch["feats"]["sym_transform"] = {
            transform_id: (
                rotation.to(torch.bfloat16),
                translation.to(torch.bfloat16),
            )
            for transform_id, (rotation, translation)
            in batch["feats"]["sym_transform"].items()
        }

        promoted = _restore_inference_geometry_precision(
            batch,
            source=snapshot,
        )

        self.assertTrue(
            torch.equal(
                promoted["coord_atom_lvl_to_be_noised"],
                coordinates,
            )
        )
        self.assertEqual(
            promoted["coord_atom_lvl_to_be_noised"].dtype,
            torch.float32,
        )

    def test_optional_group_ids_become_membership_matrix(self) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0, 0, 1, 1]),
                "sym_entity_id": np.asarray([0, 0, 0, 0]),
                "is_sym_asu": np.asarray([True, True, False, False]),
                "symmetry_id": np.asarray(["C3", "C3", "C3", "C3"]),
                "motif_constraint_group_id": np.asarray([0, 0, 1, -1]),
                "src_component": np.asarray(["A1", "A2", "A1", "A2"]),
                "atom_name": np.asarray(["CA", "CA", "CA", "CA"]),
            }
        )
        transform = AddSymmetryFeats()
        transform.make_transforms_dict = lambda _: {}
        data = {"atom_array": atom_array, "feats": {}}

        output = transform.forward(data)

        self.assertEqual(output["feats"]["symmetry_id"], "C3")

        self.assertTrue(
            torch.equal(
                output["feats"]["motif_constraint_group_membership"],
                torch.tensor(
                    [
                        [True, True, False, False],
                        [False, False, True, False],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                output["feats"]["sym_orbit_slot"],
                torch.tensor([0, 1, 0, 1]),
            )
        )
        self.assertTrue(
            bool(output["feats"]["sym_orbit_slot_verified"].item())
        )

    def test_runtime_cross_chain_groups_resolve_after_symmetry(self) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0, 0, 1, 1]),
                "sym_entity_id": np.asarray([0, 0, 0, 0]),
                "is_sym_asu": np.asarray([True, True, False, False]),
                "src_component": np.asarray(
                    ["B1", "C1", "B1", "C1"]
                ),
                "atom_name": np.asarray(["CA", "CA", "CA", "CA"]),
                "is_motif_atom_with_fixed_coord": np.asarray(
                    [True, True, True, True]
                ),
            }
        )
        transform = AddSymmetryFeats()
        transform.make_transforms_dict = lambda _: {}
        data = {
            "atom_array": atom_array,
            "feats": {},
            "specification": {
                "extra": {
                    "motif_constraint_groups": [
                        {
                            "group_id": "A-B",
                            "members": [
                                {
                                    "role": "left",
                                    "src_components": ["B1"],
                                    "sym_transform_id": 0,
                                },
                                {
                                    "role": "right",
                                    "src_components": ["C1"],
                                    "sym_transform_id": 1,
                                },
                            ],
                        },
                        {
                            "group_id": "B-A",
                            "members": [
                                {
                                    "role": "left",
                                    "src_components": ["B1"],
                                    "sym_transform_id": 1,
                                },
                                {
                                    "role": "right",
                                    "src_components": ["C1"],
                                    "sym_transform_id": 0,
                                },
                            ],
                        },
                    ]
                }
            },
        }

        output = transform.forward(data)

        self.assertTrue(
            torch.equal(
                output["feats"]["motif_constraint_group_membership"],
                torch.tensor(
                    [
                        [True, False, False, True],
                        [False, True, True, False],
                    ]
                ),
            )
        )

    def test_runtime_central_fixed_motif_groups_resolve_by_copy(self) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0, 0, 1, 1, 2, 2]),
                "sym_entity_id": np.asarray([0, 0, 0, 0, 0, 0]),
                "is_sym_asu": np.asarray(
                    [True, True, False, False, False, False]
                ),
                "src_component": np.asarray(
                    ["B1", "B2", "B1", "B2", "B1", "B2"]
                ),
                "atom_name": np.asarray(["CA"] * 6),
                "is_motif_atom_with_fixed_coord": np.asarray([True] * 6),
            }
        )
        groups = [
            {
                "group_id": f"central@C3[{transform_id}]",
                "constraint_kind": "fixed_motif",
                "members": [
                    {
                        "role": "motif",
                        "src_components": ["B1", "B2"],
                        "sym_transform_id": transform_id,
                    }
                ],
            }
            for transform_id in range(3)
        ]

        membership = AddSymmetryFeats.make_motif_constraint_group_membership(
            atom_array,
            groups,
        )

        self.assertTrue(
            torch.equal(
                membership,
                torch.tensor(
                    [
                        [True, True, False, False, False, False],
                        [False, False, True, True, False, False],
                        [False, False, False, False, True, True],
                    ]
                ),
            )
        )

    def test_central_fixed_motif_group_rejects_interface_roles(self) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0]),
                "src_component": np.asarray(["B1"]),
                "is_motif_atom_with_fixed_coord": np.asarray([True]),
            }
        )

        with self.assertRaisesRegex(ValueError, "exactly the roles"):
            AddSymmetryFeats.make_motif_constraint_group_membership(
                atom_array,
                [
                    {
                        "group_id": "bad-central",
                        "constraint_kind": "fixed_motif",
                        "members": [
                            {
                                "role": "left",
                                "src_components": ["B1"],
                                "sym_transform_id": 0,
                            }
                        ],
                    }
                ],
            )

    def test_orbit_slots_follow_atom_keys_after_copy_reordering(
        self,
    ) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0, 0, 1, 1]),
                "sym_entity_id": np.asarray([0, 0, 0, 0]),
                "is_sym_asu": np.asarray([True, True, False, False]),
                "src_component": np.asarray(["A1", "A2", "A2", "A1"]),
                "atom_name": np.asarray(["CA", "CA", "CA", "CA"]),
            }
        )

        slots = AddSymmetryFeats.make_symmetry_orbit_slots(atom_array)

        self.assertTrue(
            torch.equal(slots, torch.tensor([0, 1, 1, 0]))
        )

    def test_orbit_slots_distinguish_residues_in_generated_block(
        self,
    ) -> None:
        atom_names = np.asarray(
            ["N", "CA", "C", "O", "CB"] * 4
        )
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0] * 10 + [1] * 10),
                "sym_entity_id": np.zeros(20, dtype=int),
                "is_sym_asu": np.asarray([True] * 10 + [False] * 10),
                "src_component": np.asarray(["70-100"] * 20),
                "res_id": np.asarray(
                    [32] * 5 + [33] * 5 + [32] * 5 + [33] * 5
                ),
                "atom_name": atom_names,
            }
        )

        slots, verified = AddSymmetryFeats.make_symmetry_orbit_slots(
            atom_array,
            return_verification=True,
        )

        self.assertTrue(verified)
        self.assertEqual(len(torch.unique(slots[:10])), 10)
        self.assertTrue(torch.equal(slots[:10], slots[10:]))

    def test_orbit_slots_reject_mismatched_copy_atom_keys(self) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0, 0, 1, 1]),
                "sym_entity_id": np.asarray([0, 0, 0, 0]),
                "is_sym_asu": np.asarray([True, True, False, False]),
                "src_component": np.asarray(["A1", "A2", "A1", "A3"]),
                "atom_name": np.asarray(["CA", "CA", "CA", "CA"]),
            }
        )

        with self.assertRaisesRegex(ValueError, "same atom"):
            AddSymmetryFeats.make_symmetry_orbit_slots(atom_array)

    def test_constraint_orbit_features_preserve_cross_copy_atom_slots(
        self,
    ) -> None:
        atom_array = _AnnotationArray(
            {
                "sym_transform_id": np.asarray([0, 0, 1, 1]),
                "sym_entity_id": np.asarray([0, 0, 0, 0]),
                "is_sym_asu": np.asarray([True, True, False, False]),
                "src_component": np.asarray(
                    ["B1", "C1", "B1", "C1"]
                ),
                "atom_name": np.asarray(["CA", "CA", "CA", "CA"]),
                "is_motif_atom_with_fixed_coord": np.asarray(
                    [True, True, True, True]
                ),
            }
        )
        groups = [
            {
                "group_id": "g0",
                "members": [
                    {
                        "role": "left",
                        "source_fragment_id": "left",
                        "src_components": ["B1"],
                        "sym_transform_id": 0,
                    },
                    {
                        "role": "right",
                        "source_fragment_id": "right",
                        "src_components": ["C1"],
                        "sym_transform_id": 1,
                    },
                ],
            },
            {
                "group_id": "g1",
                "members": [
                    {
                        "role": "left",
                        "source_fragment_id": "left",
                        "src_components": ["B1"],
                        "sym_transform_id": 1,
                    },
                    {
                        "role": "right",
                        "source_fragment_id": "right",
                        "src_components": ["C1"],
                        "sym_transform_id": 0,
                    },
                ],
            },
        ]
        orbits = [
            {
                "constraint_orbit_id": "seed__orbit",
                "coupling_group_id": "seed_component",
                "group_ids": ["g0", "g1"],
                "master_group_id": "g0",
                "group_transform_ids": [0, 1],
                "mobility_mode": "orbit_rigid",
                "max_translation": 2.5,
                "max_rotation_deg": 12.0,
                "mobility_subspace": "bounded_se3",
                "mobility_proposal": "scaffold_objectives",
                "mobility_objectives": ["junction", "assembly_clash"],
                "mobility_schedule": {
                    "start_fraction": 0.05,
                    "end_fraction": 0.70,
                    "response": 0.20,
                    "max_step_translation": 0.15,
                    "max_step_rotation_deg": 0.75,
                },
            }
        ]
        transform = AddSymmetryFeats()
        transform.make_transforms_dict = lambda _: {}
        data = {
            "atom_array": atom_array,
            "feats": {},
            "specification": {
                "extra": {
                    "motif_constraint_groups": groups,
                    "motif_constraint_orbits": orbits,
                }
            },
        }

        features = transform.forward(data)["feats"]

        self.assertTrue(
            torch.equal(
                features["motif_constraint_group_atom_indices"],
                torch.tensor([[0, 3], [2, 1]]),
            )
        )
        self.assertTrue(
            torch.equal(
                features[
                    "motif_constraint_orbit_master_group_index"
                ],
                torch.tensor([0]),
            )
        )
        self.assertTrue(
            torch.equal(
                features[
                    "motif_constraint_group_orbit_transform_id"
                ],
                torch.tensor([0, 1]),
            )
        )
        self.assertTrue(
            torch.equal(
                features["motif_constraint_orbit_subspace"],
                torch.tensor([4]),
            )
        )
        self.assertTrue(
            torch.equal(
                features["motif_constraint_orbit_proposal"],
                torch.tensor([2]),
            )
        )
        self.assertTrue(
            torch.allclose(
                features["motif_constraint_orbit_schedule"],
                torch.tensor([[0.05, 0.70, 0.20, 0.15, 0.75]]),
            )
        )
        self.assertEqual(
            features["motif_constraint_orbit_objective_ids"],
            (("junction", "assembly_clash"),),
        )
        self.assertEqual(
            features["motif_constraint_orbit_ids"],
            ("seed__orbit",),
        )
        self.assertEqual(
            features["motif_constraint_orbit_component_ids"],
            ("seed_component",),
        )

    def test_atomwise_projection_does_not_mutate_input(self) -> None:
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                ]
            ]
        )
        original = coordinates.clone()
        identity = torch.eye(3)
        features = {
            "sym_entity_id": torch.tensor([0, 0, 0, 0]),
            "sym_transform_id": torch.tensor([0, 0, 1, 1]),
            "is_sym_asu": torch.tensor([True, True, False, False]),
            "sym_transform": {
                "0": (identity, torch.zeros(3)),
                "1": (identity, torch.tensor([10.0, 0.0, 0.0])),
            },
        }

        projected = apply_symmetry_to_xyz_atomwise(
            coordinates,
            features,
        )

        self.assertTrue(torch.equal(coordinates, original))
        self.assertFalse(torch.equal(projected, original))

    def test_projection_rejects_mismatched_subunit_atom_counts(self) -> None:
        coordinates = torch.zeros((1, 3, 3))
        identity = torch.eye(3)
        features = {
            "sym_entity_id": torch.tensor([0, 0, 0]),
            "sym_transform_id": torch.tensor([0, 0, 1]),
            "is_sym_asu": torch.tensor([True, True, False]),
            "sym_transform": {
                "0": (identity, torch.zeros(3)),
                "1": (identity, torch.zeros(3)),
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "same number of atoms",
        ):
            apply_symmetry_to_xyz_atomwise(
                coordinates,
                features,
            )

    def test_runtime_projection_uses_atom_array_rotation_direction(
        self,
    ) -> None:
        features = self._c3_features(atoms_per_copy=1)
        coordinates = torch.tensor(
            [[[1.0, 0.0, 0.0]] * 3],
            dtype=torch.float64,
        )

        projected = apply_symmetry_to_xyz_atomwise(
            coordinates,
            features,
            partial_diffusion=True,
        )

        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [-0.5, np.sqrt(3.0) / 2.0, 0.0],
                    [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(projected, expected, atol=1e-6))

    def test_orbit_average_uses_every_c3_copy(self) -> None:
        features = self._c3_features(atoms_per_copy=1)
        canonical = torch.tensor(
            [[[2.0, 1.0, -0.5]]],
            dtype=torch.float64,
        )
        exact = apply_symmetry_to_xyz_atomwise(
            canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        )
        observed = exact.clone()
        # All three copies contribute different canonical-frame offsets.
        canonical_offsets = torch.tensor(
            [
                [0.3, 0.0, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.0, 0.9],
            ],
            dtype=torch.float64,
        )
        for transform_id in range(3):
            rotation = features["sym_transform"][str(transform_id)][0]
            observed[:, transform_id, :] += (
                canonical_offsets[transform_id] @ rotation.T
            )

        projected = project_symmetry_orbit_average(
            observed,
            features,
            partial_diffusion=True,
        )

        expected_canonical = canonical + canonical_offsets.mean(
            dim=0
        )[None, None, :]
        expected = apply_symmetry_to_xyz_atomwise(
            expected_canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        )
        self.assertTrue(torch.allclose(projected, expected, atol=1e-6))
        rms, maximum = symmetry_orbit_residual(projected, features)
        self.assertTrue(torch.all(rms < 1e-10))
        self.assertTrue(torch.all(maximum < 1e-10))

    def test_exact_orbit_operations_require_explicit_slots(self) -> None:
        features = self._c3_features(atoms_per_copy=1)
        features.pop("sym_orbit_slot")

        with self.assertRaisesRegex(ValueError, "sym_orbit_slot"):
            project_symmetry_orbit_average(
                torch.zeros((1, 3, 3), dtype=torch.float32),
                features,
            )

    def test_bfloat16_projection_keeps_float32_exact_state(self) -> None:
        features = self._c3_features(atoms_per_copy=1)
        canonical = torch.tensor(
            [[[21.25, -7.5, 3.0]]],
            dtype=torch.float32,
        )
        exact = apply_symmetry_to_xyz_atomwise(
            canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        ).to(torch.bfloat16)

        projected = project_symmetry_orbit_average(exact, features)
        rms, maximum = symmetry_orbit_residual(projected, features)

        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(rms.dtype, torch.float32)
        self.assertTrue(torch.all(maximum < 1e-4))

    def test_coupled_noise_rotates_one_asu_sample_without_translation(
        self,
    ) -> None:
        features = self._c3_features(atoms_per_copy=1)
        raw = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [100.0, 100.0, 100.0],
                    [-100.0, -100.0, -100.0],
                ]
            ],
            dtype=torch.float64,
        )

        coupled = expand_symmetry_coupled_displacements(raw, features)

        for transform_id in range(3):
            rotation = features["sym_transform"][str(transform_id)][0]
            expected = raw[:, :1, :] @ rotation.T
            self.assertTrue(
                torch.allclose(
                    coupled[:, transform_id : transform_id + 1, :],
                    expected,
                    atol=1e-6,
                )
            )

    def test_exact_orbit_supports_nonidentity_asu_and_translations(
        self,
    ) -> None:
        asu_rotation = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        target_rotation = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        features = {
            "sym_entity_id": torch.tensor([0, 0]),
            "sym_transform_id": torch.tensor([0, 1]),
            "is_sym_asu": torch.tensor([True, False]),
            "sym_orbit_slot": torch.tensor([0, 0]),
            "sym_orbit_slot_verified": torch.tensor(True),
            "sym_transform": {
                "0": (
                    asu_rotation,
                    torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64),
                ),
                "1": (
                    target_rotation,
                    torch.tensor([-3.0, 4.0, -0.5], dtype=torch.float64),
                ),
            },
        }
        canonical = torch.tensor(
            [[[1.5, -0.25, 2.0]]],
            dtype=torch.float64,
        )
        coordinates = torch.cat(
            [
                canonical @ asu_rotation.T
                + features["sym_transform"]["0"][1],
                canonical @ target_rotation.T
                + features["sym_transform"]["1"][1],
            ],
            dim=1,
        )

        projected = project_symmetry_orbit_average(
            coordinates,
            features,
        )
        self.assertTrue(
            torch.allclose(projected, coordinates, atol=1e-8)
        )

        raw_displacement = torch.tensor(
            [[[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]],
            dtype=torch.float64,
        )
        coupled = expand_symmetry_coupled_displacements(
            raw_displacement,
            features,
        )
        canonical_displacement = (
            raw_displacement[:, :1, :] @ asu_rotation
        )
        self.assertTrue(
            torch.allclose(
                coupled[:, 1:2, :],
                canonical_displacement @ target_rotation.T,
                atol=1e-8,
            )
        )

    def test_orbit_mask_closure_detects_transform_specific_fixed_atoms(
        self,
    ) -> None:
        features = self._c3_features(atoms_per_copy=2)
        closed = torch.tensor(
            [True, False, True, False, True, False]
        )
        broken = closed.clone()
        broken[3] = True

        self.assertEqual(
            symmetry_orbit_mask_mismatch_count(closed, features),
            0,
        )
        self.assertGreater(
            symmetry_orbit_mask_mismatch_count(broken, features),
            0,
        )

    def test_explicit_orbit_slots_support_interleaved_atom_order(
        self,
    ) -> None:
        blocked = self._c3_features(atoms_per_copy=2)
        features = {
            **blocked,
            "sym_transform_id": torch.tensor([0, 1, 2, 0, 1, 2]),
            "is_sym_asu": torch.tensor(
                [True, False, False, True, False, False]
            ),
            "sym_orbit_slot": torch.tensor([0, 0, 0, 1, 1, 1]),
        }
        canonical = torch.tensor(
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.5]],
            dtype=torch.float64,
        )
        coordinates = torch.empty((1, 6, 3), dtype=torch.float64)
        for atom_index, (transform_id, slot) in enumerate(
            zip(
                features["sym_transform_id"].tolist(),
                features["sym_orbit_slot"].tolist(),
            )
        ):
            rotation = features["sym_transform"][str(transform_id)][0]
            coordinates[0, atom_index] = canonical[slot] @ rotation.T

        projected = project_symmetry_orbit_average(
            coordinates,
            features,
        )

        self.assertTrue(torch.allclose(projected, coordinates, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
