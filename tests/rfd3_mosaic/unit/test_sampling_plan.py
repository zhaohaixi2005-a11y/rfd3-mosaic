import unittest

from pydantic import ValidationError

from rfd3_mosaic.sampling_plan import (
    assembly_initialization_payload,
    compile_sampling_plan,
    design_sampling_assignments,
)
from rfd3_mosaic.schema import UserDesignSpec


def design(**sampling: object) -> UserDesignSpec:
    return UserDesignSpec.model_validate(
        {
            "name": "sampling-plan",
            "input": "motif.pdb",
            "symmetry": "C3",
            "sampling": sampling,
        }
    )


class SamplingPlanTestCase(unittest.TestCase):
    def test_omitted_initial_pose_preserves_input_coordinates(self) -> None:
        plan = compile_sampling_plan(design(seed=17, timesteps=50))

        self.assertIsNone(plan.initial_pose)
        self.assertEqual(plan.diffusion.seed, 17)
        self.assertEqual(plan.diffusion.timesteps, 50)
        self.assertEqual(plan.diffusion.designs, 1)
        self.assertEqual(assembly_initialization_payload(plan), (None, {}))

    def test_compiles_user_requested_design_count(self) -> None:
        plan = compile_sampling_plan(
            design(
                seed=17,
                timesteps=50,
                designs=1000,
                dump_trajectories=True,
            )
        )

        self.assertEqual(plan.diffusion.designs, 1000)
        self.assertEqual(plan.diffusion.replicates_per_pose, 1)
        self.assertEqual(plan.diffusion.screening_mode, "advisory")
        self.assertEqual(plan.diffusion.screening_protocol, "auto")
        self.assertTrue(plan.diffusion.retain_all_outputs)
        self.assertTrue(plan.diffusion.dump_trajectories)

    def test_variable_pose_is_resampled_per_design_by_default(self) -> None:
        plan = compile_sampling_plan(
            design(
                designs=3,
                seed=200,
                initial_pose={
                    "radius": {"minimum": 20.0, "maximum": 30.0},
                    "orientation": {"method": "uniform_so3"},
                    "seed": 100,
                },
            )
        )

        assignments = design_sampling_assignments(plan)

        self.assertEqual(
            [item.pose_seed for item in assignments],
            [100, 101, 102],
        )
        self.assertEqual(
            [item.diffusion_seed for item in assignments],
            [200, 201, 202],
        )
        self.assertEqual(
            [item.pose_index for item in assignments],
            [0, 1, 2],
        )

    def test_replicates_per_pose_is_an_explicit_expert_control(self) -> None:
        plan = compile_sampling_plan(
            design(
                designs=5,
                replicates_per_pose=2,
                seed=200,
                initial_pose={
                    "radius": {"minimum": 20.0, "maximum": 30.0},
                    "orientation": {"method": "uniform_so3"},
                    "seed": 100,
                },
            )
        )

        assignments = design_sampling_assignments(plan)

        self.assertEqual(
            [item.pose_index for item in assignments],
            [0, 0, 1, 1, 2],
        )
        self.assertEqual(
            [item.pose_seed for item in assignments],
            [100, 100, 101, 101, 102],
        )

    def test_fixed_pose_keeps_one_pose_and_varies_diffusion_only(self) -> None:
        plan = compile_sampling_plan(
            design(
                designs=3,
                seed=200,
                initial_pose={
                    "radius": {"minimum": 25.0, "maximum": 25.0},
                    "axial_offset": {"minimum": 0.0, "maximum": 0.0},
                    "orientation": {
                        "method": "fixed",
                        "rotation_deg": [0.0, 0.0, 0.0],
                    },
                    "seed": 100,
                },
            )
        )

        assignments = design_sampling_assignments(plan)

        self.assertEqual({item.pose_index for item in assignments}, {0})
        self.assertEqual({item.pose_seed for item in assignments}, {None})
        self.assertEqual(
            [item.diffusion_seed for item in assignments],
            [200, 201, 202],
        )

    def test_compiles_explicit_symmetric_scaffold_packing(self) -> None:
        packed_design = UserDesignSpec.model_validate(
            {
                "name": "packed-sampling-plan",
                "input": "motif.pdb",
                "symmetry": "C3",
                "generation": [
                    {
                        "kind": "between",
                        "from_selector": "A1",
                        "to_selector": "B2",
                        "length": 20,
                    }
                ],
                "constraints": [
                    {"kind": "fixed_xyz", "selector": "A1"},
                    {"kind": "fixed_xyz", "selector": "B2"},
                ],
                "sampling": {
                    "scaffold_packing": "symmetric_generated"
                },
            }
        )
        plan = compile_sampling_plan(packed_design)

        self.assertEqual(
            plan.diffusion.scaffold_packing,
            "symmetric_generated",
        )

    def test_supplied_interface_can_explicitly_request_second_interface(
        self,
    ) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "supplied-interface-higher-order-oligomer",
                "task": "preserve_supplied_geometry",
                "input": "seed.pdb",
                "symmetry": "C3",
                "generation": [
                    {
                        "kind": "terminal",
                        "anchor": "A1",
                        "terminus": "c",
                        "length": 20,
                    }
                ],
                "constraints": [
                    {
                        "kind": "fixed_xyz",
                        "selector": "A1",
                        "coupling_group": "supplied_interface",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1",
                        "coupling_group": "supplied_interface",
                    },
                ],
                "sampling": {"scaffold_packing": "symmetric_generated"},
            }
        )

        self.assertEqual(
            compile_sampling_plan(design).diffusion.scaffold_packing,
            "symmetric_generated",
        )

    def test_rejects_nonpositive_design_count(self) -> None:
        with self.assertRaises(ValidationError):
            design(designs=0)

    def test_rejects_more_replicates_than_designs(self) -> None:
        with self.assertRaises(ValidationError):
            design(designs=2, replicates_per_pose=3)

    def test_allows_masked_sequence_with_native_sidechain_redesign(self) -> None:
        declared = UserDesignSpec.model_validate(
            {
                "name": "masked-redesign-conditioning",
                "input": "motif.pdb",
                "symmetry": "C3",
                "conditioning": {
                    "sequence": [
                        {"selector": "A1-10", "mode": "masked"}
                    ],
                    "redesign_motif_sidechains": True,
                },
            }
        )

        self.assertTrue(declared.conditioning.redesign_motif_sidechains)
        self.assertEqual(declared.conditioning.sequence[0].mode.value, "masked")

    def test_rejects_glycine_with_native_sidechain_redesign(self) -> None:
        with self.assertRaises(ValidationError):
            UserDesignSpec.model_validate(
                {
                    "name": "glycine-redesign-conditioning",
                    "input": "motif.pdb",
                    "symmetry": "C3",
                    "conditioning": {
                        "sequence": [
                            {"selector": "A1-10", "mode": "glycine"}
                        ],
                        "redesign_motif_sidechains": True,
                    },
                }
            )

    def test_hotspot_origin_requires_hotspots(self) -> None:
        with self.assertRaises(ValidationError):
            UserDesignSpec.model_validate(
                {
                    "name": "missing-hotspot-conditioning",
                    "input": "motif.pdb",
                    "symmetry": "C3",
                    "conditioning": {"origin_strategy": "hotspots"},
                }
            )

    def test_compiles_static_radius_axial_and_uniform_so3(self) -> None:
        plan = compile_sampling_plan(
            design(
                initial_pose={
                    "radius": {"minimum": 20.0, "maximum": 30.0},
                    "axial_offset": {"minimum": -3.0, "maximum": 5.0},
                    "orientation": {"method": "uniform_so3"},
                    "seed": 3000,
                }
            )
        )

        seed, payload = assembly_initialization_payload(plan)
        self.assertEqual(seed, 3000)
        initialization = payload["motif_group"]
        self.assertEqual(
            initialization["placement"]["radius"],
            {"mean": 25.0, "range": 5.0},
        )
        self.assertEqual(
            initialization["placement"]["axial_offset"],
            {"mean": 1.0, "range": 4.0},
        )
        self.assertEqual(
            initialization["orientation"],
            {"method": "uniform_so3"},
        )

    def test_fixed_orientation_is_retained_exactly(self) -> None:
        plan = compile_sampling_plan(
            design(
                initial_pose={
                    "radius": {"minimum": 24.0, "maximum": 24.0},
                    "orientation": {
                        "method": "fixed",
                        "rotation_deg": [10.0, 20.0, 30.0],
                    },
                }
            )
        )

        _, payload = assembly_initialization_payload(plan)
        self.assertEqual(
            payload["motif_group"]["orientation"]["rotation_deg"],
            (10.0, 20.0, 30.0),
        )

    def test_negative_radius_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                initial_pose={
                    "radius": {"minimum": -1.0, "maximum": 20.0},
                }
            )

    def test_zero_radial_direction_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                initial_pose={
                    "radius": {"minimum": 20.0, "maximum": 20.0},
                    "radial_direction": [0.0, 0.0, 0.0],
                }
            )

    def test_compiles_independent_component_initial_poses(self) -> None:
        plan = compile_sampling_plan(
            design(
                initial_poses={
                    "site_alpha": {
                        "radius": {"minimum": 35.0, "maximum": 35.0},
                        "orientation": {"method": "uniform_so3"},
                        "seed": 101,
                    },
                    "site_beta": {
                        "radius": {"minimum": 55.0, "maximum": 60.0},
                        "axial_offset": {
                            "minimum": -2.0,
                            "maximum": 4.0,
                        },
                        "radial_direction": [0.0, 1.0, 0.0],
                        "seed": 202,
                    },
                }
            )
        )

        self.assertIsNone(plan.initial_pose)
        self.assertEqual(
            tuple(pose.group_id for pose in plan.component_initial_poses),
            ("site_alpha", "site_beta"),
        )
        seed, payload = assembly_initialization_payload(plan)
        self.assertIsNone(seed)
        self.assertEqual(payload["site_alpha"]["random_seed"], 101)
        self.assertEqual(payload["site_beta"]["random_seed"], 202)
        self.assertEqual(
            payload["site_beta"]["placement"]["radius"],
            {"mean": 57.5, "range": 2.5},
        )

    def test_rejects_global_and_component_initial_poses_together(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                initial_pose={
                    "radius": {"minimum": 20.0, "maximum": 20.0},
                },
                initial_poses={
                    "site_alpha": {
                        "radius": {
                            "minimum": 30.0,
                            "maximum": 30.0,
                        },
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
