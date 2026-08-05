import unittest

from pydantic import ValidationError

from rfd3_mosaic.sampling_plan import (
    assembly_initialization_payload,
    compile_sampling_plan,
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
        self.assertEqual(assembly_initialization_payload(plan), (None, {}))

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


if __name__ == "__main__":
    unittest.main()
