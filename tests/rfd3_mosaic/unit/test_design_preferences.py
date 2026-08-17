import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from rfd3_mosaic.constraint_plan import compile_constraint_plan
from rfd3_mosaic.design_preferences import compile_design_preferences
from rfd3_mosaic.experiment_worker import (
    _graph_interface_guidance_overrides,
)
from rfd3_mosaic.schema import UserDesignSpec


def create_interface_design(**updates: object) -> UserDesignSpec:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": "preference-test",
        "input": "/tmp/motif.pdb",
        "symmetry": "C3",
        "task": "create_symmetric_interface",
        "generation": [
            {
                "kind": "terminal",
                "anchor": "A1-2",
                "terminus": "n",
                "length": 20,
            }
        ],
        "constraints": [{"kind": "fixed_xyz", "selector": "A1-2"}],
    }
    payload.update(updates)
    return UserDesignSpec.model_validate(payload)


class DesignPreferencesTestCase(unittest.TestCase):
    def test_defaults_keep_arrangement_locked_and_safety_weights(self) -> None:
        design = create_interface_design()
        resolved = compile_design_preferences(design)

        self.assertEqual(design.fixed_arrangement.value, "locked")
        self.assertEqual(resolved.component_motion.value, "locked")
        self.assertIsNone(resolved.mobility_subspace)
        self.assertEqual(
            resolved.sampler_overrides[
                "graph_interface_guidance_clash_weight"
            ],
            8.0,
        )
        self.assertIn("exact_fixed_geometry", resolved.hard_contracts)

    def test_guided_motion_infers_arrangement_and_axis_subspace(self) -> None:
        design = create_interface_design(
            preferences={"component_motion": "guided"}
        )
        pose = compile_constraint_plan(design).operators[0].parameters["pose"]

        self.assertEqual(
            design.fixed_arrangement.value,
            "optimize_components",
        )
        self.assertEqual(pose["subspace"], "radial_axial_rotation")

    def test_free_motion_uses_bounded_se3(self) -> None:
        design = create_interface_design(
            preferences={"component_motion": "free"}
        )
        pose = compile_constraint_plan(design).operators[0].parameters["pose"]

        self.assertEqual(pose["subspace"], "bounded_se3")

    def test_tight_large_interface_changes_only_soft_preset(self) -> None:
        design = create_interface_design(
            preferences={
                "packing": "tight",
                "interface_area": "large",
            }
        )
        resolved = compile_design_preferences(design)
        overrides = resolved.sampler_overrides

        self.assertEqual(
            overrides["graph_interface_guidance_pairs_per_edge"],
            12,
        )
        self.assertGreater(
            overrides["graph_interface_guidance_coverage_weight"],
            1.25,
        )
        self.assertEqual(
            overrides["graph_interface_guidance_clash_weight"],
            8.0,
        )

    def test_cavity_and_diversity_resolve_to_search_controls(self) -> None:
        compact = compile_design_preferences(
            create_interface_design(
                preferences={"cavity": "compact", "diversity": "low"}
            )
        )
        open_high = compile_design_preferences(
            create_interface_design(
                preferences={"cavity": "open", "diversity": "high"}
            )
        )

        self.assertLess(
            compact.initial_radius_scale,
            open_high.initial_radius_scale,
        )
        self.assertEqual(compact.diversity_plan.global_pose_samples, 4)
        self.assertEqual(open_high.diversity_plan.global_pose_samples, 16)

    def test_explicit_conflicting_motion_controls_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "component_motion conflicts with fixed_arrangement",
        ):
            create_interface_design(
                fixed_arrangement="locked",
                preferences={"component_motion": "guided"},
            )

    def test_raw_weights_are_expert_only_and_override_the_preset(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "raw guidance weights require expert",
        ):
            create_interface_design(guidance={"shape_weight": 0.9})

        expert = UserDesignSpec.model_validate(
            {
                "schema_version": 1,
                "name": "expert-preference-test",
                "input": "/tmp/motif.pdb",
                "symmetry": "C3",
                "components": {
                    "seed": {
                        "selectors": ["A1", "A2"],
                        "geometry": "joint_rigid",
                    }
                },
                "ports": {
                    "left": {"component": "seed", "selectors": ["A1"]},
                    "right": {"component": "seed", "selectors": ["A2"]},
                },
                "interfaces": [
                    {
                        "id": "designed",
                        "between": ["left", "right"],
                        "copy_relation": {"orbit_offset": 1},
                        "relation": {"mode": "contact"},
                    }
                ],
                "connections": [
                    {
                        "id": "polymer",
                        "from": {
                            "component": "seed",
                            "selector": "A1",
                            "terminus": "c",
                        },
                        "to": {
                            "component": "seed",
                            "selector": "A2",
                            "terminus": "n",
                        },
                        "length": 20,
                        "copy_relation": {"orbit_offset": 1},
                    }
                ],
                "guidance": {
                    "shape_weight": 0.9,
                    "clash_weight": 12.0,
                },
            }
        )
        overrides = compile_design_preferences(expert).sampler_overrides
        self.assertEqual(
            overrides["graph_interface_guidance_shape_weight"],
            0.9,
        )
        self.assertEqual(
            overrides["graph_interface_guidance_clash_weight"],
            12.0,
        )

    def test_worker_reads_only_frozen_sampler_overrides(self) -> None:
        resolved = compile_design_preferences(
            create_interface_design(
                preferences={
                    "packing": "tight",
                    "interface_area": "large",
                }
            )
        )
        payload = {
            "example": {
                "extra": {
                    "resolved_design_preferences": resolved.model_dump(
                        mode="json"
                    )
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rfd3_input.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            overrides = _graph_interface_guidance_overrides(path)

        self.assertIn(
            "++inference_sampler.graph_interface_guidance_pairs_per_edge=12",
            overrides,
        )
        self.assertIn(
            "++inference_sampler.graph_interface_guidance_shape_weight=0.8",
            overrides,
        )


if __name__ == "__main__":
    unittest.main()
