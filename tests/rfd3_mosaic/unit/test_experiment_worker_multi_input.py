from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rfd3_mosaic.experiment_worker import (
    _generated_cross_chain_topology_runtime,
    _merged_rfd3_input,
    _require_compiled_pose_feasibility,
)
from rfd3_mosaic.sampling_plan import DesignSamplingAssignment


class ExperimentWorkerMultiInputTestCase(unittest.TestCase):
    def _write_capture_input(
        self,
        root: Path,
        *,
        enabled: bool,
        cross_chain_topology: bool = False,
    ) -> Path:
        path = root / "rfd3_input.json"
        path.write_text(
            json.dumps(
                {
                    "design": {
                        "extra": {
                            "resolved_design_preferences": {
                                "schema_version": 1,
                                "preset_version": "packing_preferences_v2",
                                "packing": "balanced",
                                "cavity": "auto",
                                "diversity": "medium",
                                "interface_area": "auto",
                                "component_motion": "free",
                                "mobility_subspace": "bounded_se3",
                                "initial_radius_scale": 1.0,
                                "diversity_plan": {
                                    "global_pose_samples": 8,
                                    "scope": "independent_seed_pose_search",
                                },
                                "sampler_overrides": {
                                    "enable_assembly_robust_capture": enabled
                                },
                            },
                            "generated_cross_chain_topology_guidance": {
                                "enabled": cross_chain_topology,
                                "scope": "two_fixed_anchor_generated_runs",
                                "routing_ownership_weight": 1.0,
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_required_assembly_pose_gate_accepts_only_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = self._write_capture_input(root, enabled=True)
            manifest = root / "manifest.json"
            report = {
                "evaluated": True,
                "passed": True,
                "failure_reasons": [],
            }
            manifest.write_text(
                json.dumps(
                    {"validation": {"assembly_pose_feasibility": report}}
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _require_compiled_pose_feasibility(compiled, manifest),
                report,
            )

            report["passed"] = False
            report["failure_reasons"] = ["bad cyclic wedge"]
            manifest.write_text(
                json.dumps(
                    {"validation": {"assembly_pose_feasibility": report}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bad cyclic wedge"):
                _require_compiled_pose_feasibility(compiled, manifest)

    def test_pose_gate_is_not_applied_when_capture_semantics_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = self._write_capture_input(root, enabled=False)

            self.assertIsNone(
                _require_compiled_pose_feasibility(
                    compiled,
                    root / "missing-manifest.json",
                )
            )

    def test_locked_cross_chain_route_still_requires_pose_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = self._write_capture_input(
                root,
                enabled=False,
                cross_chain_topology=True,
            )
            manifest = root / "manifest.json"
            report = {
                "evaluated": True,
                "passed": False,
                "failure_reasons": [
                    "endpoint chord intersects another fixed component"
                ],
            }
            manifest.write_text(
                json.dumps(
                    {"validation": {"assembly_pose_feasibility": report}}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "intersects another fixed"):
                _require_compiled_pose_feasibility(compiled, manifest)

    def test_cross_chain_topology_plan_is_read_from_frozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rfd3_input.json"
            path.write_text(
                json.dumps(
                    {
                        "design": {
                            "extra": {
                                "generated_cross_chain_topology_guidance": {
                                    "enabled": True,
                                    "scope": "two_fixed_anchor_generated_runs",
                                    "routing_ownership_weight": 1.0,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            plan = _generated_cross_chain_topology_runtime(path)

            self.assertIsNotNone(plan)
            self.assertEqual(plan["routing_ownership_weight"], 1.0)

    def test_disabled_cross_chain_topology_plan_is_not_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rfd3_input.json"
            path.write_text(
                json.dumps(
                    {
                        "design": {
                            "extra": {
                                "generated_cross_chain_topology_guidance": {
                                    "enabled": False,
                                    "routing_ownership_weight": 1.0,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(_generated_cross_chain_topology_runtime(path))

    def _assignment(self, design_index: int) -> DesignSamplingAssignment:
        return DesignSamplingAssignment(
            design_index=design_index,
            pose_index=0,
            replicate_index=design_index,
            pose_seed=None,
            diffusion_seed=100 + design_index,
        )

    def test_fixed_pose_replicates_keep_one_example_audit_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "pose_00000"
            pose.mkdir()
            source = pose / "rfd3_input.json"
            source.write_text(
                json.dumps(
                    {
                        "pose_00000": {
                            "input": "presymmetrized_input.cif",
                            "extra": {"symmetry_multiplicity": 3},
                        }
                    }
                ),
                encoding="utf-8",
            )
            assignments = (self._assignment(0), self._assignment(1))

            merged, examples = _merged_rfd3_input(
                root / "rfd3_input.json",
                assemblies={0: SimpleNamespace(input_path=source)},
                assignments=assignments,
            )

            self.assertEqual(len(json.loads(source.read_text())), 1)
            self.assertEqual(len(json.loads(merged.read_text())), 2)
            self.assertEqual(len(examples), 2)

    def test_merged_input_refuses_to_overwrite_compiled_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rfd3_input.json"
            source.write_text(
                '{"pose_00000":{"input":"input.cif"}}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "must not overwrite a pose-specific compiled input",
            ):
                _merged_rfd3_input(
                    source,
                    assemblies={0: SimpleNamespace(input_path=source)},
                    assignments=(self._assignment(0), self._assignment(1)),
                )


if __name__ == "__main__":
    unittest.main()
