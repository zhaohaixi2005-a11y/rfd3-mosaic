from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rfd3_mosaic.experiment_worker import _merged_rfd3_input
from rfd3_mosaic.sampling_plan import DesignSamplingAssignment


class ExperimentWorkerMultiInputTestCase(unittest.TestCase):
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
