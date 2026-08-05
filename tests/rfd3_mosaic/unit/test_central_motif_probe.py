import json
from pathlib import Path
import tempfile
import unittest


def _c3_registry() -> tuple[list[str], dict[str, list[list[float]]]]:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    r1 = [
        [-0.5, -0.8660254037844386, 0.0, 0.0],
        [0.8660254037844386, -0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    r2 = [
        [-0.5, 0.8660254037844386, 0.0, 0.0],
        [-0.8660254037844386, -0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    order = ["C3:e", "C3:r1", "C3:r2"]
    return order, {"C3:e": identity, "C3:r1": r1, "C3:r2": r2}

from rfd3_mosaic.rfd3_central_motif_probe import (
    build_central_motif_probe_input,
)


class CentralMotifProbeTestCase(unittest.TestCase):
    def test_builds_bidirectional_c3_fixed_motif_orbit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text("data_probe\n", encoding="utf-8")
            template = root / "template.json"
            order, matrices = _c3_registry()
            template.write_text(
                json.dumps(
                    {
                        "old": {
                            "dialect": 2,
                            "input": source.name,
                            "contig": "B1-31,70-70,C1-30",
                            "select_fixed_atoms": {
                                "B1-31": "ALL",
                                "C1-30": "ALL",
                            },
                            "symmetry": {
                                "id": "C3",
                                "is_symmetric_motif": True,
                            },
                            "extra": {
                                "registry_preflight": "passed",
                                "registry_transform_order": order,
                                "registry_transform_matrices": matrices,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_path = build_central_motif_probe_input(
                template,
                root / "output",
                fixed_selector="B1-31",
                n_terminal_length=20,
                c_terminal_length=25,
            )
            example = json.loads(output_path.read_text())["central_motif_c3_probe"]

            self.assertEqual(example["contig"], "20-20,B1-31,25-25")
            self.assertEqual(example["select_fixed_atoms"], {"B1-31": "ALL"})
            self.assertTrue(example["symmetry"]["use_declared_frames"])
            self.assertEqual(
                example["symmetry"]["declared_transform_order"],
                order,
            )
            self.assertEqual(
                example["symmetry"]["declared_transform_matrices"],
                matrices,
            )
            self.assertTrue((output_path.parent / "source.cif").is_file())
            groups = example["extra"]["motif_constraint_groups"]
            self.assertEqual(len(groups), 3)
            self.assertEqual(
                {group["constraint_kind"] for group in groups},
                {"fixed_motif"},
            )
            self.assertEqual(
                [
                    group["members"][0]["sym_transform_id"]
                    for group in groups
                ],
                [0, 1, 2],
            )
            orbit = example["extra"]["motif_constraint_orbits"][0]
            self.assertEqual(orbit["mobility_mode"], "fixed")
            self.assertEqual(orbit["group_transform_ids"], [0, 1, 2])

    def test_can_explicitly_recover_frames_for_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text("data_probe\n", encoding="utf-8")
            template = root / "template.json"
            template.write_text(
                json.dumps(
                    {
                        "old": {
                            "input": source.name,
                            "symmetry": {"id": "C3"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_path = build_central_motif_probe_input(
                template,
                root / "output",
                fixed_selector="B1-31",
                use_declared_frames=False,
            )
            example = json.loads(output_path.read_text())[
                "central_motif_c3_probe"
            ]

            self.assertFalse(
                example["symmetry"]["use_declared_frames"]
            )

    def test_rejects_invalid_lengths_and_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text("data_probe\n", encoding="utf-8")
            template = root / "template.json"
            template.write_text(
                json.dumps(
                    {
                        "old": {
                            "input": source.name,
                            "symmetry": {"id": "C3"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "contiguous range"):
                build_central_motif_probe_input(
                    template,
                    root / "bad-selector",
                    fixed_selector="B1,B3",
                )
            with self.assertRaisesRegex(ValueError, "positive"):
                build_central_motif_probe_input(
                    template,
                    root / "bad-length",
                    fixed_selector="B1-31",
                    n_terminal_length=0,
                )


if __name__ == "__main__":
    unittest.main()
