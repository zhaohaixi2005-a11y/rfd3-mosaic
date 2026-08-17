import json
import tempfile
import unittest
from pathlib import Path

import yaml

from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.output import compile_rfd3_input
from rfd3_mosaic.rfd3_prevalidate import prevalidate_rfd3_input
from rfd3_mosaic.schema import UserDesignSpec


def _write_motif(path: Path) -> None:
    lines = []
    serial = 1
    for residue in range(1, 4):
        for atom_index, atom in enumerate(("N", "CA", "C", "O")):
            x = 14.0 + residue * 3.8 + atom_index * 0.3
            y = 1.5 * residue
            z = 0.5 * atom_index
            lines.append(
                f"ATOM  {serial:5d} {atom:>4s} ALA A{residue:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
                f"          {atom[0]:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


class PublicCylindricalRuntimeTestCase(unittest.TestCase):
    def test_public_cylindrical_compiles_and_prevalidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            motif = root / "motif.pdb"
            _write_motif(motif)
            design = UserDesignSpec.model_validate(
                {
                    "name": "public-c3-cylindrical",
                    "input": str(motif),
                    "symmetry": "C3",
                    "generation": [
                        {
                            "kind": "terminal",
                            "anchor": "A1-3",
                            "terminus": "c",
                            "length": 8,
                        }
                    ],
                    "constraints": [
                        {
                            "kind": "cylindrical",
                            "selector": "A1-3",
                            "atoms": "ca",
                            "keep": ["radius", "axial"],
                        }
                    ],
                }
            )
            lowered = lower_user_design(design)
            config = root / "assembly.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "assembly": lowered.specification.model_dump(
                            mode="json"
                        )
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs = compile_rfd3_input(
                config,
                root / "compiled",
                example_id="public-c3-cylindrical",
                extra_metadata={
                    **lowered.runtime_constraint_metadata,
                    "constraint_plan": lowered.constraint_plan.model_dump(
                        mode="json"
                    ),
                },
            )
            emitted = json.loads(outputs.input_path.read_text())["public-c3-cylindrical"]

            self.assertEqual(emitted["select_fixed_atoms"], {"A1-3": ""})
            self.assertEqual(emitted["extra"]["motif_constraint_groups"], [])
            runtime = emitted["extra"]["cylindrical_constraints"]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(runtime[0]["keep"], ["radius", "axial"])
            self.assertEqual(
                runtime[0]["atom_keys"],
                [["A1", "CA"], ["A2", "CA"], ["A3", "CA"]],
            )

            report = prevalidate_rfd3_input(outputs.input_path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["fixed_coordinate_atom_count"], 0)
            self.assertEqual(
                report["declared_cylindrical_constraint_count"], 1
            )
            self.assertEqual(
                report["resolved_cylindrical_constraint_count"], 1
            )
            self.assertEqual(
                report["cylindrical_constrained_atom_count"], 9
            )
            self.assertEqual(
                report["cylindrical_constrained_dof_count"], 18
            )


if __name__ == "__main__":
    unittest.main()
