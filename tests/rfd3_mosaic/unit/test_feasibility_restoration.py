import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.feasibility_restoration import (
    bind_feasible_linker_lengths,
)
from rfd3_mosaic.schema import UserDesignSpec


class CandidateFeasibilityRestorationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "seed.pdb"
        self.structure.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00 20.00           C\n"
            "ATOM      2  CA  ALA B   1      10.000   0.000   0.000"
            "  1.00 20.00           C\nEND\n",
            encoding="utf-8",
        )
        self.design = UserDesignSpec.model_validate(
            {
                "name": "restoration-test",
                "input": self.structure,
                "symmetry": "C3",
                "components": {
                    "alpha": {
                        "selectors": ["A1"],
                        "geometry": "rigid",
                    },
                    "beta": {
                        "selectors": ["B1"],
                        "geometry": "rigid",
                    },
                },
                "connections": [
                    {
                        "id": "polymer_link",
                        "from": {
                            "component": "alpha",
                            "terminus": "c",
                            "selector": "A1",
                        },
                        "to": {
                            "component": "beta",
                            "terminus": "n",
                            "selector": "B1",
                        },
                        "length": {"minimum": 10, "maximum": 45},
                    }
                ],
            }
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _manifest(*requirements: int) -> dict:
        return {
            "validation": {
                "scaffold_link_geometry": {
                    "links": [
                        {
                            "source_link_id": "polymer_link",
                            "link_instance_id": (
                                f"polymer_link@motif_orbit[{index}]"
                            ),
                            "chain_break": False,
                            "minimum_required_residues_at_3_8A": required,
                        }
                        for index, required in enumerate(requirements)
                    ]
                }
            }
        }

    def test_binds_range_to_worst_physical_instance(self) -> None:
        result = bind_feasible_linker_lengths(
            self.design,
            self._manifest(20, 31, 28),
        )

        restored = result.design.connections[0]
        self.assertEqual(restored.length.minimum, 31)
        self.assertEqual(restored.length.maximum, 31)
        self.assertEqual(result.linker_bindings[0].required_minimum, 31)
        self.assertEqual(
            result.linker_bindings[0].policy,
            "configured_range_contour_sufficient",
        )
        self.assertTrue(result.changed)
        self.assertEqual(
            restored.from_endpoint,
            self.design.connections[0].from_endpoint,
        )
        self.assertEqual(
            restored.to_endpoint,
            self.design.connections[0].to_endpoint,
        )
        self.assertEqual(result.design.components, self.design.components)
        self.assertEqual(result.design.symmetry, self.design.symmetry)

    def test_resolves_canonical_lowered_connection_source_id(self) -> None:
        manifest = self._manifest(20, 31, 28)
        for report in manifest["validation"][
            "scaffold_link_geometry"
        ]["links"]:
            report["source_link_id"] = "connection__polymer_link"

        result = bind_feasible_linker_lengths(self.design, manifest)

        self.assertEqual(
            result.design.connections[0].length.minimum,
            31,
        )

    def test_rejects_ambiguous_public_and_lowered_source_ids(self) -> None:
        manifest = self._manifest(20)
        duplicate = dict(
            manifest["validation"]["scaffold_link_geometry"]["links"][0]
        )
        duplicate["source_link_id"] = "connection__polymer_link"
        duplicate["link_instance_id"] = "lowered@motif_orbit[0]"
        manifest["validation"]["scaffold_link_geometry"]["links"].append(
            duplicate
        )

        with self.assertRaisesRegex(ValueError, "ambiguously contains"):
            bind_feasible_linker_lengths(self.design, manifest)

    def test_keeps_midpoint_when_it_already_covers_every_instance(self) -> None:
        result = bind_feasible_linker_lengths(
            self.design,
            self._manifest(20, 25, 27),
        )

        restored = result.design.connections[0]
        self.assertEqual(restored.length.minimum, 27)
        self.assertEqual(restored.length.maximum, 27)
        self.assertEqual(
            result.linker_bindings[0].policy,
            "configured_range_midpoint",
        )

    def test_preserves_legacy_exact_integer_length(self) -> None:
        exact_connection = self.design.connections[0].model_copy(
            update={"length": 31}
        )
        design = self.design.model_copy(
            update={"connections": (exact_connection,)}
        )

        result = bind_feasible_linker_lengths(
            design,
            self._manifest(20, 31, 28),
        )

        self.assertEqual(result.design.connections[0].length, 31)
        self.assertEqual(result.linker_bindings[0].policy, "user_exact")
        self.assertFalse(result.changed)

    def test_rejects_only_when_user_maximum_is_physically_insufficient(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "user-authorized range.*required=46",
        ):
            bind_feasible_linker_lengths(
                self.design,
                self._manifest(20, 46, 28),
            )

    def test_no_connections_preserves_legacy_manifest_compatibility(
        self,
    ) -> None:
        design = self.design.model_copy(update={"connections": ()})

        result = bind_feasible_linker_lengths(design, {})

        self.assertIs(result.design, design)
        self.assertEqual(result.linker_bindings, ())
        self.assertFalse(result.changed)

    def test_tie_group_uses_one_length_across_all_connections(self) -> None:
        first = self.design.connections[0].model_copy(
            update={"tie_group": "unit_length"}
        )
        second = first.model_copy(
            update={
                "id": "polymer_link_2",
                "length": first.length.model_copy(
                    update={"minimum": 20, "maximum": 35}
                ),
            }
        )
        design = self.design.model_copy(
            update={"connections": (first, second)}
        )
        manifest = self._manifest(20, 31, 28)
        manifest["validation"]["scaffold_link_geometry"]["links"].extend(
            {
                "source_link_id": "polymer_link_2",
                "link_instance_id": f"polymer_link_2@motif_orbit[{index}]",
                "chain_break": False,
                "minimum_required_residues_at_3_8A": required,
            }
            for index, required in enumerate((25, 34, 30))
        )

        result = bind_feasible_linker_lengths(design, manifest)

        self.assertEqual(
            {
                connection.length.minimum
                for connection in result.design.connections
            },
            {34},
        )
        self.assertEqual(
            {binding.tie_group for binding in result.linker_bindings},
            {"unit_length"},
        )
        self.assertEqual(
            {binding.policy for binding in result.linker_bindings},
            {"tie_group_contour_sufficient"},
        )

    def test_tie_group_rejects_disjoint_user_ranges(self) -> None:
        first = self.design.connections[0].model_copy(
            update={
                "tie_group": "unit_length",
                "length": self.design.connections[0].length.model_copy(
                    update={"minimum": 10, "maximum": 15}
                ),
            }
        )
        second = first.model_copy(
            update={
                "id": "polymer_link_2",
                "length": first.length.model_copy(
                    update={"minimum": 20, "maximum": 25}
                ),
            }
        )
        design = self.design.model_copy(
            update={"connections": (first, second)}
        )
        manifest = self._manifest(10, 10, 10)
        manifest["validation"]["scaffold_link_geometry"]["links"].extend(
            {
                "source_link_id": "polymer_link_2",
                "link_instance_id": f"polymer_link_2@motif_orbit[{index}]",
                "chain_break": False,
                "minimum_required_residues_at_3_8A": 10,
            }
            for index in range(3)
        )

        with self.assertRaisesRegex(ValueError, "no common.*length"):
            bind_feasible_linker_lengths(design, manifest)


if __name__ == "__main__":
    unittest.main()
