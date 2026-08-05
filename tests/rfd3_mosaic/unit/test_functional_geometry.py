from pathlib import Path
import tempfile
import unittest

import yaml
from pydantic import ValidationError

from rfd3_mosaic.schema.functional_geometry import (
    FunctionalGeometrySpec,
    load_functional_geometry,
)


def _cooperative_site_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "three-subunit-zinc-site",
        "input": "site.pdb",
        "fragments": [
            {"id": "metal", "selector": "chain Z and resi 1"},
            {"id": "site_a", "selector": "chain A and resi 10"},
            {"id": "site_b", "selector": "chain B and resi 10"},
            {"id": "site_c", "selector": "chain C and resi 10"},
        ],
        "atoms": [
            {
                "id": "zn",
                "fragment": "metal",
                "selector": "chain Z and resi 1 and name ZN",
                "element": "Zn",
            },
            {
                "id": "his_a",
                "fragment": "site_a",
                "selector": "chain A and resi 10 and name NE2",
                "element": "N",
            },
            {
                "id": "his_b",
                "fragment": "site_b",
                "selector": "chain B and resi 10 and name NE2",
                "element": "N",
            },
            {
                "id": "asp_c",
                "fragment": "site_c",
                "selector": "chain C and resi 10 and name OD1",
                "element": "O",
            },
        ],
        "relations": [
            {
                "kind": "coordination",
                "id": "zinc_coordination",
                "center": "zn",
                "ligands": ["his_a", "his_b", "asp_c"],
                "shape": "trigonal_planar",
                "distance_target": 2.1,
            },
            {
                "kind": "angle",
                "id": "site_angle",
                "atoms": ["his_a", "zn", "his_b"],
                "target_deg": 120.0,
                "tolerance_deg": 10.0,
            },
        ],
    }


class FunctionalGeometrySchemaTestCase(unittest.TestCase):
    def test_accepts_a_cross_fragment_coordination_hyperedge(self) -> None:
        specification = FunctionalGeometrySpec.model_validate(
            _cooperative_site_payload()
        )

        coordination = specification.relations[0]
        self.assertEqual(coordination.kind, "coordination")
        self.assertEqual(len(coordination.ligands), 3)

    def test_rejects_unknown_atom_reference(self) -> None:
        payload = _cooperative_site_payload()
        payload["relations"][1]["atoms"][2] = "missing"

        with self.assertRaisesRegex(ValidationError, "unknown atoms"):
            FunctionalGeometrySpec.model_validate(payload)

    def test_coordination_must_span_fragments_by_default(self) -> None:
        payload = _cooperative_site_payload()
        for atom in payload["atoms"]:
            atom["fragment"] = "site_a"

        with self.assertRaisesRegex(ValidationError, "at least two"):
            FunctionalGeometrySpec.model_validate(payload)

    def test_shape_enforces_ligand_count(self) -> None:
        payload = _cooperative_site_payload()
        payload["relations"][0]["shape"] = "tetrahedral"

        with self.assertRaisesRegex(ValidationError, "exactly 4"):
            FunctionalGeometrySpec.model_validate(payload)

    def test_relative_pose_validates_se3(self) -> None:
        payload = _cooperative_site_payload()
        payload["relations"] = [
            {
                "kind": "relative_pose",
                "id": "pose",
                "fragments": ["site_a", "site_b"],
                "target_transform": [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ]

        with self.assertRaisesRegex(ValidationError, "not orthogonal"):
            FunctionalGeometrySpec.model_validate(payload)

    def test_load_resolves_the_structure_relative_to_specification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "site.pdb").write_text("END\n", encoding="utf-8")
            path = root / "functional.yaml"
            path.write_text(
                yaml.safe_dump(_cooperative_site_payload(), sort_keys=False),
                encoding="utf-8",
            )

            specification = load_functional_geometry(path)

        self.assertEqual(specification.input, root / "site.pdb")


if __name__ == "__main__":
    unittest.main()
