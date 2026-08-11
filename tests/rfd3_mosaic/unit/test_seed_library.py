import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.schema import SimpleCageIntentSpec
from rfd3_mosaic.seed_library import materialize_seed_library
from rfd3_mosaic.simple_resolver import enumerate_simple_design_candidates
from rfd3_mosaic.structure import read_structure_atoms


def _atom_line(
    serial: int,
    atom_name: str,
    chain: str,
    residue: int,
    coordinate: tuple[float, float, float],
) -> str:
    x, y, z = coordinate
    element = atom_name[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2s}\n"
    )


def _write_seed(
    path: Path,
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> None:
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    lines = []
    serial = 1
    for chain, y in (("A", 0.0), ("B", 3.0)):
        for residue in range(1, 5):
            for atom_name, offset in (
                ("N", (-0.5, 0.0, -0.2)),
                ("CA", (0.0, 0.0, 0.0)),
                ("C", (0.5, 0.0, 0.3)),
                ("CB", (0.0, 0.8, 1.0)),
            ):
                base = np.asarray(
                    (2.0 * (residue - 1), y, 0.0),
                    dtype=np.float64,
                ) + np.asarray(offset)
                coordinate = rotation @ base + translation
                lines.append(
                    _atom_line(
                        serial,
                        atom_name,
                        chain,
                        residue,
                        tuple(float(value) for value in coordinate),
                    )
                )
                serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _intent(first: Path, second: Path) -> SimpleCageIntentSpec:
    return SimpleCageIntentSpec.model_validate(
        {
            "name": "independent-interface-library",
            "goal": {
                "architecture": "ring",
                "composition": "auto",
                "symmetry": ["C3"],
            },
            "interface_seeds": {
                "alpha": {
                    "source": first,
                    "participants": ["A", "B"],
                    "selectors": {
                        "A": "A/1-4/*",
                        "B": "B/1-4/*",
                    },
                    "use": {"exact": 3},
                },
                "beta": {
                    "source": second,
                    "participants": ["A", "B"],
                    "selectors": {
                        "A": "A/1-4/*",
                        "B": "B/1-4/*",
                    },
                    "use": {"exact": 3},
                },
            },
            "generation": {
                "length": {"minimum": 20, "maximum": 60}
            },
        }
    )


class SeedLibraryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_independent_seed_frame_is_rigid_transform_invariant(self) -> None:
        first = self.root / "alpha.pdb"
        second = self.root / "beta.pdb"
        transformed = self.root / "beta_transformed.pdb"
        _write_seed(first)
        _write_seed(second)
        angle = math.radians(67.0)
        rotation = np.asarray(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        _write_seed(
            transformed,
            rotation=rotation,
            translation=np.asarray((81.0, -37.0, 19.0)),
        )

        original = materialize_seed_library(
            _intent(first, second),
            self.root / "original",
        )
        moved = materialize_seed_library(
            _intent(first, transformed),
            self.root / "moved",
        )

        self.assertTrue(original.independent_frames)
        self.assertTrue(moved.independent_frames)
        original_atoms = read_structure_atoms(
            original.structure_path,
            mmcif_identifier_namespace="label",
        )
        moved_atoms = read_structure_atoms(
            moved.structure_path,
            mmcif_identifier_namespace="label",
        )
        self.assertEqual(
            [
                (atom.chain_id, atom.residue_number, atom.atom_name)
                for atom in original_atoms
            ],
            [
                (atom.chain_id, atom.residue_number, atom.atom_name)
                for atom in moved_atoms
            ],
        )
        np.testing.assert_allclose(
            [atom.coordinate for atom in original_atoms],
            [atom.coordinate for atom in moved_atoms],
            atol=2.0e-3,
        )
        self.assertEqual(
            {
                seed.use.description
                for seed in original.intent.interface_seeds.values()
            },
            {"exactly 3"},
        )

    def test_global_candidates_have_explicit_per_seed_poses(self) -> None:
        first = self.root / "alpha.pdb"
        second = self.root / "beta.pdb"
        _write_seed(first)
        _write_seed(
            second,
            translation=np.asarray((500.0, -200.0, 100.0)),
        )
        materialized = materialize_seed_library(
            _intent(first, second),
            self.root / "library",
        )
        candidates = enumerate_simple_design_candidates(
            materialized.intent,
            symmetry_ids=("C3",),
            global_placement=True,
            pose_samples=2,
        )

        self.assertEqual(len(candidates), 32)
        self.assertTrue(
            all(candidate.global_pose_initialization for candidate in candidates)
        )
        self.assertEqual(
            {candidate.pose_sample_index for candidate in candidates},
            {0, 1},
        )
        for candidate in candidates:
            self.assertEqual(
                set(candidate.design.sampling.initial_poses),
                set(candidate.design.components),
            )
            self.assertTrue(
                all(
                    pose.orientation.method == "fixed"
                    and pose.radius.minimum == pose.radius.maximum
                    for pose in candidate.design.sampling.initial_poses.values()
                )
            )


if __name__ == "__main__":
    unittest.main()
