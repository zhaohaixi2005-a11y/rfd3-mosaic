import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from rfd3_mosaic.schema import SimpleCageIntentSpec
from rfd3_mosaic.seed_library import materialize_seed_library
from rfd3_mosaic.simple_resolver import (
    enumerate_simple_design_candidates,
    resolve_simple_intent,
)
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

    def assert_nested_close(
        self,
        first: object,
        second: object,
        *,
        tolerance: float = 2.0e-3,
    ) -> None:
        if isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_close(
                    first[key], second[key], tolerance=tolerance
                )
            return
        if isinstance(first, list) and isinstance(second, list):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second, strict=True):
                self.assert_nested_close(
                    left, right, tolerance=tolerance
                )
            return
        if (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(second, (int, float))
            and not isinstance(second, bool)
        ):
            self.assertAlmostEqual(
                float(first),
                float(second),
                delta=tolerance,
            )
            return
        self.assertEqual(first, second)

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

    def test_strict_replay_is_invariant_to_seed_file_world_frame(self) -> None:
        first = self.root / "alpha.pdb"
        second = self.root / "beta.pdb"
        transformed = self.root / "beta_transformed.pdb"
        _write_seed(first)
        _write_seed(second)
        angle = math.radians(41.0)
        rotation = np.asarray(
            (
                (math.cos(angle), 0.0, math.sin(angle)),
                (0.0, 1.0, 0.0),
                (-math.sin(angle), 0.0, math.cos(angle)),
            ),
            dtype=np.float64,
        )
        _write_seed(
            transformed,
            rotation=rotation,
            translation=np.asarray((-103.0, 59.0, 27.0)),
        )

        common = {
            "symmetry_ids": ("C3",),
            "pose_samples": 1,
            "seed_start": 73,
            "timesteps": 10,
            "top_count": 1,
            "max_candidates": 64,
            "pose_optimize_top": 1,
            "pose_optimization_levels": 1,
        }
        original = resolve_simple_intent(
            _intent(first, second),
            self.root / "original-resolution",
            **common,
        )
        moved = resolve_simple_intent(
            _intent(first, transformed),
            self.root / "moved-resolution",
            **common,
        )

        self.assertEqual(original["accepted_count"], moved["accepted_count"])
        self.assertEqual(original["selected_count"], moved["selected_count"])
        self.assertGreater(original["selected_count"], 0, original["ranking"])
        first_rank = original["ranking"][0]
        moved_rank = moved["ranking"][0]
        for key in (
            "topology_id",
            "connection_order",
            "connection_orbit_offset",
            "hard_clashes",
            "failed_required_interfaces",
            "infeasible_links",
            "replay_validated",
        ):
            self.assertEqual(first_rank[key], moved_rank[key], key)
        first_design = yaml.safe_load(
            Path(first_rank["resolved_design"]).read_text(
                encoding="utf-8"
            )
        )
        moved_design = yaml.safe_load(
            Path(moved_rank["resolved_design"]).read_text(
                encoding="utf-8"
            )
        )
        # The canonicalized input paths live in different resolution
        # directories.  Every executable declaration downstream of that
        # path must nevertheless be identical.
        first_design.pop("input")
        moved_design.pop("input")
        # A rigid transform written through the 0.001 A PDB coordinate
        # format cannot be recovered byte-for-byte.  The executable pose is
        # nevertheless invariant within two input coordinate quanta.
        self.assert_nested_close(first_design, moved_design)

    def test_independent_seed_participant_preserves_multiple_helices(
        self,
    ) -> None:
        first = self.root / "alpha_multi_helix.pdb"
        second = self.root / "beta_multi_helix.pdb"
        _write_seed(first)
        _write_seed(second)
        payload = _intent(first, second).model_dump(mode="json")
        for seed in payload["interface_seeds"].values():
            seed["selectors"] = {
                "A": "A/1-2/*,A/3-4/*",
                "B": "B/1-2/*,B/3-4/*",
            }
        intent = SimpleCageIntentSpec.model_validate(payload)

        materialized = materialize_seed_library(
            intent,
            self.root / "multi-helix-library",
        )

        self.assertTrue(materialized.independent_frames)
        for seed in materialized.intent.interface_seeds.values():
            self.assertTrue(
                all(
                    len(selector.split(",")) == 2
                    for selector in seed.selectors.values()
                )
            )
        self.assertEqual(
            len(read_structure_atoms(
                materialized.structure_path,
                mmcif_identifier_namespace="label",
            )),
            64,
        )

    def test_shared_file_can_explicitly_request_unknown_relative_pose(
        self,
    ) -> None:
        shared = self.root / "shared.pdb"
        _write_seed(shared)
        payload = _intent(shared, shared).model_dump(mode="json")
        payload["seed_layout"] = "solve"
        intent = SimpleCageIntentSpec.model_validate(payload)

        materialized = materialize_seed_library(
            intent,
            self.root / "shared-solve",
        )

        self.assertTrue(materialized.independent_frames)
        self.assertEqual(
            materialized.manifest["relative_seed_pose"],
            "solve",
        )
        self.assertNotEqual(materialized.structure_path, shared)
        self.assertEqual(
            len({
                participant
                for seed in materialized.intent.interface_seeds.values()
                for participant in seed.participants
            }),
            4,
        )

    def test_shared_file_auto_preserves_relative_pose(self) -> None:
        shared = self.root / "shared.pdb"
        _write_seed(shared)

        materialized = materialize_seed_library(
            _intent(shared, shared),
            self.root / "shared-auto",
        )

        self.assertFalse(materialized.independent_frames)
        self.assertEqual(
            materialized.manifest["relative_seed_pose"],
            "preserved",
        )

    def test_preserve_input_rejects_separate_coordinate_frames(self) -> None:
        first = self.root / "alpha.pdb"
        second = self.root / "beta.pdb"
        _write_seed(first)
        _write_seed(second)
        payload = _intent(first, second).model_dump(mode="json")
        payload["seed_layout"] = "preserve_input"
        intent = SimpleCageIntentSpec.model_validate(payload)

        with self.assertRaisesRegex(ValueError, "one shared input"):
            materialize_seed_library(
                intent,
                self.root / "invalid-preserve",
            )

    def test_solve_layout_requires_a_relative_multi_seed_problem(self) -> None:
        first = self.root / "alpha.pdb"
        _write_seed(first)
        payload = _intent(first, first).model_dump(mode="json")
        payload["seed_layout"] = "solve"
        payload["interface_seeds"] = {
            "alpha": payload["interface_seeds"]["alpha"]
        }

        with self.assertRaisesRegex(ValueError, "at least two"):
            SimpleCageIntentSpec.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
