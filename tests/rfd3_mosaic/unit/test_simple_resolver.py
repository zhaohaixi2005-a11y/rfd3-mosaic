import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from rfd3_mosaic.cli import main
from rfd3_mosaic.compile import expand_symmetry_instances
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.schema import SimpleCageIntentSpec
from rfd3_mosaic.simple_resolver import enumerate_simple_design_candidates


class SimpleIntentResolverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "interface_seed.pdb"
        # Enumeration is deliberately structure-independent, but the emitted
        # public designs must retain a real, replayable input path.
        self.structure.write_text("END\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _intent(
        self,
        *,
        use: object = "auto",
        symmetries: tuple[str, ...] = ("C3",),
    ) -> SimpleCageIntentSpec:
        return SimpleCageIntentSpec.model_validate(
            {
                "name": "ordinary-interface-ring",
                "input": self.structure,
                "goal": {
                    "architecture": "ring",
                    "composition": "homomer",
                    "symmetry": list(symmetries),
                },
                "interface_seeds": {
                    "supplied_interface": {
                        "participants": ["A", "B"],
                        "selectors": {
                            "A": "A/1-2/*",
                            "B": "B/1-2/*",
                        },
                        "use": use,
                        "geometry": "preserve_exact",
                    }
                },
                "generation": {
                    "length": {"minimum": 20, "maximum": 60}
                },
            }
        )

    @staticmethod
    def _candidate_signature(candidate) -> tuple[object, ...]:
        return (
            candidate.candidate_id,
            candidate.symmetry,
            candidate.topology_id,
            tuple(candidate.connection_order),
            tuple(candidate.unresolved_variables),
            candidate.design.model_dump(mode="json"),
        )

    def test_c3_enumerates_four_deterministic_direction_offset_candidates(
        self,
    ) -> None:
        intent = self._intent(symmetries=("C3",))

        first = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9300,
        )
        second = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9300,
        )

        # Two possible polymer directions multiplied by the two distinct C3
        # nearest-neighbour offsets.  The resolver must expose all four rather
        # than silently choosing the first topology it happens to enumerate.
        self.assertEqual(len(first), 4)
        self.assertEqual(
            tuple(map(self._candidate_signature, first)),
            tuple(map(self._candidate_signature, second)),
        )
        self.assertEqual(
            len({candidate.candidate_id for candidate in first}),
            4,
        )
        self.assertEqual(
            len({candidate.topology_id for candidate in first}),
            4,
        )
        self.assertEqual(
            len({tuple(candidate.connection_order) for candidate in first}),
            2,
        )

    def test_c2_deduplicates_inverse_neighbour_offset(self) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(symmetries=("C2",)),
            symmetry_ids=("C2",),
        )

        # In C2, +1 and -1 identify the same non-identity group element.  The
        # two polymer directions remain distinct, but the inverse offset must
        # not be emitted twice.
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            len({candidate.topology_id for candidate in candidates}),
            2,
        )

    def test_candidates_are_standard_expert_designs_on_the_shared_path(
        self,
    ) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(),
            symmetry_ids=("C3",),
        )

        for candidate in candidates:
            design = candidate.design
            self.assertEqual(candidate.symmetry, "C3")
            self.assertEqual(design.symmetry, "C3")
            self.assertEqual(design.input, self.structure)
            self.assertEqual(design.user_mode, "expert")
            self.assertEqual(len(design.components), 1)
            component = next(iter(design.components.values()))
            self.assertEqual(component.geometry, "joint_rigid")
            self.assertEqual(
                set(component.selectors),
                {"A/1-2/*", "B/1-2/*"},
            )
            self.assertEqual(len(design.ports), 2)
            self.assertEqual(len(design.interfaces), 1)
            self.assertEqual(
                design.interfaces[0].relation.mode,
                "preserve_input",
            )
            self.assertEqual(len(design.connections), 1)
            self.assertFalse(design.generation)
            self.assertFalse(design.constraints)

    def test_exact_and_range_usage_filter_incompatible_cyclic_orders(
        self,
    ) -> None:
        exact = enumerate_simple_design_candidates(
            self._intent(
                use={"exact": 3},
                symmetries=("C2", "C3", "C4"),
            ),
        )
        ranged = enumerate_simple_design_candidates(
            self._intent(
                use={"minimum": 2, "maximum": 3},
                symmetries=("C2", "C3", "C4"),
            ),
        )

        self.assertEqual({candidate.symmetry for candidate in exact}, {"C3"})
        self.assertEqual(
            {candidate.symmetry for candidate in ranged},
            {"C2", "C3"},
        )

    def test_exact_partial_c4_orbit_is_lowered_after_c2_seed_validation(
        self,
    ) -> None:
        lines = []
        serial = 1
        base = (
            (1, "N", (1.0, 0.0, 0.0)),
            (1, "CA", (1.3, 0.4, 0.2)),
            (1, "C", (1.6, 0.5, 0.7)),
            (1, "O", (1.8, 0.7, 1.0)),
            (2, "N", (2.0, 0.2, 1.2)),
            (2, "CA", (2.3, 0.5, 1.5)),
            (2, "C", (2.6, 0.7, 2.0)),
            (2, "O", (2.8, 0.9, 2.2)),
        )
        for chain, sign in (("A", 1.0), ("B", -1.0)):
            for residue, atom_name, (x, y, z) in base:
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} ALA "
                    f"{chain}{residue:4d}    "
                    f"{sign*x:8.3f}{sign*y:8.3f}{z:8.3f}"
                    f"{1.0:6.2f}{20.0:6.2f}"
                    f"          {atom_name[0]:>2s}\n"
                )
                serial += 1
        self.structure.write_text(
            "".join(lines) + "END\n",
            encoding="utf-8",
        )
        intent = self._intent(use={"exact": 2}, symmetries=("C4",))

        candidates = enumerate_simple_design_candidates(intent)

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertIsNotNone(candidate.design.finite_orbit_action)
            lowered = lower_user_design(candidate.design)
            instances = expand_symmetry_instances(lowered.specification)
            orbit = instances.constraint_orbits["motif_orbit"]
            self.assertEqual(len(orbit.transform_ids), 2)
            self.assertEqual(len(orbit.group_instance_ids), 2)

    def test_multi_participant_interface_fails_closed(self) -> None:
        intent = SimpleCageIntentSpec.model_validate(
            {
                "name": "three-participant-site",
                "input": self.structure,
                "goal": {"architecture": "ring", "symmetry": ["C3"]},
                "interface_seeds": {
                    "cooperative_site": {
                        "participants": ["A", "B", "C"],
                        "selectors": {
                            "A": "A/1-2/*",
                            "B": "B/1-2/*",
                            "C": "C/1-2/*",
                        },
                    }
                },
                "generation": {"length": 40},
            }
        )

        with self.assertRaises((NotImplementedError, ValueError)) as context:
            enumerate_simple_design_candidates(intent)

        message = str(context.exception).lower()
        self.assertTrue(
            any(
                word in message
                for word in ("multi-participant", "hyperedge", "binary")
            ),
            message,
        )

    def test_multi_seed_homomer_requires_path_equivalence_proof(
        self,
    ) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["interface_seeds"]["second_interface"] = {
            "participants": ["C", "D"],
            "selectors": {
                "C": "C/1-2/*",
                "D": "D/1-2/*",
            },
            "use": "auto",
            "geometry": "preserve_exact",
        }
        intent = SimpleCageIntentSpec.model_validate(payload)

        with self.assertRaises((NotImplementedError, ValueError)) as context:
            enumerate_simple_design_candidates(intent)

        message = str(context.exception).lower()
        self.assertIn("homomer", message)
        self.assertTrue(
            any(word in message for word in ("equivalent", "equivalence")),
            message,
        )

    def test_resolve_cli_dispatches_simple_intent_and_prints_manifest(
        self,
    ) -> None:
        intent_path = self.root / "simple_intent.yaml"
        intent_path.write_text(
            yaml.safe_dump(
                self._intent().model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_directory = self.root / "resolution"
        expected = {
            "resolver": "mock-simple-resolver",
            "candidate_count": 4,
            "accepted_count": 4,
            "selected_count": 4,
            "ranking": [],
            "manifest_path": str(
                output_directory / "resolution_manifest.json"
            ),
        }
        stdout = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent",
            return_value=expected,
        ) as resolver, redirect_stdout(stdout):
            main(
                [
                    "resolve",
                    str(intent_path),
                    "--output-dir",
                    str(output_directory),
                    "--format",
                    "json",
                ]
            )

        self.assertEqual(json.loads(stdout.getvalue()), expected)
        resolver.assert_called_once()
        positional, keywords = resolver.call_args
        self.assertIsInstance(positional[0], SimpleCageIntentSpec)
        self.assertEqual(positional[0].input, self.structure.resolve())
        self.assertEqual(positional[1], output_directory)
        self.assertEqual(keywords["source_path"], intent_path)
        self.assertIsNone(keywords["symmetry_ids"])
        self.assertEqual(keywords["pose_samples"], 1)
        self.assertEqual(keywords["seed_start"], 0)
        self.assertEqual(keywords["top_count"], 20)
        self.assertEqual(keywords["max_candidates"], 4096)
        self.assertTrue(keywords["optimize_poses"])
        self.assertEqual(keywords["pose_optimize_top"], 4)
        self.assertEqual(keywords["pose_optimization_levels"], 3)
        self.assertEqual(keywords["pose_maximum_translation"], 12.0)
        self.assertEqual(keywords["pose_maximum_rotation_deg"], 25.0)

    def test_resolve_cli_rejects_already_standard_public_design(self) -> None:
        design_path = self.root / "standard_design.yaml"
        design_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "already-standard",
                    "input": str(self.structure),
                    "symmetry": "C3",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        stderr = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent"
        ) as resolver, redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(
                [
                    "resolve",
                    str(design_path),
                    "--output-dir",
                    str(self.root / "unused"),
                ]
            )

        resolver.assert_not_called()
        self.assertIn(
            "resolve expects kind: simple_cage_intent",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
