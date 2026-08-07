import tempfile
import unittest
from pathlib import Path

import yaml

from rfd3_mosaic.graph_search import (
    graph_neighbour_assignments,
    search_graph_design,
)
from rfd3_mosaic.schema import load_user_design


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = (
    REPOSITORY_ROOT
    / "examples/rfd3_mosaic/public_multi_face_component.yaml"
)


class GraphSearchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary_directory.name)
        self.design = load_user_design(EXAMPLE)
        # This fixture exercises search/replay mechanics rather than interface
        # discovery. Explicit zero preserves its historical geometry-only
        # meaning; production preserve_input edges now require contact.
        interface = self.design.interfaces[0]
        relation = interface.relation.model_copy(
            update={"minimum_heavy_atom_contacts": 0}
        )
        self.design = self.design.model_copy(
            update={
                "interfaces": (
                    interface.model_copy(update={"relation": relation}),
                )
            }
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_enumerates_canonical_nonidentity_neighbours(self) -> None:
        assignments = graph_neighbour_assignments(
            self.design,
            interface_ids=("alpha_beta_neighbour",),
        )

        self.assertEqual(
            assignments,
            (
                {"alpha_beta_neighbour": "C3:r1"},
                {"alpha_beta_neighbour": "C3:r2"},
            ),
        )

    def test_rejects_combinatorial_search_before_compilation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding max_combinations"):
            graph_neighbour_assignments(
                self.design,
                interface_ids=("alpha_beta_neighbour",),
                include_identity=True,
                max_combinations=2,
            )

    def test_search_ranks_and_freezes_replayable_public_designs(self) -> None:
        report = search_graph_design(
            self.design,
            self.output / "search",
            source_path=EXAMPLE,
            interface_ids=("alpha_beta_neighbour",),
            top_count=2,
        )

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["failed_compilation_count"], 0)
        self.assertEqual(report["replay_failure_count"], 0)
        self.assertEqual(report["selected_count"], 2)
        self.assertTrue(Path(report["manifest_path"]).is_file())
        self.assertEqual(
            {
                candidate["neighbour_transforms"][
                    "alpha_beta_neighbour"
                ]
                for candidate in report["ranking"]
            },
            {"C3:r1", "C3:r2"},
        )
        for candidate in report["ranking"]:
            self.assertTrue(candidate["accepted"])
            self.assertTrue(candidate["replay_validated"])
            self.assertTrue(Path(candidate["replay_directory"]).is_dir())
            # The standalone compiler keeps the two selected source
            # fragments as separate chains in each of the three C3 copies.
            self.assertEqual(candidate["replay_chain_count"], 6)
            resolved = Path(candidate["resolved_design"])
            self.assertTrue(resolved.is_file())
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            relation = payload["interfaces"][0]["copy_relation"]
            self.assertIn(relation["transform"], {"C3:r1", "C3:r2"})

    def test_search_keeps_unsatisfied_output_targets_for_diffusion(self) -> None:
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["input"] = str(self.design.input)
        payload["interfaces"][0]["relation"] = {
            "mode": "contact",
            "distance": {"minimum": 0.01, "maximum": 0.02},
        }
        config = self.output / "output_target.yaml"
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        design = load_user_design(config)

        report = search_graph_design(
            design,
            self.output / "output_target_search",
            source_path=config,
            interface_ids=("alpha_beta_neighbour",),
        )

        # A contact relation is an output-stage sampler objective.  Its
        # absence in the initialized assembly must not prevent the second
        # supported workflow: generating a new interface around a supplied
        # motif.  The search report still makes the unmet target explicit.
        self.assertEqual(report["accepted_count"], 2)
        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(report["replay_failure_count"], 0)
        self.assertTrue(
            all(
                candidate["requires_diffusion_interface_formation"]
                for candidate in report["ranking"]
            )
        )

    def test_search_does_not_select_linker_infeasible_candidates(self) -> None:
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["input"] = str(self.design.input)
        # The selected endpoint separation requires more than one residue.
        # This is a genuine static impossibility, unlike an output-stage
        # contact which diffusion is explicitly responsible for creating.
        payload["connections"][0]["length"] = {
            "minimum": 1,
            "maximum": 1,
        }
        config = self.output / "linker_infeasible.yaml"
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        design = load_user_design(config)

        report = search_graph_design(
            design,
            self.output / "linker_infeasible_search",
            source_path=config,
            interface_ids=("alpha_beta_neighbour",),
        )

        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["replay_failure_count"], 0)
        self.assertEqual(
            list(
                (
                    self.output
                    / "linker_infeasible_search/selected"
                ).iterdir()
            ),
            [],
        )

    def test_search_compares_multiple_candidate_symmetries(self) -> None:
        report = search_graph_design(
            self.design,
            self.output / "multi_symmetry_search",
            source_path=EXAMPLE,
            symmetry_ids=("C2", "C3"),
            interface_ids=("alpha_beta_neighbour",),
            top_count=3,
        )

        self.assertEqual(report["searched_symmetries"], ["C2", "C3"])
        self.assertIsNone(report["symmetry"])
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(
            {candidate["symmetry"] for candidate in report["ranking"]},
            {"C2", "C3"},
        )
        for candidate in report["ranking"]:
            if not candidate.get("resolved_design"):
                continue
            payload = yaml.safe_load(
                Path(candidate["resolved_design"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["symmetry"], candidate["symmetry"])


if __name__ == "__main__":
    unittest.main()
