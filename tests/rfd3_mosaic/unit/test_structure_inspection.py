import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

from rfd3_mosaic.cli import main
from rfd3_mosaic.schema import SimpleCageIntentSpec, load_simple_cage_intent
from rfd3_mosaic.simple_architecture import analyze_simple_architectures
from rfd3_mosaic.structure_inspection import (
    inspect_declared_interface_relation,
    inspect_declared_interface_seed,
    inspect_structure_interfaces,
    simple_intent_payload,
    write_structure_inspection,
)


def _atom_line(
    serial: int,
    atom_name: str,
    chain: str,
    residue: int,
    x: float,
    y: float,
    z: float,
) -> str:
    element = atom_name[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2s}\n"
    )


class StructureInspectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "two_interfaces.pdb"
        lines: list[str] = []
        serial = 1
        for chain, y in (("A", 0.0), ("B", 3.0), ("C", 50.0)):
            for residue, x in ((1, 0.0), (2, 2.0)):
                for atom_name, dz in (("CA", 0.0), ("CB", 1.0)):
                    lines.append(
                        _atom_line(
                            serial,
                            atom_name,
                            chain,
                            residue,
                            x,
                            y,
                            dz,
                        )
                    )
                    serial += 1
        self.structure.write_text("".join(lines) + "END\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_detects_chain_pair_and_emits_short_intent(self) -> None:
        inspection = inspect_structure_interfaces(self.structure)

        self.assertEqual(
            tuple(item.chain_id for item in inspection.chains),
            ("A", "B", "C"),
        )
        self.assertEqual(len(inspection.interface_candidates), 1)
        candidate = inspection.interface_candidates[0]
        self.assertEqual((candidate.left_chain, candidate.right_chain), ("A", "B"))
        self.assertEqual(candidate.left_selector, "A/1-2/*")
        self.assertEqual(candidate.right_selector, "B/1-2/*")

        payload = simple_intent_payload(
            inspection,
            name="ordinary-cage",
            symmetries=("C3", "T"),
        )
        intent = SimpleCageIntentSpec.model_validate(payload)
        self.assertEqual(intent.goal.symmetry, ("C3", "T"))
        self.assertEqual(
            intent.interface_seeds["interface_A_B"].participants,
            ("A", "B"),
        )
        self.assertEqual(
            intent.interface_seeds["interface_A_B"].use.description,
            "auto",
        )
        self.assertEqual(intent.inspection.contact_cutoff, 4.5)
        incidence = {
            item.chain_id: item
            for item in inspection.component_interface_sets
        }
        self.assertEqual(incidence["A"].detected_port_count, 1)
        self.assertEqual(incidence["B"].interface_ids, ("interface_A_B",))
        self.assertEqual(incidence["C"].detected_port_count, 0)

    def test_separates_disconnected_contact_patches_on_same_chain_pair(
        self,
    ) -> None:
        structure = self.root / "two_patches.pdb"
        lines: list[str] = []
        serial = 1
        for chain, y in (("A", 0.0), ("B", 3.0)):
            for residue, x in ((1, 0.0), (2, 2.0), (10, 40.0), (11, 42.0)):
                for atom_name, dz in (("CA", 0.0), ("CB", 1.0)):
                    lines.append(
                        _atom_line(
                            serial,
                            atom_name,
                            chain,
                            residue,
                            x,
                            y,
                            dz,
                        )
                    )
                    serial += 1
        structure.write_text("".join(lines) + "END\n", encoding="utf-8")

        inspection = inspect_structure_interfaces(structure)

        self.assertEqual(len(inspection.interface_candidates), 2)
        self.assertEqual(
            {item.interface_id for item in inspection.interface_candidates},
            {
                "interface_A_B_patch_001",
                "interface_A_B_patch_002",
            },
        )
        self.assertEqual(
            {item.left_selector for item in inspection.interface_candidates},
            {"A/1-2/*", "A/10-11/*"},
        )
        self.assertTrue(
            all(
                item.contact_patch_count == 2
                for item in inspection.interface_candidates
            )
        )
        incidence = {
            item.chain_id: item.detected_port_count
            for item in inspection.component_interface_sets
        }
        self.assertEqual(incidence, {"A": 2, "B": 2})

    def test_written_intent_replays_nondefault_inspection_parameters(self) -> None:
        inspection = inspect_structure_interfaces(
            self.structure,
            contact_cutoff=3.25,
            minimum_atom_contacts=2,
            minimum_contact_residues_per_side=1,
        )
        _, intent_path = write_structure_inspection(
            inspection,
            self.root / "inspection",
            intent_name="replayable-cage",
        )

        intent = load_simple_cage_intent(intent_path)
        self.assertEqual(intent.inspection.contact_cutoff, 3.25)
        self.assertEqual(intent.inspection.minimum_atom_contacts, 2)
        self.assertEqual(
            intent.inspection.minimum_contact_residues_per_side,
            1,
        )

    def test_declared_seed_binds_exact_user_selectors(self) -> None:
        evidence = inspect_declared_interface_seed(
            self.structure,
            interface_id="selected_ab",
            left_chain="A",
            right_chain="B",
            left_selector="A/1-2/*",
            right_selector="B/1-2/*",
        )

        self.assertGreaterEqual(evidence.heavy_atom_contact_count, 4)
        with self.assertRaisesRegex(ValueError, "matched no atoms"):
            inspect_declared_interface_seed(
                self.structure,
                interface_id="bad_ab",
                left_chain="A",
                right_chain="B",
                left_selector="A/90-99/*",
                right_selector="B/1-2/*",
            )

    def test_inspect_cli_writes_a_valid_editable_intent(self) -> None:
        output = StringIO()
        destination = self.root / "ordinary"

        with redirect_stdout(output):
            main(
                [
                    "inspect",
                    str(self.structure),
                    "--output-dir",
                    str(destination),
                    "--symmetry",
                    "C3",
                    "--symmetry",
                    "T",
                    "--subunits-min",
                    "12",
                    "--subunits-max",
                    "60",
                    "--diameter-min",
                    "80",
                    "--diameter-max",
                    "160",
                ]
            )

        intent_path = destination / "simple_design.yaml"
        self.assertTrue(intent_path.is_file())
        payload = yaml.safe_load(intent_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["kind"], "simple_cage_intent")
        self.assertEqual(payload["goal"]["symmetry"], ["C3", "T"])
        self.assertEqual(
            payload["goal"]["subunits"],
            {"minimum": 12, "maximum": 60},
        )
        self.assertEqual(
            payload["goal"]["diameter_angstrom"],
            {"minimum": 80.0, "maximum": 160.0},
        )
        self.assertEqual(
            payload["interface_seeds"]["interface_A_B"]["use"],
            "auto",
        )
        self.assertIn("editable simple YAML", output.getvalue())
        self.assertIn("detected component faces", output.getvalue())

        validation = StringIO()
        with redirect_stdout(validation):
            main(["validate", str(intent_path)])
        self.assertIn(
            "Simple cage intent validation: PASSED",
            validation.getvalue(),
        )
        self.assertIn("executable: no", validation.getvalue())

        plan_json = StringIO()
        with redirect_stdout(plan_json):
            main(["plan", str(intent_path), "--format", "json"])
        plan_payload = yaml.safe_load(plan_json.getvalue())
        self.assertEqual(plan_payload["authoring_mode"], "ordinary")
        self.assertEqual(plan_payload["resolution_stage"], "intent")
        self.assertFalse(plan_payload["executable"])
        self.assertEqual(
            plan_payload["generation"]["length"],
            {"minimum": 40, "maximum": 100},
        )
        declared_sets = {
            item["participant"]: item
            for item in plan_payload["declared_component_interface_sets"]
        }
        self.assertEqual(declared_sets["A"]["declared_port_count"], 1)
        self.assertEqual(
            declared_sets["B"]["interface_ids"],
            ["interface_A_B"],
        )
        self.assertIn(
            "directed scaffold connection order",
            plan_payload["blocking_unresolved_variables"],
        )

    def test_interface_use_resolves_exact_stabilizer_coset_orbits(self) -> None:
        inspection = inspect_structure_interfaces(self.structure)
        payload = simple_intent_payload(
            inspection,
            name="twelve-interface-cage",
            architecture="cage",
        )
        payload["interface_seeds"]["interface_A_B"]["use"] = 12
        intent = SimpleCageIntentSpec.model_validate(payload)

        hypotheses = analyze_simple_architectures(intent)
        accepted = {item.symmetry for item in hypotheses if item.accepted}

        self.assertEqual(accepted, {"D6", "T", "O", "I"})
        by_symmetry = {item.symmetry: item for item in hypotheses}
        self.assertEqual(
            by_symmetry["T"].interface_orbit_actions["interface_A_B"][
                "stabilizer_order"
            ],
            1,
        )
        self.assertEqual(
            by_symmetry["O"].interface_orbit_actions["interface_A_B"][
                "stabilizer_order"
            ],
            2,
        )
        self.assertEqual(
            by_symmetry["I"].interface_orbit_actions["interface_A_B"][
                "stabilizer_order"
            ],
            5,
        )

    def test_three_instances_in_t_use_order_four_stabilizer(self) -> None:
        inspection = inspect_structure_interfaces(self.structure)
        payload = simple_intent_payload(
            inspection,
            name="three-interface-cage",
            architecture="cage",
        )
        payload["interface_seeds"]["interface_A_B"]["use"] = 3
        intent = SimpleCageIntentSpec.model_validate(payload)

        hypotheses = analyze_simple_architectures(intent)
        tetrahedral = next(
            item for item in hypotheses if item.symmetry == "T"
        )

        self.assertTrue(tetrahedral.accepted)
        action = tetrahedral.interface_orbit_actions["interface_A_B"]
        self.assertEqual(action["orbit_size"], 3)
        self.assertEqual(action["stabilizer_order"], 4)
        self.assertTrue(action["requires_geometric_stabilizer_validation"])

    def test_multi_participant_interface_uses_connected_contact_graph(self) -> None:
        # C is far away in the fixture, so A-B-C is not one connected site.
        disconnected = inspect_declared_interface_relation(
            self.structure,
            interface_id="three_way",
            participants=("A", "B", "C"),
            selectors={
                "A": "A/1-2/*",
                "B": "B/1-2/*",
                "C": "C/1-2/*",
            },
        )

        self.assertFalse(disconnected.contact_graph_connected)
        self.assertEqual(disconnected.active_contact_pairs, (("A", "B"),))


if __name__ == "__main__":
    unittest.main()
