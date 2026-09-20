import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray

from rfd3_mosaic.rfd3_scaffold_audit import _audit_final_generated_route_ownership
from rfd3_mosaic.structure import AtomRecord
from rfd3_mosaic.validation.generated_route_ownership import audit_generated_route_ownership
from rfd3_mosaic.validation.scaffold_validity import audit_scaffold_geometry


def _c3_input(*, center: bool = False):
    coordinates, chains, residues, fixed = [], [], [], []
    for i, chain in enumerate(("A", "B", "C")):
        theta = i * 2.0 * np.pi / 3.0
        rotation = np.asarray([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        local = np.asarray([[20.0, -8.0, 0.0], [20.0, 0.0, 0.0], [20.0, 8.0, 0.0]])
        local = local @ rotation.T
        if center:
            local[1] = [0.0, 0.0, i * 7.0]
        coordinates.extend(local)
        chains.extend([chain] * 3)
        residues.extend([1, 2, 3])
        fixed.extend([True, False, True])
    return dict(
        coordinates=np.asarray(coordinates),
        chain_ids=chains,
        residue_numbers=residues,
        fixed_mask=fixed,
    )


class GeneratedRouteOwnershipTestCase(unittest.TestCase):
    def test_axis_ties_fail_even_without_ca_clashes(self):
        inputs = _c3_input(center=True)
        atoms = tuple(
            AtomRecord("ATOM", i + 1, "CA", "", "GLY", chain, residue, "", tuple(xyz), "C")
            for i, (chain, residue, xyz) in enumerate(zip(
                inputs["chain_ids"], inputs["residue_numbers"], inputs["coordinates"], strict=True
            ))
        )
        geometry = audit_scaffold_geometry(atoms)
        report = audit_generated_route_ownership(**inputs)

        self.assertEqual(geometry["summary"]["ca_clash_count"], 0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["checked_sample_count"], 9)
        self.assertEqual(report["checked_pair_count"], 18)
        self.assertGreaterEqual(report["violated_sample_count"], 3)
        self.assertAlmostEqual(report["maximum_excess_angstrom"], 1.6)
        self.assertFalse(report["topological_non_interlocking_guarantee"])

    def test_separate_c3_regions_pass_without_excluding_interfaces(self):
        report = audit_generated_route_ownership(**_c3_input())

        self.assertTrue(report["passed"])
        self.assertTrue(report["applicable"])
        self.assertEqual(report["violated_sample_count"], 0)
        self.assertEqual(report["bounded_generated_run_count"], 3)

    def test_midpoint_violation_is_detected_when_generated_ca_passes(self):
        report = audit_generated_route_ownership(
            coordinates=np.asarray([
                [0, -6, 0], [0, 5, 0], [0, 6, 0],
                [-6, 0, 0], [5, 0, 0], [6, 0, 0],
            ]),
            chain_ids=["A"] * 3 + ["B"] * 3,
            residue_numbers=[1, 2, 3] * 2,
            fixed_mask=[True, False, True] * 2,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["ambiguous_reference_pair_count"], 0)
        for run in report["runs"]:
            self.assertEqual(len(run["violating_samples"]), 1)
            self.assertEqual(run["violating_samples"][0]["sample_kind"], "edge_midpoint")
            self.assertAlmostEqual(run["maximum_excess_angstrom"], 0.3)

    def test_rigid_pose_covariance(self):
        inputs = _c3_input(center=True)
        first = audit_generated_route_ownership(**inputs)
        rotation = np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        inputs["coordinates"] = inputs["coordinates"] @ rotation.T + [17, -9, 5]
        moved = audit_generated_route_ownership(**inputs)

        self.assertEqual(first["violated_sample_count"], moved["violated_sample_count"])
        self.assertAlmostEqual(first["maximum_excess_angstrom"], moved["maximum_excess_angstrom"])

    def test_coincident_reversed_chords_are_reported(self):
        report = audit_generated_route_ownership(
            coordinates=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 5], [0, 0, 0]]),
            chain_ids=["A"] * 3 + ["B"] * 3,
            residue_numbers=[1, 2, 3] * 2,
            fixed_mask=[True, False, True] * 2,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["ambiguous_reference_pair_count"], 1)

    def test_terminal_runs_and_residue_gaps_are_explicitly_outside_coverage(self):
        report = audit_generated_route_ownership(
            coordinates=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            chain_ids=["A"] * 4,
            residue_numbers=[1, 2, 8, 9],
            fixed_mask=[True, False, False, True],
        )
        self.assertFalse(report["applicable"])
        self.assertEqual(report["bounded_generated_run_count"], 0)
        self.assertEqual(report["uncovered_generated_run_count"], 2)
        self.assertEqual(report["uncovered_generated_ca_count"], 2)

    def test_reference_margin_conflict_is_detected_for_distinct_nearby_chords(self):
        report = audit_generated_route_ownership(
            coordinates=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]]),
            chain_ids=["A"] * 3 + ["B"] * 3,
            residue_numbers=[1, 2, 3] * 2,
            fixed_mask=[True, False, True] * 2,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["ambiguous_reference_pair_count"], 0)
        self.assertEqual(report["reference_margin_conflict_count"], 2)
        conflict = report["reference_margin_conflicts"][0]
        self.assertAlmostEqual(conflict["distance_difference_upper_bound_angstrom"], 1.0)
        self.assertAlmostEqual(conflict["required_maximum_margin_angstrom"], 1.6)

    def test_invalid_and_missing_coordinates_fail_closed(self):
        inputs = _c3_input()
        inputs["coordinates"][1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            audit_generated_route_ownership(**inputs)
        with self.assertRaisesRegex(ValueError, "positive"):
            audit_generated_route_ownership(**_c3_input(), routing_anchor_taper_residues=0)

    def test_final_audit_uses_output_coordinates_and_runtime_fixed_mask(self):
        inputs = _c3_input(center=True)
        # Use the actual parser container: it does NOT carry model feature
        # annotation is_protein until the inference feature transform runs.
        runtime = AtomArray(9)
        runtime.atom_name = np.asarray(["CA"] * 9)
        runtime.res_name = np.asarray(["GLY"] * 9)
        runtime.chain_id = np.asarray(inputs["chain_ids"])
        runtime.res_id = np.asarray(inputs["residue_numbers"])
        runtime.set_annotation("is_motif_atom_with_fixed_coord", np.asarray(inputs["fixed_mask"]))
        runtime.coord = _c3_input()["coordinates"]
        self.assertFalse(hasattr(runtime, "is_protein"))
        atoms = tuple(
            AtomRecord("ATOM", i + 1, "CA", "", "GLY", chain, residue, "", tuple(xyz), "C")
            for i, (chain, residue, xyz) in enumerate(zip(
                inputs["chain_ids"], inputs["residue_numbers"], inputs["coordinates"], strict=True
            ))
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps({"test": {"extra": {"generated_cross_chain_topology_guidance": {"enabled": True}}}}))
            report = _audit_final_generated_route_ownership(input_path=path, output_atoms=atoms, atom_array=runtime)
            self.assertTrue(report["declared"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["anchor_coordinate_source"], "final_output")
            with self.assertRaisesRegex(ValueError, "align"):
                _audit_final_generated_route_ownership(input_path=path, output_atoms=atoms[:-1], atom_array=runtime)


if __name__ == "__main__":
    unittest.main()
