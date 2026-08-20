from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.backbone_comparison import (
    _run_directories,
    compare_hoyeung_backbone_campaign,
    flatness_twist_descriptors,
    paper_backbone_protocol,
)
from rfd3_mosaic.rfd3_batch_screen import (
    _ca_coordinates_by_chain,
    cyclic_ring_descriptors,
)
from rfd3_mosaic.structure import read_pdb_atoms


def _atom_line(
    serial: int,
    chain: str,
    residue: int,
    coordinate: tuple[float, float, float],
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d} {'CA':>4s} {'ALA':>3s} {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_c3(path: Path) -> None:
    coordinates = {
        "A": ((3.0, 0.0, 0.0), (17.0, 0.0, 2.0)),
        "B": ((3.5, 1.0, 1.0), (-13.5, 16.32, -1.0)),
        "C": ((3.0, -1.0, 0.5), (-13.0, -16.32, -0.5)),
    }
    lines = []
    serial = 1
    for chain, chain_coordinates in coordinates.items():
        for residue, coordinate in enumerate(chain_coordinates, start=1):
            lines.append(_atom_line(serial, chain, residue, coordinate))
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


class BackboneComparisonTestCase(unittest.TestCase):
    def test_local_campaign_recovers_unique_run_from_frozen_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            run = run_root / "2026-08-20" / "local-design" / "local-123"
            run.mkdir(parents=True)
            index = run_root / ".rfd3-mosaic" / "jobs"
            index.mkdir(parents=True)
            (index / "local-123.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "job_id": "local-123",
                        "experiment": "local-design",
                        "campaign": "local-campaign",
                        "run_directory": str(run),
                    }
                ),
                encoding="utf-8",
            )
            design = root / "shard.yaml"
            design.write_text(
                "name: local-design\n"
                "output:\n"
                "  root: /ignored\n"
                "  campaign: local-campaign\n",
                encoding="utf-8",
            )

            directories, unavailable = _run_directories(
                {
                    "records": [
                        {
                            "shard_index": 0,
                            "job_id": None,
                            "design": str(design),
                        }
                    ]
                },
                run_root=run_root,
            )

        self.assertEqual(directories, [run.resolve()])
        self.assertEqual(unavailable, [])

    def test_protocol_records_only_reported_backbone_contract(self) -> None:
        protocol = paper_backbone_protocol()
        self.assertEqual(
            protocol["matched_generation_case"]["interface_seed"],
            "LHD101",
        )
        self.assertEqual(
            protocol["matched_generation_case"]["diversity_analysis_backbones"],
            1000,
        )
        self.assertEqual(
            protocol["paper_backbone_screen"]["selection"],
            "lowest 50th percentile",
        )
        self.assertIn(
            "SolubleMPNN sequence design",
            protocol["out_of_scope_for_this_report"],
        )

    def test_flatness_and_twist_follow_written_ring_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            structure = Path(directory) / "c3.pdb"
            _write_c3(structure)
            atoms = read_pdb_atoms(structure)
            ring = cyclic_ring_descriptors(
                _ca_coordinates_by_chain(atoms),
                expected_order=3,
            )
            descriptors = flatness_twist_descriptors(atoms, ring)

        self.assertTrue(descriptors["available"])
        self.assertEqual(
            descriptors["method"],
            "paper_written_definition_reimplemented",
        )
        self.assertGreaterEqual(descriptors["flatness_degrees"], 0.0)
        self.assertLessEqual(descriptors["flatness_degrees"], 90.0)
        self.assertGreaterEqual(descriptors["twist_degrees"], 0.0)
        self.assertLessEqual(descriptors["twist_degrees"], 90.0)

    def test_campaign_report_keeps_missing_stride_out_of_strict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            run = run_root / "campaign" / "experiment" / "123"
            audit_dir = run / "audits" / "example_0"
            audit_dir.mkdir(parents=True)
            structure = run / "example_0_model_0.pdb"
            result_json = run / "example_0_model_0.json"
            _write_c3(structure)
            result_json.write_text("{}\n", encoding="utf-8")

            scaffold = audit_dir / "scaffold_validity_audit.json"
            scaffold.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "summary": {
                            "chain_break_count": 0,
                            "ca_clash_count": 0,
                            "maximum_symmetry_coordinate_rmsd": 0.00001,
                            "maximum_chain_ca_radius_of_gyration": 7.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            constraint = audit_dir / "constraint_orbit_audit.json"
            constraint.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "summary": {
                            "maximum_acceptance_rmsd": 0.00002,
                            "atom_completeness": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = {
                "status": "completed",
                "requested_designs": 1,
                "produced_designs": 1,
                "accepted_designs": 1,
                "design_results": [
                    {
                        "design_index": 0,
                        "design_id": "example_0",
                        "result_json": str(result_json),
                        "accepted": True,
                        "rejection_reason": None,
                        "reports": [str(scaffold), str(constraint)],
                    }
                ],
            }
            (run / "experiment_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            index = run_root / ".rfd3-mosaic" / "jobs"
            index.mkdir(parents=True)
            (index / "123.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "job_id": "123",
                        "run_directory": str(run),
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "campaign_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "total_designs": 1,
                        "records": [{"shard_index": 0, "job_id": "123"}],
                    }
                ),
                encoding="utf-8",
            )
            artifacts = compare_hoyeung_backbone_campaign(
                manifest,
                output_directory=root / "comparison",
                run_root=run_root,
            )
            payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["summary"]["generation_complete"])
        self.assertEqual(payload["summary"]["worker_accepted_count"], 1)
        self.assertEqual(payload["summary"]["seed_preserved_count"], 1)
        self.assertIsNone(payload["summary"]["loop_percentage"])
        self.assertEqual(
            payload["records"][0]["paper_backbone_filter"],
            "rg_lowest_half_only",
        )


if __name__ == "__main__":
    unittest.main()
