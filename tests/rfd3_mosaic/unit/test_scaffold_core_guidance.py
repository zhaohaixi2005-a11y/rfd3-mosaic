import json
import tempfile
import unittest
from pathlib import Path

import torch
from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig,
    apply_scaffold_core_guidance,
    build_scaffold_core_topology,
    generated_chain_core_centers,
    project_generated_polymer_continuity,
    robust_interface_capture_energy,
    scaffold_core_energy,
    worst_support_deficit_energy,
)
from rfd3.trainer.rfd3 import _copy_sampler_diagnostics

from rfd3_mosaic.rfd3_scaffold_core_audit import (
    audit_scaffold_core_guidance,
)


def features(tokens_per_chain: int = 8):
    count = 2 * tokens_per_chain
    return {
        "atom_to_token_map": torch.arange(count),
        "asym_id": torch.tensor([0] * tokens_per_chain + [1] * tokens_per_chain),
        "residue_index": torch.tensor(list(range(tokens_per_chain)) * 2),
        "is_ca": torch.ones(count, dtype=torch.bool),
        "is_protein": torch.ones(count, dtype=torch.bool),
    }


class ScaffoldCoreGuidanceTestCase(unittest.TestCase):
    def test_fixed_backbone_sidechain_is_not_generated_scaffold(self) -> None:
        topology = build_scaffold_core_topology(
            {
                "atom_to_token_map": torch.tensor([0, 0, 1, 1]),
                "asym_id": torch.tensor([0, 0]),
                "residue_index": torch.tensor([0, 1]),
                "is_ca": torch.tensor([True, False, True, False]),
                "is_protein": torch.tensor([True, True]),
            },
            torch.tensor([True, False, False, False]),
        )

        self.assertEqual(topology.generated_token_mask.tolist(), [False, True])
        self.assertEqual(
            topology.generated_atom_mask.tolist(),
            [False, False, True, True],
        )

    def test_support_weighted_center_downweights_one_long_unsupported_arm(self) -> None:
        topology = build_scaffold_core_topology(
            {
                "atom_to_token_map": torch.arange(7),
                "asym_id": torch.zeros(7, dtype=torch.long),
                "residue_index": torch.arange(7),
                "is_ca": torch.ones(7, dtype=torch.bool),
                "is_protein": torch.ones(7, dtype=torch.bool),
            },
            torch.tensor([True, False, False, False, False, False, False]),
        )
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 0.5, 0.0],
                [30.0, 0.5, 0.0],
            ]
        )
        ordinary, supported = generated_chain_core_centers(
            coordinates,
            topology,
            ScaffoldCoreGuidanceConfig(sequence_separation=2),
        )

        self.assertLess(float(supported[0, 0]), float(ordinary[0, 0]))

    def test_capture_targets_each_seed_copy_to_local_two_chain_midpoint(self) -> None:
        topology = build_scaffold_core_topology(
            features(tokens_per_chain=4),
            torch.tensor([True, False, False, False] * 2),
        )
        coordinates = torch.tensor(
            [
                [-2.0, 0.0, 0.0],
                [-2.0, 1.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-2.0, 3.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 3.0, 0.0],
            ]
        )
        groups = torch.tensor([[0, 4]])
        centered = robust_interface_capture_energy(
            coordinates,
            topology,
            ScaffoldCoreGuidanceConfig(sequence_separation=2),
            groups,
            capture_progress=0.0,
        )
        shifted = coordinates.clone()
        shifted[[0, 4], 0] += 4.0
        displaced = robust_interface_capture_energy(
            shifted,
            topology,
            ScaffoldCoreGuidanceConfig(sequence_separation=2),
            groups,
            capture_progress=0.0,
        )

        self.assertLess(float(centered), float(displaced))

    def test_worst_support_focuses_on_one_contiguous_unsupported_run(self) -> None:
        config = ScaffoldCoreGuidanceConfig(
            sequence_separation=4,
            target_supported_contacts=2.0,
        )
        contiguous = torch.tensor([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
        scattered = torch.tensor([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0])

        contiguous_energy = worst_support_deficit_energy(contiguous, config)
        scattered_energy = worst_support_deficit_energy(scattered, config)

        self.assertGreater(contiguous_energy.item(), scattered_energy.item())

    def test_routing_ownership_penalizes_generated_run_in_wrong_chain_cell(
        self,
    ) -> None:
        topology = build_scaffold_core_topology(
            features(tokens_per_chain=5),
            torch.tensor(
                [True, False, False, False, True] * 2,
                dtype=torch.bool,
            ),
        )
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 8.0, 0.0],
                [4.0, 8.0, 0.0],
                [6.0, 8.0, 0.0],
                [8.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [2.0, 10.0, 0.0],
                [4.0, 10.0, 0.0],
                [6.0, 10.0, 0.0],
                [8.0, 10.0, 0.0],
            ]
        )
        config = ScaffoldCoreGuidanceConfig(
            routing_ownership_weight=1.0,
            clash_weight=0.0,
            continuity_weight=0.0,
        )

        wrong = scaffold_core_energy(coordinates, topology, config)
        corrected = coordinates.clone()
        corrected[1:4, 1] = 0.0
        right = scaffold_core_energy(corrected, topology, config)

        self.assertEqual(len(topology.generated_runs), 2)
        self.assertGreater(wrong.routing_ownership.item(), 0.0)
        self.assertGreater(
            wrong.routing_ownership_violation_fraction.item(),
            0.0,
        )
        self.assertAlmostEqual(right.routing_ownership.item(), 0.0)

    def test_routing_guidance_moves_only_generated_tokens_toward_own_cell(
        self,
    ) -> None:
        fixed = torch.tensor(
            [True, False, False, False, True] * 2,
            dtype=torch.bool,
        )
        topology = build_scaffold_core_topology(
            features(tokens_per_chain=5),
            fixed,
        )
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 8.0, 0.0],
                    [4.0, 8.0, 0.0],
                    [6.0, 8.0, 0.0],
                    [8.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [2.0, 10.0, 0.0],
                    [4.0, 10.0, 0.0],
                    [6.0, 10.0, 0.0],
                    [8.0, 10.0, 0.0],
                ]
            ]
        )
        config = ScaffoldCoreGuidanceConfig(
            routing_ownership_weight=1.0,
            clash_weight=0.0,
            continuity_weight=0.0,
            maximum_token_step=0.4,
        )

        guided, diagnostics = apply_scaffold_core_guidance(
            coordinates,
            topology,
            progress=0.5,
            config=config,
            projector=lambda value: value,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertLess(
            diagnostics["final"]["routing_ownership"],
            diagnostics["initial"]["routing_ownership"],
        )
        self.assertTrue(torch.equal(guided[0, fixed], coordinates[0, fixed]))

    def test_polymer_projection_restores_generated_path_without_moving_fixed(
        self,
    ) -> None:
        f = {
            "atom_to_token_map": torch.arange(5),
            "asym_id": torch.zeros(5, dtype=torch.long),
            "residue_index": torch.arange(5),
            "is_ca": torch.ones(5, dtype=torch.bool),
            "is_protein": torch.ones(5, dtype=torch.bool),
        }
        fixed = torch.tensor([True, False, False, False, False])
        topology = build_scaffold_core_topology(f, fixed)
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [40.0, 0.0, 0.0],
                    [41.0, 0.0, 0.0],
                    [42.0, 0.0, 0.0],
                    [43.0, 0.0, 0.0],
                ]
            ]
        )

        projected, diagnostics = project_generated_polymer_continuity(
            coordinates,
            topology,
            iterations=128,
            projector=lambda value: value,
        )

        distances = torch.linalg.vector_norm(
            projected[0, 1:] - projected[0, :-1],
            dim=-1,
        )
        self.assertTrue(diagnostics["applied"])
        self.assertTrue(diagnostics["within_tolerance"])
        self.assertTrue(torch.equal(projected[0, 0], coordinates[0, 0]))
        self.assertTrue(torch.all(torch.abs(distances - 3.8) <= 0.5 + 1e-6))

    def test_polymer_projection_respects_both_fixed_linker_anchors(self) -> None:
        f = {
            "atom_to_token_map": torch.arange(5),
            "asym_id": torch.zeros(5, dtype=torch.long),
            "residue_index": torch.arange(5),
            "is_ca": torch.ones(5, dtype=torch.bool),
            "is_protein": torch.ones(5, dtype=torch.bool),
        }
        fixed = torch.tensor([True, False, False, False, True])
        topology = build_scaffold_core_topology(f, fixed)
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 8.0, 0.0],
                    [21.0, -4.0, 0.0],
                    [18.0, 2.0, 0.0],
                    [15.2, 0.0, 0.0],
                ]
            ]
        )

        projected, diagnostics = project_generated_polymer_continuity(
            coordinates,
            topology,
            iterations=128,
            projector=lambda value: value,
        )

        distances = torch.linalg.vector_norm(
            projected[0, 1:] - projected[0, :-1],
            dim=-1,
        )
        self.assertTrue(diagnostics["within_tolerance"])
        self.assertTrue(torch.equal(projected[0, fixed], coordinates[0, fixed]))
        self.assertTrue(torch.all(torch.abs(distances - 3.8) <= 0.5 + 1e-6))

    def test_crossing_backbone_segments_are_detected_without_ca_point_clash(
        self,
    ) -> None:
        topology = build_scaffold_core_topology(
            features(tokens_per_chain=2),
            torch.zeros(4, dtype=torch.bool),
        )
        coordinates = torch.tensor(
            [
                [-3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, -3.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        energy = scaffold_core_energy(
            coordinates,
            topology,
            ScaffoldCoreGuidanceConfig(
                intra_chain_weight=0.0,
                inter_chain_excess_penalty=0.0,
                continuity_weight=0.0,
                clash_distance=3.2,
            ),
        )

        self.assertEqual(energy.clash.item(), 0.0)
        self.assertAlmostEqual(
            energy.cross_chain_segment_clash.item(),
            3.2**2,
            places=5,
        )
        self.assertAlmostEqual(
            energy.minimum_cross_chain_segment_distance.item(),
            0.0,
        )
        self.assertGreater(energy.total.item(), 0.0)

    def test_fixed_fixed_segments_are_not_repulsed(self) -> None:
        f = {
            "atom_to_token_map": torch.arange(5),
            "asym_id": torch.tensor([0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([0, 1, 0, 1, 2]),
            "is_ca": torch.ones(5, dtype=torch.bool),
            "is_protein": torch.ones(5, dtype=torch.bool),
        }
        fixed = torch.tensor([True, True, True, True, False])
        topology = build_scaffold_core_topology(f, fixed)
        coordinates = torch.tensor(
            [
                [-3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, -3.0, 0.0],
                [0.0, 3.0, 0.0],
                [10.0, 3.0, 0.0],
            ]
        )
        energy = scaffold_core_energy(
            coordinates,
            topology,
            ScaffoldCoreGuidanceConfig(
                intra_chain_weight=0.0,
                continuity_weight=0.0,
                clash_distance=1.0,
            ),
        )

        self.assertEqual(energy.cross_chain_segment_clash.item(), 0.0)

    def test_result_writer_preserves_core_diagnostics_for_every_design(
        self,
    ) -> None:
        payload = {
            "scaffold_core_guidance_diagnostics": {
                "runtime_active": True,
                "applied_steps": 3,
            },
            "generated_polymer_continuity_diagnostics": {
                "runtime_active": True,
                "all_steps_within_tolerance": True,
            },
        }
        metadata = {0: {"metrics": {}}, 1: {"metrics": {}}}

        _copy_sampler_diagnostics(payload, metadata)

        for design in metadata.values():
            self.assertIs(
                design["scaffold_core_guidance_diagnostics"],
                payload["scaffold_core_guidance_diagnostics"],
            )
            self.assertIs(
                design["generated_polymer_continuity_diagnostics"],
                payload["generated_polymer_continuity_diagnostics"],
            )

    def test_explicit_soft_inter_penalty_distinguishes_broad_contact(
        self,
    ) -> None:
        f = features()
        fixed = torch.zeros(16, dtype=torch.bool)
        topology = build_scaffold_core_topology(f, fixed)
        config = ScaffoldCoreGuidanceConfig(
            intra_chain_weight=0.0,
            inter_chain_weight=0.0,
            inter_chain_excess_penalty=1.0,
            clash_weight=0.0,
            continuity_weight=0.0,
            sequence_separation=2,
        )
        left = torch.stack(
            (torch.arange(8.0) * 4.0, torch.zeros(8), torch.zeros(8)),
            dim=-1,
        )
        scattered_right = left + torch.tensor([0.0, 30.0, 0.0])
        scattered_right[0] = left[0] + torch.tensor([0.0, 7.0, 0.0])
        broad_right = left + torch.tensor([0.0, 7.0, 0.0])

        scattered = scaffold_core_energy(
            torch.cat((left, scattered_right)), topology, config
        )
        broad = scaffold_core_energy(torch.cat((left, broad_right)), topology, config)

        self.assertGreater(
            broad.inter_chain_excess.item(),
            scattered.inter_chain_excess.item(),
        )
        self.assertGreater(
            broad.generated_inter_chain_contact_coverage.item(),
            scattered.generated_inter_chain_contact_coverage.item(),
        )
        self.assertGreater(broad.total.item(), scattered.total.item())

    def test_inter_weight_has_no_implicit_repulsive_core_energy(self) -> None:
        f = features()
        topology = build_scaffold_core_topology(
            f,
            torch.zeros(16, dtype=torch.bool),
        )
        chain = torch.stack(
            (torch.arange(8.0) * 4.0, torch.zeros(8), torch.zeros(8)),
            dim=-1,
        )
        coordinates = torch.cat((chain, chain + torch.tensor([0.0, 7.0, 0.0])))
        low_inter = scaffold_core_energy(
            coordinates,
            topology,
            ScaffoldCoreGuidanceConfig(
                intra_chain_weight=1.0,
                inter_chain_weight=0.1,
                sequence_separation=2,
            ),
        )
        high_inter = scaffold_core_energy(
            coordinates,
            topology,
            ScaffoldCoreGuidanceConfig(
                intra_chain_weight=1.0,
                inter_chain_weight=1.0,
                sequence_separation=2,
            ),
        )

        self.assertGreater(low_inter.inter_chain_excess.item(), 0.0)
        self.assertAlmostEqual(low_inter.total.item(), high_inter.total.item())

    def test_intra_guidance_moves_only_generated_tokens_and_lowers_energy(
        self,
    ) -> None:
        f = features()
        fixed = torch.zeros(16, dtype=torch.bool)
        fixed[[0, 7, 8, 15]] = True
        topology = build_scaffold_core_topology(f, fixed)
        config = ScaffoldCoreGuidanceConfig(
            intra_chain_weight=1.0,
            inter_chain_weight=0.0,
            sequence_separation=2,
            clash_distance=0.1,
            backbone_tolerance=100.0,
            maximum_token_step=0.4,
        )
        chain = torch.stack(
            (torch.arange(8.0) * 6.0, torch.zeros(8), torch.zeros(8)),
            dim=-1,
        )
        coordinates = torch.cat((chain, chain + torch.tensor([0.0, 40.0, 0.0])))[
            None, ...
        ]

        guided, diagnostics = apply_scaffold_core_guidance(
            coordinates,
            topology,
            progress=0.5,
            config=config,
            projector=lambda value: value,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertLess(
            diagnostics["final"]["total"],
            diagnostics["initial"]["total"],
        )
        self.assertTrue(torch.equal(guided[0, fixed], coordinates[0, fixed]))
        self.assertFalse(torch.equal(guided, coordinates))
        self.assertLessEqual(
            diagnostics["maximum_adjacent_token_step_difference"],
            config.maximum_adjacent_token_step_difference + 1e-6,
        )

    def test_audit_reports_metrics_without_hard_coding_lhd_contact_counts(
        self,
    ) -> None:
        metrics = {
            "total": 1.0,
            "long_range_contacts": 0.2,
            "normalized_rg": 0.1,
            "tertiary_support": 0.3,
            "inter_chain_excess": 0.1,
            "clash": 0.0,
            "continuity": 0.0,
            "mean_normalized_rg": 2.5,
            "mean_tertiary_support_fraction": 0.7,
            "generated_inter_chain_contact_pairs": 4.0,
            "generated_inter_chain_contact_coverage": 0.05,
            "minimum_generated_inter_chain_distance": 6.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = root / "input.json"
            result = root / "result.json"
            compiled.write_text(
                json.dumps(
                    {
                        "case": {
                            "extra": {
                                "scaffold_core_guidance": {
                                    "mode": "intra_inter",
                                    "intra_chain_weight": 1.0,
                                    "inter_chain_weight": 0.1,
                                    "inter_chain_excess_penalty": 0.0,
                                    "quality_contract": {
                                        "required": True,
                                        "maximum_mean_normalized_rg": 2.6,
                                        "minimum_mean_tertiary_support_fraction": 0.5,
                                        "maximum_long_range_contact_deficit": 0.25,
                                    },
                                }
                            }
                        }
                    }
                )
            )
            result.write_text(
                json.dumps(
                    {
                        "scaffold_core_guidance_diagnostics": {
                            "runtime_active": True,
                            "chain_count": 3,
                            "config": {
                                "intra_chain_weight": 1.0,
                                "inter_chain_weight": 0.1,
                                "inter_chain_excess_penalty": 0.0,
                            },
                            "steps": [
                                {
                                    "applied": True,
                                    "initial": {"total": 2.0},
                                    "final": {"total": 1.5},
                                }
                            ],
                            "applied_steps": 1,
                            "final_metrics": metrics,
                        }
                    }
                )
            )
            report = audit_scaffold_core_guidance(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["summary"]["final_metrics"]["generated_inter_chain_contact_pairs"],
            4.0,
        )
        self.assertTrue(report["summary"]["scientific_quality_satisfied"])
        self.assertTrue(report["summary"]["declared_quality_targets_satisfied"])

    def test_audit_rejects_executed_but_open_scaffold(self) -> None:
        metrics = {
            "total": 1.0,
            "long_range_contacts": 1.0,
            "normalized_rg": 1.0,
            "tertiary_support": 1.0,
            "inter_chain_excess": 0.0,
            "clash": 0.0,
            "continuity": 0.0,
            "mean_normalized_rg": 3.4,
            "mean_tertiary_support_fraction": 0.1,
            "generated_inter_chain_contact_pairs": 0.0,
            "generated_inter_chain_contact_coverage": 0.0,
            "minimum_generated_inter_chain_distance": 20.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = root / "input.json"
            result = root / "result.json"
            compiled.write_text(
                json.dumps(
                    {
                        "case": {
                            "extra": {
                                "scaffold_core_guidance": {
                                    "mode": "intra_inter",
                                    "intra_chain_weight": 1.0,
                                    "inter_chain_weight": 0.1,
                                    "inter_chain_excess_penalty": 0.0,
                                    "quality_contract": {
                                        "required": True,
                                        "maximum_mean_normalized_rg": 2.6,
                                        "minimum_mean_tertiary_support_fraction": 0.5,
                                        "maximum_long_range_contact_deficit": 0.25,
                                    },
                                }
                            }
                        }
                    }
                )
            )
            result.write_text(
                json.dumps(
                    {
                        "scaffold_core_guidance_diagnostics": {
                            "runtime_active": True,
                            "chain_count": 3,
                            "config": {
                                "intra_chain_weight": 1.0,
                                "inter_chain_weight": 0.1,
                                "inter_chain_excess_penalty": 0.0,
                            },
                            "steps": [
                                {
                                    "applied": True,
                                    "initial": {"total": 2.0},
                                    "final": {"total": 1.0},
                                }
                            ],
                            "applied_steps": 1,
                            "final_metrics": metrics,
                        }
                    }
                )
            )
            report = audit_scaffold_core_guidance(
                compiled_input=compiled,
                result_json=result,
            )

            report_only_payload = json.loads(compiled.read_text())
            report_only_payload["case"]["extra"]["scaffold_core_guidance"][
                "quality_contract"
            ]["required"] = False
            compiled.write_text(json.dumps(report_only_payload))
            report_only = audit_scaffold_core_guidance(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["scientific_quality_satisfied"])
        self.assertFalse(report["summary"]["declared_quality_targets_satisfied"])
        self.assertTrue(report_only["passed"])
        self.assertFalse(report_only["summary"]["scientific_quality_satisfied"])
        self.assertTrue(report_only["summary"]["quality_gate_satisfied"])


if __name__ == "__main__":
    unittest.main()
