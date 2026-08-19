import json
import tempfile
import unittest
from pathlib import Path

import torch

from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig,
    apply_scaffold_core_guidance,
    build_scaffold_core_topology,
    scaffold_core_energy,
)
from rfd3.trainer.rfd3 import _copy_sampler_diagnostics
from rfd3_mosaic.rfd3_scaffold_core_audit import (
    audit_scaffold_core_guidance,
)


def features(tokens_per_chain: int = 8):
    count = 2 * tokens_per_chain
    return {
        "atom_to_token_map": torch.arange(count),
        "asym_id": torch.tensor(
            [0] * tokens_per_chain + [1] * tokens_per_chain
        ),
        "residue_index": torch.tensor(
            list(range(tokens_per_chain)) * 2
        ),
        "is_ca": torch.ones(count, dtype=torch.bool),
        "is_protein": torch.ones(count, dtype=torch.bool),
    }


class ScaffoldCoreGuidanceTestCase(unittest.TestCase):
    def test_result_writer_preserves_core_diagnostics_for_every_design(
        self,
    ) -> None:
        payload = {
            "scaffold_core_guidance_diagnostics": {
                "runtime_active": True,
                "applied_steps": 3,
            }
        }
        metadata = {0: {"metrics": {}}, 1: {"metrics": {}}}

        _copy_sampler_diagnostics(payload, metadata)

        for design in metadata.values():
            self.assertIs(
                design["scaffold_core_guidance_diagnostics"],
                payload["scaffold_core_guidance_diagnostics"],
            )

    def test_soft_inter_penalty_distinguishes_incidental_from_broad_contact(
        self,
    ) -> None:
        f = features()
        fixed = torch.zeros(16, dtype=torch.bool)
        topology = build_scaffold_core_topology(f, fixed)
        config = ScaffoldCoreGuidanceConfig(
            intra_chain_weight=0.0,
            inter_chain_weight=0.0,
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
        broad = scaffold_core_energy(
            torch.cat((left, broad_right)), topology, config
        )

        self.assertGreater(
            broad.inter_chain_excess.item(),
            scattered.inter_chain_excess.item(),
        )
        self.assertGreater(
            broad.generated_inter_chain_contact_coverage.item(),
            scattered.generated_inter_chain_contact_coverage.item(),
        )

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
        coordinates = torch.cat(
            (chain, chain + torch.tensor([0.0, 40.0, 0.0]))
        )[None, ...]

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
                                "automatic_symmetric_scaffold_packing": {
                                    "mode": "symmetric_generated",
                                    "intra_chain_weight": 1.0,
                                    "inter_chain_weight": 0.1,
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
            report["summary"]["final_metrics"]
            ["generated_inter_chain_contact_pairs"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
