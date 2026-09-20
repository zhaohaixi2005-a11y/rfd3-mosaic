import unittest

import numpy as np

from rfd3_mosaic.validation.generated_route_ownership import (
    audit_generated_route_ownership,
)
from rfd3_mosaic.validation.scaffold_contract import (
    audit_scaffold_contract,
    validate_scaffold_contract,
)


def contract_fixture():
    # Two separated curved CA paths; the old straight-chord ownership rejects
    # their upper arms even though the declared curved reference is feasible.
    a = np.array(
        [
            [0, 0, 0],
            [0, 3.8, 0],
            [0, 7.6, 0],
            [0, 11.4, 0],
            [3.8, 11.4, 0],
            [7.6, 11.4, 0],
            [7.6, 7.6, 0],
            [7.6, 3.8, 0],
            [7.6, 0, 0],
        ],
        dtype=float,
    )
    xyz = np.concatenate((a, a + [0, 0, 3.8]))
    fixed = [True, True, False, False, False, False, False, True, True] * 2
    contract = {
        "schema_version": 1,
        "residues": [
            {
                "chain_id": "AB"[i // 9],
                "residue_number": i % 9 + 1,
                "fixed": fixed[i],
                "reference_ca": v.tolist(),
            }
            for i, v in enumerate(xyz)
        ],
        "helix_blocks": [
            {"id": f"{chain}{side}", "residue_indices": [i + offset for i in indices]}
            for chain, offset in (("A", 0), ("B", 9))
            for side, indices in (("left", [1, 2, 3]), ("right", [5, 6, 7]))
        ],
        "support_edges": [{"left": c + "left", "right": c + "right"} for c in "AB"],
        "limits": {
            "maximum_ca_deviation": 2.0,
            "contact_distance": 8.0,
            "minimum_helix_contact_fraction": 1.0,
            "minimum_interchain_segment_distance": 1.0,
            "minimum_ca_bond_distance": 3.3,
            "maximum_ca_bond_distance": 4.3,
            "fixed_ca_tolerance": 1e-4,
            "geometry_tolerance": 1e-3,
            "maximum_unsupported_run": 0,
        },
    }
    return contract, xyz, np.array(fixed)


def audit(contract, xyz, *, align_fixed=False):
    return audit_scaffold_contract(
        contract=contract,
        coordinates=xyz,
        chain_ids=[r["chain_id"] for r in contract["residues"]],
        residue_numbers=[r["residue_number"] for r in contract["residues"]],
        fixed_mask=[r["fixed"] for r in contract["residues"]],
        align_fixed=align_fixed,
    )


class ReferenceScaffoldAuditTests(unittest.TestCase):
    def test_feasible_curved_reference_passes_despite_old_route_false_positive(self):
        contract, xyz, fixed = contract_fixture()
        old = audit_generated_route_ownership(
            coordinates=xyz,
            chain_ids=[r["chain_id"] for r in contract["residues"]],
            residue_numbers=[r["residue_number"] for r in contract["residues"]],
            fixed_mask=fixed,
        )
        self.assertFalse(old["passed"])
        self.assertTrue(audit(contract, xyz)["passed"])

    def test_gross_shape_and_unsupported_blocks_are_rejected(self):
        contract, xyz, fixed = contract_fixture()
        xyz[2:4, 0] += 20
        result = audit(contract, xyz)
        self.assertFalse(result["passed"])
        failed = {
            name
            for name, value in zip(
                result["deficit_categories"], result["deficits_angstrom"]
            )
            if value > 1e-3
        }
        self.assertIn("reference_ca_deviation", failed)
        self.assertIn("block_contact", failed)

    def test_invalid_reference_cannot_disable_previous_guard(self):
        contract, _, _ = contract_fixture()
        contract["residues"][4]["reference_ca"][1] += 10
        with self.assertRaisesRegex(ValueError, "reference violates"):
            validate_scaffold_contract(contract)

    def test_missing_limits_and_cross_chain_support_fail_closed(self):
        contract, _, _ = contract_fixture()
        del contract["limits"]["maximum_ca_deviation"]
        with self.assertRaises(ValueError):
            validate_scaffold_contract(contract)
        contract, _, _ = contract_fixture()
        contract["support_edges"][0]["right"] = "Bright"
        with self.assertRaisesRegex(ValueError, "same physical chain"):
            validate_scaffold_contract(contract)

    def test_common_rigid_alignment_preserves_metrics_but_independent_chain_motion_does_not(
        self,
    ):
        contract, xyz, _ = contract_fixture()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        moved = xyz @ rotation + [10, 20, -7]
        self.assertTrue(audit(contract, moved, align_fixed=True)["passed"])
        moved[9:] += [3, 0, 0]
        self.assertFalse(audit(contract, moved, align_fixed=True)["passed"])


class ReferenceScaffoldTensorTests(unittest.TestCase):
    def _bound(self, contract, xyz, fixed):
        import torch
        from rfd3.inference.symmetry.scaffold_core_guidance import (
            build_scaffold_core_topology,
        )

        f = {
            "atom_to_token_map": torch.arange(18),
            "asym_id": torch.arange(2).repeat_interleave(9),
            "residue_index": torch.arange(1, 10).repeat(2),
            "is_ca": torch.ones(18, dtype=torch.bool),
            "is_protein": torch.ones(18, dtype=torch.bool),
            "motif_pos": torch.tensor(xyz, dtype=torch.float64),
            "mosaic_scaffold_contract": contract,
            "mosaic_scaffold_ca_atom_indices": torch.arange(18),
        }
        return build_scaffold_core_topology(f, torch.tensor(fixed)), f

    def test_tensor_and_numpy_residuals_agree_on_nontrivial_perturbation(self):
        import torch
        from rfd3.inference.symmetry.generated_routes import generated_route_deficits

        contract, xyz, fixed = contract_fixture()
        topology, _ = self._bound(contract, xyz, fixed)
        moved = xyz.copy()
        moved[~fixed] += np.random.default_rng(21).normal(0, 1.4, (sum(~fixed), 3))
        tensor = generated_route_deficits(
            torch.tensor(moved, dtype=torch.float64), topology
        )
        independent = audit(contract, moved)
        np.testing.assert_allclose(
            tensor.numpy(), independent["deficits_angstrom"], atol=1e-8
        )
        self.assertGreater(independent["violated_constraint_count"], 0)
        self.assertTrue(
            torch.all(
                generated_route_deficits(
                    torch.tensor(xyz, dtype=torch.float64), topology
                )
                == 0
            )
        )

    def test_clean_guidance_moves_towards_template_preserving_fixed_and_copy_relationship(
        self,
    ):
        import torch
        from rfd3.inference.symmetry.generated_routes import (
            apply_generated_route_guidance,
            generated_route_deficits,
        )
        from rfd3.inference.symmetry.scaffold_core_guidance import (
            ScaffoldCoreGuidanceConfig,
        )

        contract, xyz, fixed = contract_fixture()
        contract["limits"]["maximum_ca_deviation"] = 0.2
        topology, _ = self._bound(contract, xyz, fixed)
        x = torch.tensor(xyz, dtype=torch.float64)[None]
        x[:, ~fixed, 1] += 0.3
        original = x.clone()

        def projector(value):
            result = value.clone()
            result[:, 9:] = result[:, :9] + result.new_tensor([0, 0, 3.8])
            result[:, fixed] = original[:, fixed]
            return result

        result, trace = apply_generated_route_guidance(
            x,
            topology,
            progress=0.5,
            config=ScaffoldCoreGuidanceConfig(routing_ownership_weight=1),
            projector=projector,
        )
        self.assertTrue(trace["applied"])
        self.assertLess(
            float(generated_route_deficits(result[0], topology).square().sum()),
            float(generated_route_deficits(x[0], topology).square().sum()),
        )
        torch.testing.assert_close(result[:, fixed], original[:, fixed])
        torch.testing.assert_close(
            result[:, 9:] - result[:, :9],
            torch.tensor([0, 0, 3.8], dtype=result.dtype).expand(1, 9, 3),
        )

    def test_wrong_fixed_frame_and_mobile_seed_contract_are_rejected(self):
        import torch
        from rfd3.inference.symmetry.scaffold_core_guidance import (
            build_scaffold_core_topology,
        )

        contract, xyz, fixed = contract_fixture()
        _, features = self._bound(contract, xyz, fixed)
        features["motif_pos"] = features["motif_pos"] + 1
        with self.assertRaisesRegex(ValueError, "same declared frame"):
            build_scaffold_core_topology(features, torch.tensor(fixed))
        features["motif_pos"] -= 1
        features["motif_constraint_orbit_mobility_mode"] = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "validated coupled reference transport plan"):
            build_scaffold_core_topology(features, torch.tensor(fixed))

    def test_sampler_activates_contract_without_legacy_route_switch_or_links(self):
        import torch
        from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry

        contract, xyz, fixed = contract_fixture()
        xyz[:9] += [10, 0, 0]
        xyz[9:] = xyz[:9] * [-1, -1, 1]
        for record, point in zip(contract["residues"], xyz):
            record["reference_ca"] = point.tolist()
        _, features = self._bound(contract, xyz, fixed)
        features.update(
            {
                "symmetry_id": "C2",
                "sym_entity_id": torch.zeros(18, dtype=torch.long),
                "sym_transform_id": torch.arange(2).repeat_interleave(9),
                "is_sym_asu": torch.arange(18) < 9,
                "sym_orbit_slot": torch.arange(9).repeat(2),
                "sym_orbit_slot_verified": torch.tensor(True),
                "sym_transform": {
                    "0": (torch.eye(3), torch.zeros(3)),
                    "1": (torch.diag(torch.tensor([-1.0, -1.0, 1.0])), torch.zeros(3)),
                },
                "ref_element": torch.zeros(18, dtype=torch.long),
                "is_motif_atom_with_fixed_coord": torch.tensor(fixed),
                "motif_constraint_group_membership": torch.tensor(fixed)[None],
                "partial_t": torch.full((18,), 10.0),
            }
        )
        coordinates = torch.tensor(xyz, dtype=torch.float32)[None]
        features["motif_pos"] = coordinates[0]

        class CleanReference(torch.nn.Module):
            def forward(self, X_noisy_L, **_):
                return {"X_L": coordinates.to(X_noisy_L)}

        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            num_timesteps=6,
            step_scale=1.0,
            preserve_fixed_motif_during_symmetry=True,
            require_motif_constraint_groups=True,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )
        self.assertFalse(sampler.enable_generated_cross_chain_topology_guidance)
        with torch.no_grad():
            result = sampler.sample_diffusion_like_af3(
                f=features,
                diffusion_module=CleanReference(),
                diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=coordinates,
                initializer_outputs={},
                ref_initializer_outputs=None,
                f_ref=None,
            )
        diagnostics = result["scaffold_core_guidance_diagnostics"]
        self.assertTrue(diagnostics["runtime_active"])
        self.assertTrue(
            diagnostics["route_contract"]["replaces_straight_chord_ownership"]
        )
        self.assertEqual(diagnostics["route_contract"]["limits"], contract["limits"])
        self.assertGreater(len(diagnostics["route_steps"]), 0)
        self.assertTrue(
            all(
                step["contract_kind"] == "explicit_full_scaffold"
                for step in diagnostics["route_steps"]
            )
        )
        torch.testing.assert_close(result["X_L"][:, fixed], coordinates[:, fixed])


class ReferenceScaffoldAdapterTests(unittest.TestCase):
    def test_final_adapter_uses_declared_contract_and_checks_identity(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from rfd3_mosaic.rfd3_scaffold_audit import (
            _audit_final_generated_route_ownership,
        )

        contract, xyz, fixed = contract_fixture()
        chain_ids = np.array([r["chain_id"] for r in contract["residues"]])
        res_ids = np.array([r["residue_number"] for r in contract["residues"]])
        inputs = SimpleNamespace(
            atom_name=np.array(["CA"] * 18),
            is_protein=np.ones(18, dtype=bool),
            chain_id=chain_ids,
            res_id=res_ids,
            is_motif_atom_with_fixed_coord=fixed,
        )
        outputs = [
            SimpleNamespace(
                record_type="ATOM",
                atom_name="CA",
                chain_id="XY"[i // 9],
                residue_number=int(res_ids[i]),
                insertion_code="",
                coordinate=xyz[i],
            )
            for i in range(18)
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(
                json.dumps(
                    {
                        "case": {
                            "extra": {
                                "mosaic_scaffold_contract": contract,
                                "generated_cross_chain_topology_guidance": {
                                    "enabled": True
                                },
                            }
                        }
                    }
                )
            )
            result = _audit_final_generated_route_ownership(
                input_path=path, output_atoms=outputs, atom_array=inputs
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["measurement"], "explicit_full_scaffold_contract")
            outputs[2].residue_number += 1
            with self.assertRaisesRegex(ValueError, "cannot align"):
                _audit_final_generated_route_ownership(
                    input_path=path, output_atoms=outputs, atom_array=inputs
                )

    def test_native_dispatch_refuses_to_ignore_scaffold_metadata(self):
        from rfd3.utils.inference import ensure_inference_sampler_matches_design_spec

        contract, _, _ = contract_fixture()
        example = {"partial_t": 5.0, "extra": {"mosaic_scaffold_contract": contract}}
        compatible = {
            "kind": "symmetry",
            "symmetry_state_mode": "orbit_average",
            "symmetry_noise_mode": "coupled",
            "preserve_fixed_motif_during_symmetry": True,
        }
        ensure_inference_sampler_matches_design_spec({"case": example}, compatible)
        for override in (
            {"kind": "default"},
            {"symmetry_state_mode": "legacy_asu"},
            {"symmetry_noise_mode": "independent"},
            {"symmetry_execution_backend": "local_neighbourhood"},
            {"enable_orbit_rigid_motif_mobility": True},
        ):
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, "would ignore"),
            ):
                ensure_inference_sampler_matches_design_spec(
                    {"case": example}, {**compatible, **override}
                )
        example["partial_t"] = None
        with self.assertRaisesRegex(ValueError, "partial_t"):
            ensure_inference_sampler_matches_design_spec({"case": example}, compatible)


if __name__ == "__main__":
    unittest.main()
