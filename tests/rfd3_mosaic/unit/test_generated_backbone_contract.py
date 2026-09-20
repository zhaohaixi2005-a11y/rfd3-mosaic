"""Actual-backbone contract regressions; no checkpoint, GPU or deposited fixture."""

import numpy as np
import pytest

from rfd3_mosaic.validation.generated_backbone import (
    audit_generated_backbone,
    default_backbone_policy,
    identify_alpha_helices,
)
from rfd3_mosaic.validation.scaffold_contract import (
    _measure,
    audit_scaffold_contract,
    validate_scaffold_contract,
)
from test_scaffold_builder import _reference


def backbone_contract_fixture():
    one = _reference(20, compact=True)
    xyz = np.concatenate((one, one + [0, 0, 35]))
    fixed = np.tile([True, True] + [False] * 18 + [True, True], 2)
    chains = ["A"] * 22 + ["B"] * 22
    segments = identify_alpha_helices(xyz, chains, ["ALA"] * 44)["segments"]
    contract = {
        "schema_version": 2,
        "residues": [
            {
                "chain_id": chains[i],
                "residue_number": i % 22 + 1,
                "fixed": bool(fixed[i]),
                "reference_ca": bb[1].tolist(),
                "reference_backbone": bb.tolist(),
                "residue_name": "ALA",
            }
            for i, bb in enumerate(xyz)
        ],
        "helix_blocks": [
            {"id": f"h{i}", "residue_indices": ids} for i, ids in enumerate(segments)
        ],
        "support_edges": [{"left": "h0", "right": "h1"}, {"left": "h2", "right": "h3"}],
        "limits": {
            "maximum_ca_deviation": 1.0,
            "contact_distance": 8.0,
            "minimum_helix_contact_fraction": 0.2,
            "minimum_interchain_segment_distance": 1.0,
            "minimum_ca_bond_distance": 3.3,
            "maximum_ca_bond_distance": 4.3,
            "fixed_ca_tolerance": 0.01,
            "geometry_tolerance": 0.001,
            "maximum_unsupported_run": 6,
        },
        "backbone_policy": default_backbone_policy(),
    }
    return contract, xyz


def _audit(contract, xyz):
    return audit_scaffold_contract(
        contract=contract,
        coordinates=xyz[:, 1],
        chain_ids=[r["chain_id"] for r in contract["residues"]],
        residue_numbers=[r["residue_number"] for r in contract["residues"]],
        fixed_mask=[r["fixed"] for r in contract["residues"]],
        backbone_coordinates=xyz,
        residue_names=["ALA"] * len(xyz),
    )


def test_reference_actual_backbone_passes_and_loops_are_not_forced_helical():
    contract, xyz = backbone_contract_fixture()
    validated = validate_scaffold_contract(contract)
    result = _audit(validated, xyz)
    assert result["passed"]
    quality = result["generated_backbone"]
    assert quality["required"]
    assert quality["actual_generated_alpha_residue_count"] < sum(
        not r["fixed"] for r in contract["residues"]
    )
    assert quality["uncovered_generated_alpha_residues"] == []
    assert quality["geometry_failure_count"] == 0


def test_fixed_only_blocks_cannot_satisfy_version2():
    contract, _ = backbone_contract_fixture()
    for r in contract["residues"]:
        r["fixed"] = True
    contract["residues"][10]["fixed"] = False  # Generated loop outside declared H.
    with pytest.raises(ValueError, match="fixed-only"):
        validate_scaffold_contract(contract)


def test_missing_actual_generated_helix_is_rejected():
    contract, xyz = backbone_contract_fixture()
    # The B-chain H segments still exist even when the author omits them.
    contract["helix_blocks"] = contract["helix_blocks"][:2]
    contract["support_edges"] = contract["support_edges"][:1]
    with pytest.raises(ValueError, match="undeclared_generated_alpha_residues"):
        validate_scaffold_contract(contract)


def test_declared_boundary_tolerance_allows_one_residue_extension_not_omitted_arm():
    contract, xyz = backbone_contract_fixture()
    # Actual H begins at index12; author marked13. The one-residue boundary
    # tolerance is symmetric, and support still measures index12 as well.
    contract["helix_blocks"][1]["residue_indices"] = list(range(13, 21))
    result = _audit(contract, xyz)
    assert result["passed"]
    support = result["generated_backbone"]["actual_helix_support"][1]
    assert support["residue_indices"][0] == 12
    contract["helix_blocks"][1]["residue_indices"] = list(range(14, 21))
    with pytest.raises(ValueError, match="undeclared_generated_alpha_residues"):
        validate_scaffold_contract(contract)


def test_splitting_one_long_actual_helix_does_not_create_tertiary_support():
    xyz = _reference(20)
    contract, _ = backbone_contract_fixture()
    contract["residues"] = contract["residues"][:22]
    for record, bb in zip(contract["residues"], xyz):
        record["reference_ca"], record["reference_backbone"] = (
            bb[1].tolist(),
            bb.tolist(),
        )
    contract["helix_blocks"] = [
        {"id": "left", "residue_indices": list(range(1, 9))},
        {"id": "right", "residue_indices": list(range(10, 21))},
    ]
    contract["support_edges"] = [{"left": "left", "right": "right"}]
    report = audit_generated_backbone(
        contract=contract, backbone_coordinates=xyz, residue_names=["ALA"] * 22
    )
    assert not report["passed"]
    assert "same_actual_helix_split_into_blocks" in {
        f["kind"] for f in report["helix_failures"]
    }


@pytest.mark.parametrize(
    "atom,displacement,expected",
    [
        (0, [0, 0, 4], "N_CA_length"),
        (2, [4, 0, 0], "CA_C_length"),
        (3, [0, 4, 0], "C_O_length"),
    ],
)
def test_ca_unchanged_cannot_hide_bad_generated_backbone(atom, displacement, expected):
    contract, xyz = backbone_contract_fixture()
    moved = xyz.copy()
    moved[5, atom] += displacement
    assert _measure(contract, moved[:, 1])["passed"]  # Old CA-only test passes.
    result = _audit(contract, moved)
    assert not result["passed"]
    assert expected in {
        f["kind"] for f in result["generated_backbone"]["geometry_failures"]
    }


def test_actual_helix_loss_is_detected_with_unchanged_ca_and_co_bond_length():
    contract, xyz = backbone_contract_fixture()
    moved = xyz.copy()
    for i in range(1, 9):
        moved[i, 3] = 2 * moved[i, 2] - moved[i, 3]
    result = _audit(contract, moved)
    assert not result["generated_backbone"]["helix_support_passed"]
    assert "declared_block_not_one_actual_helix" in {
        f["kind"] for f in result["generated_backbone"]["helix_failures"]
    }


def test_nonlocal_generated_backbone_collision_is_required():
    contract, xyz = backbone_contract_fixture()
    moved = xyz.copy()
    moved[5, 3] = moved[16, 0]
    report = _audit(contract, moved)["generated_backbone"]
    assert not report["passed"]
    assert report["nonlocal_backbone_clash_count"] >= 1


def test_maximum_unsupported_run_is_required_and_cannot_be_ignored():
    contract, xyz = backbone_contract_fixture()
    del contract["limits"]["maximum_unsupported_run"]
    with pytest.raises(ValueError, match="maximum_unsupported_run"):
        validate_scaffold_contract(contract)
    contract["limits"]["maximum_unsupported_run"] = 0
    report = audit_generated_backbone(
        contract=contract, backbone_coordinates=xyz, residue_names=["ALA"] * len(xyz)
    )
    assert not report["helix_support_passed"]
    assert any(x["maximum_unsupported_run"] > 0 for x in report["actual_helix_support"])


def test_v2_requires_real_backbone_and_same_ca_identity():
    contract, xyz = backbone_contract_fixture()
    kwargs = dict(
        contract=contract,
        coordinates=xyz[:, 1],
        chain_ids=[r["chain_id"] for r in contract["residues"]],
        residue_numbers=[r["residue_number"] for r in contract["residues"]],
        fixed_mask=[r["fixed"] for r in contract["residues"]],
    )
    with pytest.raises(ValueError, match="requires actual N/CA/C/O"):
        audit_scaffold_contract(**kwargs)
    other = xyz.copy()
    other[4, 1] += 0.1
    with pytest.raises(ValueError, match="differ from audited CA"):
        audit_scaffold_contract(
            **kwargs, backbone_coordinates=other, residue_names=["ALA"] * len(xyz)
        )


def test_reference_quality_cannot_be_bypassed_by_ca_only_reference():
    contract, _ = backbone_contract_fixture()
    contract["residues"][5]["reference_backbone"][3][0] += 4
    with pytest.raises(ValueError, match="required generated-backbone"):
        validate_scaffold_contract(contract)


def test_torch_and_numpy_v2_ca_residuals_match_union_support_windows():
    import torch
    from rfd3.inference.symmetry.reference_scaffold import reference_scaffold_deficits
    from rfd3.inference.symmetry.scaffold_core_guidance import (
        build_scaffold_core_topology,
    )

    contract, xyz = backbone_contract_fixture()
    fixed = np.array([r["fixed"] for r in contract["residues"]])
    features = {
        "atom_to_token_map": torch.arange(44).repeat_interleave(4),
        "asym_id": torch.arange(2).repeat_interleave(22),
        "residue_index": torch.arange(1, 23).repeat(2),
        "is_ca": torch.tensor([False, True, False, False]).repeat(44),
        "is_protein": torch.ones(44, dtype=torch.bool),
        "motif_pos": torch.tensor(xyz.reshape(-1, 3), dtype=torch.float64),
        "mosaic_scaffold_contract": contract,
        "mosaic_scaffold_ca_atom_indices": torch.arange(1, 176, 4),
        "mosaic_scaffold_backbone_atom_indices": torch.arange(176).reshape(44, 4),
    }
    topology = build_scaffold_core_topology(
        features, torch.tensor(fixed).repeat_interleave(4)
    )
    moved = xyz.copy()
    moved[~fixed] += np.random.default_rng(2).normal(0, 0.1, (sum(~fixed), 1, 3))
    residual = reference_scaffold_deficits(
        torch.tensor(moved.reshape(-1, 3)), topology.scaffold_contract
    )
    np.testing.assert_allclose(
        residual.numpy(),
        _measure(contract, moved[:, 1])["deficits_angstrom"],
        atol=1e-8,
    )

    from rfd3.inference.symmetry.reference_scaffold import reference_backbone_deficits

    original = torch.tensor(xyz.reshape(-1, 3), requires_grad=True)
    assert torch.all(
        reference_backbone_deficits(original, topology.scaffold_contract) == 0
    )
    damaged = xyz.copy()
    damaged[5, 3] += [0, 3, 0]
    variable = torch.tensor(damaged.reshape(-1, 3), requires_grad=True)
    violation = reference_backbone_deficits(variable, topology.scaffold_contract)
    assert float(violation.max().detach()) > 0
    violation.square().sum().backward()
    assert torch.isfinite(variable.grad).all()
    assert torch.linalg.vector_norm(variable.grad) > 0


def test_legacy_ca_contract_is_explicitly_marked_not_backbone_evaluated():
    contract, xyz = backbone_contract_fixture()
    contract["schema_version"] = 1
    del contract["backbone_policy"]
    for residue in contract["residues"]:
        del residue["reference_backbone"], residue["residue_name"]
    report = audit_scaffold_contract(
        contract=contract,
        coordinates=xyz[:, 1],
        chain_ids=[r["chain_id"] for r in contract["residues"]],
        residue_numbers=[r["residue_number"] for r in contract["residues"]],
        fixed_mask=[r["fixed"] for r in contract["residues"]],
    )
    assert (
        report["generated_backbone"]["status"]
        == "legacy_ca_only_contract_not_evaluated"
    )


def test_final_structure_adapter_requires_actual_backbone_after_chain_rename(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    import json
    from rfd3_mosaic.rfd3_scaffold_audit import _audit_final_generated_route_ownership
    from rfd3_mosaic.structure.pdb import AtomRecord

    contract, xyz = backbone_contract_fixture()
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"example": {"extra": {"mosaic_scaffold_contract": contract}}})
    )
    source = SimpleNamespace(
        is_protein=np.ones(44, dtype=bool),
        atom_name=np.array(["CA"] * 44),
        chain_id=np.array([r["chain_id"] for r in contract["residues"]]),
        res_id=np.array([r["residue_number"] for r in contract["residues"]]),
        is_motif_atom_with_fixed_coord=np.array(
            [r["fixed"] for r in contract["residues"]]
        ),
    )
    output = tuple(
        AtomRecord(
            "ATOM",
            4 * i + j + 1,
            atom,
            "",
            "ALA",
            "XY"[i // 22],
            i % 22 + 1,
            "",
            tuple(coordinate),
            atom[0],
        )
        for i, bb in enumerate(xyz)
        for j, (atom, coordinate) in enumerate(zip(("N", "CA", "C", "O"), bb))
    )
    result = _audit_final_generated_route_ownership(
        input_path=input_path, output_atoms=output, atom_array=source
    )
    assert result["passed"] and result["generated_backbone"]["required"]
    corrupted = list(output)
    corrupted[5 * 4 + 3] = replace(
        corrupted[5 * 4 + 3], coordinate=tuple(xyz[5, 3] + [0, 5, 0])
    )
    result = _audit_final_generated_route_ownership(
        input_path=input_path, output_atoms=tuple(corrupted), atom_array=source
    )
    assert not result["passed"]
    assert not result["generated_backbone"]["geometry_passed"]
    missing = tuple(
        a
        for a in output
        if not (a.chain_id == "X" and a.residue_number == 6 and a.atom_name == "O")
    )
    with pytest.raises(ValueError, match="Incomplete N/CA/C/O"):
        _audit_final_generated_route_ownership(
            input_path=input_path, output_atoms=missing, atom_array=source
        )
