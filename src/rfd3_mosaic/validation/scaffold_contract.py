"""Explicit full-scaffold geometry contract; no learned success thresholds.

CA geometry and declared block contacts are proxies, not an all-atom folding
certificate. Reference displacements additionally bound interchain segment
separation during the common linear reference-to-candidate deformation.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


LIMITS = {
    "maximum_ca_deviation",
    "contact_distance",
    "minimum_helix_contact_fraction",
    "minimum_interchain_segment_distance",
    "minimum_ca_bond_distance",
    "maximum_ca_bond_distance",
    "fixed_ca_tolerance",
    "geometry_tolerance",
}


def _keys(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(required) - set(optional)
        or set(required) - set(value)
    ):
        raise ValueError(
            f"Scaffold contract requires keys {sorted(required)}; optional {sorted(optional)}"
        )


def _segment_distances(a, b, c, d):
    """All distances between finite segments a--b and c--d, in NumPy."""

    def point_segment(p, start, end):
        v = end - start
        t = np.clip(
            np.sum((p[:, None] - start) * v, axis=-1)
            / np.maximum(np.sum(v * v, axis=-1), 1e-12),
            0,
            1,
        )
        return np.linalg.norm(p[:, None] - start - t[..., None] * v, axis=-1)

    u, v, w = b - a, d - c, a[:, None] - c
    aa, bb, cc = np.sum(u * u, axis=-1)[:, None], u @ v.T, np.sum(v * v, axis=-1)[None]
    dd, ee = np.sum(u[:, None] * w, axis=-1), np.sum(v[None] * w, axis=-1)
    det = aa * cc - bb * bb
    s, t = (
        (bb * ee - cc * dd) / np.maximum(det, 1e-12),
        (aa * ee - bb * dd) / np.maximum(det, 1e-12),
    )
    interior = np.linalg.norm(
        a[:, None] + s[..., None] * u[:, None] - c - t[..., None] * v, axis=-1
    )
    interior = np.where(
        (det > 1e-12) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), interior, np.inf
    )
    return np.minimum.reduce(
        [
            point_segment(a, c, d),
            point_segment(b, c, d),
            point_segment(c, a, b).T,
            point_segment(d, a, b).T,
            interior,
        ]
    )


def contract_arrays(contract):
    records = contract["residues"]
    xyz = np.asarray([r["reference_ca"] for r in records], dtype=float)
    fixed = np.asarray([r["fixed"] for r in records], dtype=bool)
    chain_names = list(dict.fromkeys(r["chain_id"] for r in records))
    chains = [
        np.asarray(
            [i for i, r in enumerate(records) if r["chain_id"] == name], dtype=int
        )
        for name in chain_names
    ]
    blocks = {
        b["id"]: np.asarray(b["residue_indices"], dtype=int)
        for b in contract["helix_blocks"]
    }
    return xyz, fixed, chains, blocks


def _measure(contract, xyz):
    reference, fixed, chains, blocks = contract_arrays(contract)
    limits = contract["limits"]
    tolerance = limits["geometry_tolerance"]
    deviation = np.linalg.norm(xyz - reference, axis=-1)
    values, labels = [], []

    def add(name, value):
        values.extend(
            np.maximum(0, np.asarray(value, dtype=float)).reshape(-1).tolist()
        )
        labels.extend([name] * np.asarray(value).size)

    add("reference_ca_deviation", deviation[~fixed] - limits["maximum_ca_deviation"])
    add("fixed_ca_deviation", deviation[fixed] - limits["fixed_ca_tolerance"])
    block_reports = []
    for edge in contract["support_edges"]:
        for left, right in (
            (edge["left"], edge["right"]),
            (edge["right"], edge["left"]),
        ):
            ids, partners = blocks[left], blocks[right]
            nearest = np.linalg.norm(xyz[ids, None] - xyz[partners], axis=-1).min(
                axis=1
            )
            k = math.ceil(limits["minimum_helix_contact_fraction"] * len(ids))
            add("block_contact", np.sort(nearest)[:k] - limits["contact_distance"])
            supported = nearest <= limits["contact_distance"] + tolerance
            longest = current = 0
            for present in supported:
                current = 0 if present else current + 1
                longest = max(longest, current)
            if "maximum_unsupported_run" in limits and contract["schema_version"] == 1:
                width = limits["maximum_unsupported_run"] + 1
                add(
                    "unsupported_window",
                    [
                        nearest[i : i + width].min() - limits["contact_distance"]
                        for i in range(len(ids) - width + 1)
                    ],
                )
            block_reports.append(
                {
                    "block": left,
                    "partner": right,
                    "supported_fraction": float(supported.mean()),
                    "required_supported_residues": k,
                    "maximum_unsupported_run": longest,
                }
            )
    if contract["schema_version"] == 2:
        # A residue is supported by any declared distinct partner helix; do
        # not demand every individual edge support its entire length.
        neighbours = {name: set() for name in blocks}
        for edge in contract["support_edges"]:
            neighbours[edge["left"]].add(edge["right"])
            neighbours[edge["right"]].add(edge["left"])
        width = limits["maximum_unsupported_run"] + 1
        for name, ids in blocks.items():
            if not np.any(~fixed[ids]) or not neighbours[name]:
                continue
            partners = np.concatenate(
                [blocks[other] for other in sorted(neighbours[name])]
            )
            nearest = np.linalg.norm(xyz[ids, None] - xyz[partners], axis=-1).min(
                axis=1
            )
            add(
                "unsupported_window",
                [
                    nearest[i : i + width].min() - limits["contact_distance"]
                    for i in range(len(ids) - width + 1)
                ],
            )
    separations = []
    for i, ids in enumerate(chains):
        bonds = np.linalg.norm(xyz[ids[1:]] - xyz[ids[:-1]], axis=-1)
        add("ca_bond_lower", limits["minimum_ca_bond_distance"] - bonds)
        add("ca_bond_upper", bonds - limits["maximum_ca_bond_distance"])
        for j in range(i + 1, len(chains)):
            other = chains[j]
            delta = float(
                _segment_distances(
                    reference[ids[:-1]],
                    reference[ids[1:]],
                    reference[other[:-1]],
                    reference[other[1:]],
                ).min()
            )
            lower = delta - float(deviation[ids].max()) - float(deviation[other].max())
            actual = float(
                _segment_distances(
                    xyz[ids[:-1]], xyz[ids[1:]], xyz[other[:-1]], xyz[other[1:]]
                ).min()
            )
            add(
                "reference_deformation_clearance",
                limits["minimum_interchain_segment_distance"] - lower,
            )
            add(
                "interchain_segment_clearance",
                limits["minimum_interchain_segment_distance"] - actual,
            )
            separations.append(
                {
                    "chain_indices": [i, j],
                    "reference_minimum_angstrom": delta,
                    "linear_deformation_lower_bound_angstrom": lower,
                    "actual_minimum_angstrom": actual,
                }
            )
    deficits = np.asarray(values, dtype=float)
    return {
        "passed": bool(np.all(deficits <= tolerance)),
        "deficits_angstrom": deficits.tolist(),
        "deficit_categories": labels,
        "maximum_excess_angstrom": float(deficits.max(initial=0)),
        "squared_excess_angstrom2": float(deficits @ deficits),
        "violated_constraint_count": int((deficits > tolerance).sum()),
        "maximum_ca_deviation_angstrom": float(deviation.max(initial=0)),
        "block_contacts": block_reports,
        "interchain_separations": separations,
        "guarantee_scope": "CA interchain segment separation during the common linear reference-to-output deformation; not self-knot, all-atom, secondary-structure, packing or folding certification",
    }


def validate_scaffold_contract(contract: dict) -> dict:
    """Validate schema AND the reference's declared geometry; fail closed."""
    version = contract.get("schema_version") if isinstance(contract, dict) else None
    _keys(
        contract,
        {"schema_version", "residues", "helix_blocks", "support_edges", "limits"}
        | ({"backbone_policy"} if version == 2 else set()),
    )
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported scaffold contract schema_version")
    out = copy.deepcopy(contract)
    if version == 2:
        from .generated_backbone import validate_backbone_policy

        out["backbone_policy"] = validate_backbone_policy(out["backbone_policy"])
    _keys(out["limits"], LIMITS, {"maximum_unsupported_run"})
    limits = out["limits"]
    for key in LIMITS:
        value = limits[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Scaffold limit {key} must be explicit finite positive")
        limits[key] = float(value)
    if (
        limits["minimum_helix_contact_fraction"] > 1
        or limits["minimum_ca_bond_distance"] >= limits["maximum_ca_bond_distance"]
    ):
        raise ValueError("Invalid scaffold contact fraction or CA bond interval")
    if limits["geometry_tolerance"] >= limits["minimum_interchain_segment_distance"]:
        raise ValueError(
            "Scaffold geometry tolerance must be below positive segment clearance"
        )
    if "maximum_unsupported_run" in limits and (
        type(limits["maximum_unsupported_run"]) is not int
        or limits["maximum_unsupported_run"] < 0
    ):
        raise ValueError("maximum_unsupported_run must be a nonnegative integer")
    if version == 2 and "maximum_unsupported_run" not in limits:
        raise ValueError("Version-2 scaffold requires explicit maximum_unsupported_run")
    if not isinstance(out["residues"], list) or len(out["residues"]) < 4:
        raise ValueError("Scaffold contract requires a complete CA reference")
    identities = []
    for r in out["residues"]:
        _keys(
            r,
            {"chain_id", "residue_number", "fixed", "reference_ca"}
            | ({"reference_backbone", "residue_name"} if version == 2 else set()),
        )
        if (
            not isinstance(r["chain_id"], str)
            or not r["chain_id"]
            or type(r["residue_number"]) is not int
            or type(r["fixed"]) is not bool
        ):
            raise ValueError("Invalid scaffold residue identity/fixed flag")
        xyz = np.asarray(r["reference_ca"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Scaffold reference CA must be finite XYZ")
        r["reference_ca"] = xyz.tolist()
        if version == 2:
            bb = np.asarray(r["reference_backbone"], dtype=float)
            if (
                bb.shape != (4, 3)
                or not np.isfinite(bb).all()
                or not np.allclose(bb[1], xyz, atol=1e-8, rtol=0)
                or not isinstance(r["residue_name"], str)
                or r["residue_name"]
                not in {
                    "ALA",
                    "ARG",
                    "ASN",
                    "ASP",
                    "CYS",
                    "GLN",
                    "GLU",
                    "GLY",
                    "HIS",
                    "ILE",
                    "LEU",
                    "LYS",
                    "MET",
                    "PHE",
                    "PRO",
                    "SER",
                    "THR",
                    "TRP",
                    "TYR",
                    "VAL",
                }
            ):
                raise ValueError(
                    "Version-2 scaffold requires standard residue names and finite reference N/CA/C/O matching reference_ca"
                )
            r["reference_backbone"] = bb.tolist()
        identities.append((r["chain_id"], r["residue_number"]))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate scaffold residue identities")
    reference, fixed, chains, _ = contract_arrays({**out, "helix_blocks": []})
    if (
        not fixed.any()
        or fixed.all()
        or fixed.sum() < 3
        or np.linalg.matrix_rank(
            reference[fixed] - reference[fixed].mean(axis=0), tol=1e-6
        )
        < 2
    ):
        raise ValueError(
            "Scaffold requires generated residues and at least three noncollinear fixed CAs"
        )
    for ids in chains:
        if (
            len(ids) < 2
            or np.any(np.diff(ids) != 1)
            or any(
                identities[b][1] - identities[a][1] != 1
                for a, b in zip(ids[:-1], ids[1:])
            )
        ):
            raise ValueError(
                "Scaffold requires contiguous ordered complete physical chains"
            )
    if not isinstance(out["helix_blocks"], list) or not out["helix_blocks"]:
        raise ValueError("Scaffold contract requires declared helix blocks")
    seen = set()
    used = set()
    blocks = {}
    for block in out["helix_blocks"]:
        _keys(block, {"id", "residue_indices"})
        name, ids = block["id"], block["residue_indices"]
        if (
            not isinstance(name, str)
            or not name
            or name in seen
            or not isinstance(ids, list)
            or len(ids) < 2
            or any(type(i) is not int or not 0 <= i < len(reference) for i in ids)
        ):
            raise ValueError("Invalid scaffold helix block identity/indices")
        if (
            any(b - a != 1 for a, b in zip(ids[:-1], ids[1:]))
            or len({identities[i][0] for i in ids}) != 1
            or used.intersection(ids)
        ):
            raise ValueError(
                "Scaffold blocks must be contiguous, disjoint, and within one chain"
            )
        seen.add(name)
        used.update(ids)
        blocks[name] = ids
    if not isinstance(out["support_edges"], list) or not out["support_edges"]:
        raise ValueError("Scaffold contract requires explicit block support edges")
    edges = set()
    supported = set()
    for edge in out["support_edges"]:
        _keys(edge, {"left", "right"})
        a, b = edge["left"], edge["right"]
        if (
            a not in blocks
            or b not in blocks
            or a == b
            or tuple(sorted((a, b))) in edges
        ):
            raise ValueError("Invalid or duplicate scaffold support edge")
        if identities[blocks[a][0]][0] != identities[blocks[b][0]][0]:
            raise ValueError(
                "Scaffold helix support must be within the same physical chain"
            )
        if min(abs(i - j) for i in blocks[a] for j in blocks[b]) < 2:
            raise ValueError(
                "Scaffold support blocks must not count consecutive local neighbours"
            )
        edges.add(tuple(sorted((a, b))))
        supported.update((a, b))
    if any(
        np.any(~fixed[ids]) and name not in supported for name, ids in blocks.items()
    ):
        raise ValueError("Every generated helix block requires a support edge")
    if version == 2 and not any(np.any(~fixed[ids]) for ids in blocks.values()):
        raise ValueError(
            "Version-2 scaffold requires an actual generated helix block; fixed-only blocks are insufficient"
        )
    initial = _measure(out, reference)
    if not initial["passed"]:
        bad = sorted(
            {
                label
                for label, value in zip(
                    initial["deficit_categories"], initial["deficits_angstrom"]
                )
                if value > limits["geometry_tolerance"]
            }
        )
        raise ValueError(f"Scaffold reference violates its own contract: {bad}")
    if version == 2:
        from .generated_backbone import audit_generated_backbone

        backbone = audit_generated_backbone(
            contract=out,
            backbone_coordinates=[r["reference_backbone"] for r in out["residues"]],
            residue_names=[r["residue_name"] for r in out["residues"]],
        )
        if not backbone["passed"]:
            kinds = sorted(
                {
                    f["kind"]
                    for f in backbone["geometry_failures"] + backbone["helix_failures"]
                }
            )
            if backbone["nonlocal_backbone_clash_count"]:
                kinds.append("nonlocal_backbone_clash")
            raise ValueError(
                f"Scaffold reference violates required generated-backbone contract: {kinds}"
            )
    return out


def audit_scaffold_contract(
    *,
    contract,
    coordinates,
    chain_ids,
    residue_numbers,
    fixed_mask,
    align_fixed=False,
    backbone_coordinates=None,
    residue_names=None,
) -> dict[str, Any]:
    """Independent final CA audit with optional ONE global fixed-CA alignment."""
    validated = validate_scaffold_contract(contract)
    reference, fixed, _, _ = contract_arrays(validated)
    xyz = np.asarray(coordinates, dtype=float)
    expected = [(r["chain_id"], r["residue_number"]) for r in validated["residues"]]
    if (
        xyz.shape != reference.shape
        or not np.isfinite(xyz).all()
        or list(zip(chain_ids, residue_numbers)) != expected
        or not np.array_equal(np.asarray(fixed_mask, dtype=bool), fixed)
    ):
        raise ValueError(
            "Scaffold audit CA identities, counts, coordinates or fixed mask mismatch"
        )
    if align_fixed:
        a, b = reference[fixed].mean(axis=0), xyz[fixed].mean(axis=0)
        u, _, vt = np.linalg.svd((reference[fixed] - a).T @ (xyz[fixed] - b))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        # Bring output into the reference frame with one proper rigid transform.
        xyz = (xyz - b) @ rotation.T + a
    report = _measure(validated, xyz)
    if validated["schema_version"] == 2:
        from .generated_backbone import audit_generated_backbone

        if backbone_coordinates is None or residue_names is None:
            raise ValueError(
                "Version-2 scaffold audit requires actual N/CA/C/O coordinates and residue names"
            )
        raw_ca = np.asarray(coordinates, dtype=float)
        bb = np.asarray(backbone_coordinates, dtype=float)
        if bb.shape != (len(raw_ca), 4, 3) or not np.allclose(
            bb[:, 1], raw_ca, atol=1e-8, rtol=0
        ):
            raise ValueError(
                "Scaffold audit backbone CA coordinates differ from audited CA coordinates"
            )
        backbone = audit_generated_backbone(
            contract=validated,
            backbone_coordinates=bb,
            residue_names=residue_names,
        )
        report["generated_backbone"] = backbone
        report["passed"] = bool(report["passed"] and backbone["passed"])
    else:
        report["generated_backbone"] = {
            "required": False,
            "evaluated": False,
            "status": "legacy_ca_only_contract_not_evaluated",
        }
    report.update(
        declared=True,
        applicable=True,
        measurement="explicit_full_scaffold_contract",
        alignment="one_global_fixed_ca_proper_rotation"
        if align_fixed
        else "declared_common_frame",
        thresholds=validated["limits"],
        contract_schema_version=validated["schema_version"],
    )
    return report
