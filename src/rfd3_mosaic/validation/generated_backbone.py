"""Required generated-backbone validation for scaffold contract version 2.

This is a heavy-backbone geometry and alpha-helix support check, not DSSP,
MolProbity, a side-chain packing score, or a folding/stability certificate.
Alpha assignment uses the DSSP electrostatic expression and consecutive
four-turn rule, without DSSP's complete secondary-structure priority/ranking.
See ``method_provenance()`` for equations, sources and engineering tolerances.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

BACKBONE_ATOMS = ("N", "CA", "C", "O")
STANDARD_AMINO_ACIDS = {
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


def default_backbone_policy() -> dict:
    """Explicit, serializable initial policy; not empirically calibrated success cutoffs."""
    return {
        "schema_version": 1,
        "bond_length_tolerance_angstrom": 0.10,
        "bond_angle_tolerance_degrees": 15.0,
        "peptide_planarity_tolerance_degrees": 20.0,
        "minimum_nonlocal_backbone_distance_angstrom": 2.0,
        "minimum_declared_helix_fraction": 0.8,
        "helix_boundary_tolerance_residues": 1,
    }


def validate_backbone_policy(policy) -> dict:
    expected = default_backbone_policy()
    if not isinstance(policy, dict) or set(policy) != set(expected):
        raise ValueError(f"backbone_policy requires exactly {sorted(expected)}")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Unsupported backbone_policy schema_version")
    out = dict(policy)
    for key in set(expected) - {"schema_version", "helix_boundary_tolerance_residues"}:
        value = out[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid positive finite backbone policy {key}")
        out[key] = float(value)
    if not 0 < out["minimum_declared_helix_fraction"] <= 1:
        raise ValueError("minimum_declared_helix_fraction must be in (0,1]")
    if any(
        out[k] >= 90
        for k in ("bond_angle_tolerance_degrees", "peptide_planarity_tolerance_degrees")
    ):
        raise ValueError("Backbone angular tolerances must be below 90 degrees")
    boundary = out["helix_boundary_tolerance_residues"]
    if type(boundary) is not int or boundary < 0:
        raise ValueError(
            "helix_boundary_tolerance_residues must be a nonnegative integer"
        )
    return out


def method_provenance() -> dict:
    return {
        "alpha_method": "dssp_like_alpha_four_turns_v1",
        "alpha_formula": "H_j=N_j+(C_(j-1)-O_(j-1))/|C_(j-1)-O_(j-1)|; E_ij=27.888*(1/r_ON+1/r_CH-1/r_OH-1/r_CN) kcal/mol; E<-0.5 for consecutive i->i+4 turns",
        "alpha_scope": "N/CA/C/O-derived alpha-helix assignment; not full DSSP; proline does not donate H; no assignment across chain breaks",
        "alpha_source": "https://github.com/PDB-REDO/dssp/blob/trunk/libdssp/src/dssp.cpp",
        "geometry_source": "https://doi.org/10.1107/S0108767391001071",
        "geometry_targets": "Engh-Huber standard backbone targets; Gly/Pro bond targets distinguished",
        "tolerance_provenance": "Explicit engineering rejection tolerances in backbone_policy, not published universal cutoffs or a calibrated biological success model",
        "collision_scope": "N/CA/C/O pairs separated by at least two residues, or on distinct chains; at least one generated residue; fixed-fixed contacts excluded; conservative heavy-atom minimum distance, not MolProbity clashscore",
        "support_scope": "Actual distinct alpha helices within the same physical chain; CA contact support, not coiled-coil or side-chain packing certification",
    }


def _angle(a, b, c):
    u, v = a - b, c - b
    denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    cosine = np.sum(u * v, axis=-1) / np.maximum(denom, 1e-12)
    return np.where(
        denom > 1e-12, np.degrees(np.arccos(np.clip(cosine, -1, 1))), np.nan
    )


def _dihedral(a, b, c, d):
    axis = c - b
    length = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.maximum(length, 1e-12)
    u, v = a - b, d - c
    u = u - np.sum(u * axis, axis=-1, keepdims=True) * axis
    v = v - np.sum(v * axis, axis=-1, keepdims=True) * axis
    valid = (
        (length[..., 0] > 1e-12)
        & (np.linalg.norm(u, axis=-1) > 1e-12)
        & (np.linalg.norm(v, axis=-1) > 1e-12)
    )
    value = np.degrees(
        np.arctan2(np.sum(np.cross(axis, u) * v, axis=-1), np.sum(u * v, axis=-1))
    )
    return np.where(valid, value, np.nan)


def _longest_false_run(supported):
    longest = current = 0
    for value in supported:
        current = 0 if value else current + 1
        longest = max(current, longest)
    return longest


def identify_alpha_helices(backbone_coordinates, chain_ids, residue_names) -> dict:
    """Assign maximal contiguous H segments from actual backbone hydrogen bonds.

    H placement and the -0.5 kcal/mol threshold follow DSSP. Only alpha
    four-turns are assigned; no inference of beta/loop identity is attempted.
    The chain input must already have consecutive residue numbering.
    """
    xyz = np.asarray(backbone_coordinates, dtype=float)
    chains = np.asarray(chain_ids, dtype=str)
    names = np.asarray(residue_names, dtype=str)
    if (
        xyz.shape != (len(chains), 4, 3)
        or len(names) != len(chains)
        or not np.isfinite(xyz).all()
    ):
        raise ValueError(
            "Alpha assignment requires finite [residue,N/CA/C/O,XYZ] coordinates and names"
        )
    n = len(xyz)
    mask = np.zeros(n, dtype=bool)
    # C--N continuity is a prerequisite for a four-turn, not a substitute for
    # the stricter generated peptide checks in the required geometry audit.
    connected = (chains[:-1] == chains[1:]) & (
        np.linalg.norm(xyz[:-1, 2] - xyz[1:, 0], axis=-1) < 2.0
    )
    turn = np.zeros(n, dtype=bool)
    energies = []
    for i in range(n - 4):
        j = i + 4
        if names[j] == "PRO" or not connected[i:j].all():
            continue
        co = xyz[j - 1, 2] - xyz[j - 1, 3]
        length = np.linalg.norm(co)
        if length <= 1e-12:
            continue
        h, donor, carbon, oxygen = (
            xyz[j, 0] + co / length,
            xyz[j, 0],
            xyz[i, 2],
            xyz[i, 3],
        )
        ds = np.asarray(
            [
                np.linalg.norm(oxygen - donor),
                np.linalg.norm(carbon - h),
                np.linalg.norm(oxygen - h),
                np.linalg.norm(carbon - donor),
            ]
        )
        if np.any(ds < 0.5):
            continue  # Degenerate atoms are invalid geometry, never evidence of H.
        energy = 27.888 * (1 / ds[0] + 1 / ds[1] - 1 / ds[2] - 1 / ds[3])
        turn[i] = energy < -0.5
        if turn[i]:
            energies.append(
                {
                    "acceptor_index": i,
                    "donor_index": j,
                    "energy_kcal_mol": float(energy),
                }
            )
    for i in range(1, n - 4):
        if turn[i - 1] and turn[i]:
            mask[i : i + 4] = True
    segments = []
    for i in range(n):
        if not mask[i]:
            continue
        if not segments or segments[-1][-1] != i - 1 or chains[i] != chains[i - 1]:
            segments.append([])
        segments[-1].append(i)
    return {
        "helix_mask": mask,
        "segments": segments,
        "alpha_hydrogen_bonds": energies,
        "method": "dssp_like_alpha_four_turns_v1",
    }


def backbone_from_atoms(atoms, contract, *, chain_mapping=None):
    """Resolve every N/CA/C/O against contract identities; never guess missing atoms.

    ``chain_mapping`` maps contract chain IDs to output IDs when a native
    writer has renamed chains; callers must validate that mapping separately.
    """
    lookup = {}
    for atom in atoms:
        if atom.record_type != "ATOM" or atom.atom_name not in BACKBONE_ATOMS:
            continue
        key = (atom.chain_id, atom.residue_number, atom.insertion_code, atom.atom_name)
        if key in lookup:
            raise ValueError(f"Duplicate backbone atom {key}")
        lookup[key] = atom
    coordinates, names = [], []
    for residue in contract["residues"]:
        chain = residue["chain_id"]
        if chain_mapping is not None:
            chain = chain_mapping[chain]
        keys = [(chain, residue["residue_number"], "", name) for name in BACKBONE_ATOMS]
        if any(key not in lookup for key in keys):
            raise ValueError(
                f"Incomplete N/CA/C/O backbone at {chain}{residue['residue_number']}"
            )
        records = [lookup[key] for key in keys]
        if len({a.residue_name for a in records}) != 1:
            raise ValueError("Conflicting backbone residue names")
        coordinates.append([a.coordinate for a in records])
        names.append(records[0].residue_name)
    return np.asarray(coordinates, dtype=float), names


def audit_generated_backbone(
    *, contract, backbone_coordinates, residue_names
) -> dict[str, Any]:
    """Shared reference/final required check for version-2 complete scaffolds.

    The caller validates the contract schema before this function; no CA
    alignment is required because every check here is rigid-frame invariant.
    """
    policy = validate_backbone_policy(contract["backbone_policy"])
    xyz = np.asarray(backbone_coordinates, dtype=float)
    names = np.asarray(residue_names, dtype=str)
    residues = contract["residues"]
    n = len(residues)
    if xyz.shape != (n, 4, 3) or names.shape != (n,) or not np.isfinite(xyz).all():
        raise ValueError(
            "Backbone audit requires complete finite N/CA/C/O coordinates in contract order"
        )
    if any(name not in STANDARD_AMINO_ACIDS for name in names):
        raise ValueError(
            "Generated-backbone audit currently supports standard amino acids only"
        )
    fixed = np.asarray([r["fixed"] for r in residues], dtype=bool)
    generated = ~fixed
    chains = np.asarray([r["chain_id"] for r in residues])
    geometry_failures = []
    measured = {}

    def check(label, actual, target, relevant, tolerance):
        actual, target = np.broadcast_arrays(np.asarray(actual), np.asarray(target))
        error = np.abs(actual - target)
        relevant = np.asarray(relevant, dtype=bool)
        bad = relevant & (~np.isfinite(error) | (error > tolerance))
        measured[label] = {
            "checked": int(relevant.sum()),
            "failures": int(bad.sum()),
            "maximum_error": float(error[relevant].max(initial=0))
            if np.isfinite(error[relevant]).all()
            else None,
            "tolerance": float(tolerance),
        }
        for index in np.flatnonzero(bad):
            geometry_failures.append(
                {
                    "kind": label,
                    "residue_index": int(index),
                    "observed": float(actual[index])
                    if np.isfinite(actual[index])
                    else None,
                    "target": float(target[index]),
                }
            )

    bt = policy["bond_length_tolerance_angstrom"]
    at = policy["bond_angle_tolerance_degrees"]
    pt = policy["peptide_planarity_tolerance_degrees"]
    nca_target = np.where(names == "GLY", 1.451, np.where(names == "PRO", 1.466, 1.458))
    cac_target = np.where(names == "GLY", 1.516, 1.525)
    check(
        "N_CA_length",
        np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=-1),
        nca_target,
        generated,
        bt,
    )
    check(
        "CA_C_length",
        np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=-1),
        cac_target,
        generated,
        bt,
    )
    check(
        "C_O_length",
        np.linalg.norm(xyz[:, 2] - xyz[:, 3], axis=-1),
        1.231,
        generated,
        bt,
    )
    check("N_CA_C_angle", _angle(xyz[:, 0], xyz[:, 1], xyz[:, 2]), 111.2, generated, at)
    check("CA_C_O_angle", _angle(xyz[:, 1], xyz[:, 2], xyz[:, 3]), 120.8, generated, at)
    adjacent = chains[:-1] == chains[1:]
    relevant = adjacent & (generated[:-1] | generated[1:])
    cn_target = np.where(names[1:] == "PRO", 1.341, 1.329)
    check(
        "C_N_peptide_length",
        np.linalg.norm(xyz[:-1, 2] - xyz[1:, 0], axis=-1),
        cn_target,
        relevant,
        bt,
    )
    check(
        "CA_C_N_angle",
        _angle(xyz[:-1, 1], xyz[:-1, 2], xyz[1:, 0]),
        116.2,
        relevant,
        at,
    )
    check(
        "C_N_CA_angle", _angle(xyz[:-1, 2], xyz[1:, 0], xyz[1:, 1]), 121.7, relevant, at
    )
    check(
        "O_C_N_angle", _angle(xyz[:-1, 3], xyz[:-1, 2], xyz[1:, 0]), 123.0, relevant, at
    )
    omega = np.abs(_dihedral(xyz[:-1, 1], xyz[:-1, 2], xyz[1:, 0], xyz[1:, 1]))
    # Both cis and trans are planar; do not ban genuine cis-proline/non-Pro
    # conformations. Their presence is explicit in the report.
    check(
        "peptide_omega_planarity",
        np.minimum(omega, np.abs(180 - omega)),
        0,
        relevant,
        pt,
    )
    carbonyl = np.abs(_dihedral(xyz[:-1, 1], xyz[:-1, 2], xyz[1:, 0], xyz[:-1, 3]))
    check(
        "carbonyl_peptide_planarity",
        np.minimum(carbonyl, np.abs(180 - carbonyl)),
        0,
        relevant,
        pt,
    )

    clashes = []
    clash_count = 0
    minimum = float("inf")
    cutoff = policy["minimum_nonlocal_backbone_distance_angstrom"]
    # Per-residue vectorization bounds memory while checking complete atoms.
    for i in range(n):
        js = np.arange(i + 1, n)
        valid = (generated[i] | generated[js]) & (
            (chains[i] != chains[js]) | (js - i > 1)
        )
        js = js[valid]
        if not len(js):
            continue
        distance = np.linalg.norm(xyz[i, None, :, None] - xyz[js, None, :], axis=-1)
        minimum = min(minimum, float(distance.min()))
        bad_pairs = np.argwhere(distance < cutoff)
        clash_count += len(bad_pairs)
        for j, a, b in bad_pairs[: max(0, 200 - len(clashes))]:
            clashes.append(
                {
                    "left_residue_index": i,
                    "right_residue_index": int(js[j]),
                    "left_atom": BACKBONE_ATOMS[a],
                    "right_atom": BACKBONE_ATOMS[b],
                    "distance_angstrom": float(distance[j, a, b]),
                }
            )

    helices = identify_alpha_helices(xyz, chains, names)
    hmask = helices["helix_mask"]
    segments = helices["segments"]
    segment_id = np.full(n, -1, dtype=int)
    for index, ids in enumerate(segments):
        segment_id[ids] = index
    block_assignment, block_reports, failures = {}, [], []
    covered = np.zeros(n, dtype=bool)
    owner = {}
    slack = policy["helix_boundary_tolerance_residues"]
    for block in contract["helix_blocks"]:
        ids = np.asarray(block["residue_indices"], dtype=int)
        present = np.unique(segment_id[ids][segment_id[ids] >= 0])
        fraction = float(hmask[ids].mean())
        valid = (
            len(present) == 1
            and len(ids) >= 4
            and fraction >= policy["minimum_declared_helix_fraction"]
        )
        actual = int(present[0]) if len(present) == 1 else None
        if actual is not None:
            bounds = segments[actual]
            valid = (
                valid and ids[0] >= bounds[0] - slack and ids[-1] <= bounds[-1] + slack
            )
            if actual in owner:
                valid = False
                failures.append(
                    {
                        "kind": "same_actual_helix_split_into_blocks",
                        "blocks": [owner[actual], block["id"]],
                        "actual_helix_index": actual,
                    }
                )
            owner[actual] = block["id"]
        if not valid:
            failures.append(
                {"kind": "declared_block_not_one_actual_helix", "block": block["id"]}
            )
        else:
            block_assignment[block["id"]] = actual
            # Apply the declared boundary tolerance symmetrically: a cap may
            # lose H character, or actual H may extend just beyond the drawn
            # block. Only that same maximal H segment is covered; a new helix
            # elsewhere, or a long omitted arm, cannot inherit coverage.
            actual_ids = np.asarray(segments[actual], dtype=int)
            covered[
                actual_ids[
                    (actual_ids >= ids[0] - slack) & (actual_ids <= ids[-1] + slack)
                ]
            ] = True
        block_reports.append(
            {
                "block": block["id"],
                "actual_helix_index": actual,
                "actual_alpha_fraction": fraction,
                "passed": bool(valid),
            }
        )
    missing = np.flatnonzero(generated & hmask & ~covered).tolist()
    if missing:
        failures.append(
            {"kind": "undeclared_generated_alpha_residues", "residue_indices": missing}
        )
    if not any(
        np.any(generated[b["residue_indices"]]) for b in contract["helix_blocks"]
    ):
        failures.append({"kind": "no_generated_helix_block"})
    # Every edge must join different maximal actual helices, not two windows
    # cut from the same long helix. Support is measured over ACTUAL helices.
    neighbours = {i: set() for i in range(len(segments))}
    for edge in contract["support_edges"]:
        a, b = block_assignment.get(edge["left"]), block_assignment.get(edge["right"])
        if (
            a is None
            or b is None
            or a == b
            or chains[segments[a][0]] != chains[segments[b][0]]
        ):
            failures.append(
                {"kind": "edge_not_distinct_same_chain_actual_helices", **edge}
            )
            continue
        neighbours[a].add(b)
        neighbours[b].add(a)
    support_reports = []
    distance = contract["limits"]["contact_distance"]
    fraction = contract["limits"]["minimum_helix_contact_fraction"]
    maximum_run = contract["limits"]["maximum_unsupported_run"]
    for index, ids_list in enumerate(segments):
        ids = np.asarray(ids_list)
        if not generated[ids].any():
            continue
        partners = sorted(j for other in neighbours[index] for j in segments[other])
        nearest = (
            np.linalg.norm(xyz[ids, None, 1] - xyz[partners, 1], axis=-1).min(axis=1)
            if partners
            else np.full(len(ids), np.inf)
        )
        supported = nearest <= distance + contract["limits"]["geometry_tolerance"]
        # Inspect the complete actual helix, not just its declared subset.
        # The required fraction concerns its GENERATED residues: well-packed
        # immutable residues cannot mask an unsupported generated extension.
        longest = _longest_false_run(supported)
        generated_fraction = float(supported[generated[ids]].mean())
        generated_longest = _longest_false_run(supported | ~generated[ids])
        passed = bool(
            generated_fraction >= fraction and generated_longest <= maximum_run
        )
        if not passed:
            failures.append(
                {"kind": "actual_generated_helix_support", "actual_helix_index": index}
            )
        support_reports.append(
            {
                "actual_helix_index": index,
                "residue_indices": ids_list,
                "generated_residue_count": int(generated[ids].sum()),
                "partner_helix_indices": sorted(neighbours[index]),
                "supported_fraction": float(supported.mean()),
                "maximum_unsupported_run": longest,
                "generated_supported_fraction": generated_fraction,
                "maximum_generated_unsupported_run": generated_longest,
                "passed": passed,
            }
        )
    return {
        "measurement": "required_generated_backbone_and_actual_alpha_support",
        "schema_version": 1,
        "required": True,
        "passed": not geometry_failures and not clash_count and not failures,
        "geometry_passed": not geometry_failures and not clash_count,
        "helix_support_passed": not failures,
        "policy": policy,
        "method": method_provenance(),
        "geometry_checks": measured,
        "geometry_failure_count": len(geometry_failures),
        "geometry_failures": geometry_failures,
        "nonlocal_backbone_clash_count": clash_count,
        "nonlocal_backbone_clashes": clashes,
        "nonlocal_backbone_clashes_truncated": clash_count > len(clashes),
        "minimum_nonlocal_backbone_distance_angstrom": minimum
        if math.isfinite(minimum)
        else None,
        "generated_peptide_cis_count": int(((omega < 90) & relevant).sum()),
        "actual_alpha_segments": segments,
        "actual_generated_alpha_residue_count": int((hmask & generated).sum()),
        "uncovered_generated_alpha_residues": missing,
        "declared_block_checks": block_reports,
        "actual_helix_support": support_reports,
        "helix_failures": failures,
        "guarantee_scope": "Declared backbone geometry and same-chain distinct actual alpha-helix CA support; not full DSSP, side-chain packing, coiled-coil, folding or biological success certification",
    }
