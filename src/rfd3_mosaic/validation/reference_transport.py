"""Deterministic, frame-equivariant transport of a complete scaffold prior.

Rigid seed motions are interpolated in SE(3), not in XYZ. This is a proposal
construction rule, not a feasibility theorem: every transported reference is
revalidated and each accepted reference transition has a conservative CA
segment-separation certificate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from .scaffold_contract import (
    _segment_distances,
    contract_arrays,
    validate_scaffold_contract,
)

METHOD = "seed_se3_screw_arclength_v1"


def transport_fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def proper_transform(value, *, tolerance=1e-6):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=tolerance, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=tolerance, rtol=0
        )
        or abs(np.linalg.det(matrix[:3, :3]) - 1) > tolerance
    ):
        raise ValueError("Reference transport requires a finite proper SE(3) transform")
    return matrix


def interpolate_se3(left, right, fraction):
    """T(u)=T_left exp(u log(T_left^-1 T_right)), principal screw branch.

    The pi branch is ambiguous and is rejected rather than selecting a
    coordinate-axis-dependent rotation. Bounded local proposals normally
    remain well inside this branch.
    """
    a, b = proper_transform(left), proper_transform(right)
    u = float(fraction)
    if not 0 <= u <= 1:
        raise ValueError("SE(3) interpolation fraction must lie in [0, 1]")
    if u == 0:
        return a.copy()
    if u == 1:
        return b.copy()
    relative = np.linalg.inv(a) @ b
    rotation, translation = relative[:3, :3], relative[:3, 3]
    theta = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    if theta >= math.pi - 1e-6:
        raise ValueError(
            "Reference transport relative rotation reaches the ambiguous pi branch"
        )
    omega = (rotation - rotation.T) * (
        0.5 if theta < 1e-7 else theta / (2 * math.sin(theta))
    )

    def exponential(scale):
        w = scale * omega
        angle = scale * theta
        if angle < 1e-6:
            r = np.eye(3) + w + w @ w / 2
            v = np.eye(3) + w / 2 + w @ w / 6
        else:
            r = (
                np.eye(3)
                + math.sin(angle) / angle * w
                + (1 - math.cos(angle)) / angle**2 * (w @ w)
            )
            v = (
                np.eye(3)
                + (1 - math.cos(angle)) / angle**2 * w
                + (angle - math.sin(angle)) / angle**3 * (w @ w)
            )
        return r, v

    _, full_v = exponential(1.0)
    velocity = np.linalg.solve(full_v, translation)
    r, v = exponential(u)
    increment = np.eye(4)
    increment[:3, :3] = r
    increment[:3, 3] = v @ (u * velocity)
    return proper_transform(a @ increment)


def build_reference_transport_plan(
    contract,
    groups,
    orbits,
    registry_transform_order,
    registry_transform_matrices,
    fixed_atom_records,
):
    """Bind ownership to compiler-declared physical joint seeds, never proximity.

    fixed_atom_records contain chain_id, residue_number, atom_name, coordinate
    for exactly the selected fixed atoms of the complete preexpanded input.
    """
    contract = validate_scaffold_contract(contract)
    if contract["schema_version"] != 2:
        raise ValueError(
            "Mobile full-scaffold transport requires the complete backbone contract v2"
        )
    records = copy.deepcopy(list(fixed_atom_records))
    keys = [(r["chain_id"], r["residue_number"], r["atom_name"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Reference transport repeats a fixed atom identity")
    xyz = np.asarray([r["coordinate"] for r in records], dtype=float)
    if xyz.shape != (len(records), 3) or not np.isfinite(xyz).all():
        raise ValueError("Reference transport fixed coordinates must be finite")
    ca_lookup = {
        (r["chain_id"], r["residue_number"]): i
        for i, r in enumerate(contract["residues"])
    }
    definitions = {str(group["group_id"]): group for group in groups}
    if len(definitions) != len(groups):
        raise ValueError("Reference transport repeats a joint seed group")
    bound_groups = {}
    atom_owners = {}
    residue_owners = {}
    matrices = [
        proper_transform(registry_transform_matrices[key]).tolist()
        for key in registry_transform_order
    ]
    # The compiler's cyclic generator identifies the primary axis, including
    # D2 where the three twofold axes cannot be ranked by angle alone.
    axis = None
    if "r1" in registry_transform_matrices:
        generator = proper_transform(registry_transform_matrices["r1"])
        _, singular, vh = np.linalg.svd(np.eye(3) - generator[:3, :3])
        if singular[1] > 1e-8:
            point = np.linalg.lstsq(
                np.eye(3) - generator[:3, :3], generator[:3, 3], rcond=None
            )[0]
            axis = {"point": point.tolist(), "direction": vh[-1].tolist()}
    for orbit in orbits:
        ids = orbit["group_ids"]
        transforms = orbit["group_transform_ids"]
        if len(ids) != len(transforms) or len(ids) != len(matrices):
            raise ValueError(
                "Reference transport supports only full regular seed orbits"
            )
        for group_id, transform_index in zip(ids, transforms, strict=True):
            definition = definitions[group_id]
            components = {
                value
                for member in definition["members"]
                for value in member["src_components"]
            }
            atom_ids = [
                i for i, key in enumerate(keys) if f"{key[0]}{key[1]}" in components
            ]
            if (
                len(atom_ids) < 3
                or np.linalg.matrix_rank(
                    xyz[atom_ids] - xyz[atom_ids].mean(axis=0), tol=1e-6
                )
                < 2
            ):
                raise ValueError(
                    "A transported joint seed needs three noncollinear selected fixed atoms"
                )
            ca_ids = []
            for i in atom_ids:
                if i in atom_owners:
                    raise ValueError("Reference transport seed groups overlap")
                atom_owners[i] = group_id
                key = keys[i]
                if key[2] == "CA":
                    residue_index = ca_lookup[key[:2]]
                    if not contract["residues"][residue_index]["fixed"]:
                        raise ValueError(
                            "Reference transport assigns a generated CA to a seed"
                        )
                    ca_ids.append(residue_index)
                    residue_owners[residue_index] = group_id
            bound_groups[group_id] = {
                "orbit_id": orbit["constraint_orbit_id"],
                "master_group_id": orbit["master_group_id"],
                "transform_index": int(transform_index),
                "atom_indices": atom_ids,
                "fixed_residue_indices": ca_ids,
                "mobile": orbit["mobility_mode"] == "orbit_rigid",
                "maximum_translation": float(orbit.get("max_translation") or 0),
                "maximum_rotation_degrees": float(orbit.get("max_rotation_deg") or 0),
                "mobility_subspace": orbit.get("mobility_subspace"),
            }
    fixed_ids = {i for i, r in enumerate(contract["residues"]) if r["fixed"]}
    if set(atom_owners) != set(range(len(records))) or set(residue_owners) != fixed_ids:
        raise ValueError(
            "Reference transport must cover every selected fixed atom and fixed CA"
        )
    reference, _, chains, _ = contract_arrays(contract)
    weights = [None] * len(reference)
    for index, owner in residue_owners.items():
        weights[index] = {"left": owner, "right": owner, "fraction": 0.0}
    for chain in chains:
        start = 0
        while start < len(chain):
            if int(chain[start]) in fixed_ids:
                start += 1
                continue
            end = start
            while end < len(chain) and int(chain[end]) not in fixed_ids:
                end += 1
            if start == 0 or end == len(chain):
                raise ValueError(
                    "Reference transport requires two fixed anchors per generated run"
                )
            ids = chain[start - 1 : end + 1]
            arclength = np.r_[
                0.0, np.cumsum(np.linalg.norm(np.diff(reference[ids], axis=0), axis=1))
            ]
            for offset, index in enumerate(chain[start:end], start=1):
                weights[int(index)] = {
                    "left": residue_owners[int(chain[start - 1])],
                    "right": residue_owners[int(chain[end])],
                    "fraction": float(arclength[offset] / arclength[-1]),
                }
            start = end
    return {
        "schema_version": 1,
        "method": METHOD,
        "base_contract_sha256": transport_fingerprint(contract),
        "fixed_atoms": records,
        "groups": bound_groups,
        "registry_transforms": matrices,
        "residue_transforms": weights,
        "symmetry_axis": axis,
    }


def validate_group_motions(plan, motions, *, tolerance):
    if set(motions) != set(plan["groups"]):
        raise ValueError(
            "Transport motion identities differ from frozen joint seed groups"
        )
    matrices = {key: proper_transform(value) for key, value in motions.items()}
    atoms = np.asarray([r["coordinate"] for r in plan["fixed_atoms"]])
    for key, group in plan["groups"].items():
        matrix = matrices[key]
        if not group["mobile"] and not np.allclose(
            matrix, np.eye(4), atol=tolerance, rtol=0
        ):
            raise ValueError("A locked seed moved in reference transport")
        master_id = group["master_group_id"]
        master = plan["groups"][master_id]
        center = atoms[master["atom_indices"]].mean(axis=0)
        move = matrices[master_id]
        displacement = move[:3, :3] @ center + move[:3, 3] - center
        angle = math.degrees(
            math.acos(float(np.clip((np.trace(move[:3, :3]) - 1) / 2, -1, 1)))
        )
        if (
            np.linalg.norm(displacement) > group["maximum_translation"] + tolerance
            or angle > group["maximum_rotation_degrees"] + tolerance
        ):
            raise ValueError(
                "Transported joint seed exceeds its original rigid motion bounds"
            )
        subspace = group["mobility_subspace"]
        if subspace in {
            "radial",
            "radial_axial",
            "radial_rotation",
            "radial_axial_rotation",
        }:
            if plan.get("symmetry_axis") is None:
                raise ValueError(
                    "Directional reference transport requires a declared cyclic axis"
                )
            axis = np.asarray(plan["symmetry_axis"]["direction"])
            offset = center - np.asarray(plan["symmetry_axis"]["point"])
            radial = offset - (offset @ axis) * axis
            if np.linalg.norm(radial) <= tolerance:
                raise ValueError("Radial transport is undefined at the symmetry axis")
            radial /= np.linalg.norm(radial)
            forbidden = displacement - (displacement @ radial) * radial
            if subspace in {"radial_axial", "radial_axial_rotation"}:
                forbidden -= (displacement @ axis) * axis
            if np.linalg.norm(forbidden) > tolerance:
                raise ValueError(
                    "Reference transport violates the declared translation subspace"
                )
        elif subspace == "tilt_only":
            if np.linalg.norm(displacement) > tolerance:
                raise ValueError(
                    "Tilt-only reference transport cannot translate the seed center"
                )
        elif group["mobile"] and subspace not in {None, "bounded_se3"}:
            raise ValueError("Unknown reference transport mobility subspace")
        action = np.asarray(plan["registry_transforms"][group["transform_index"]])
        master_action = np.asarray(
            plan["registry_transforms"][master["transform_index"]]
        )
        relative_action = action @ np.linalg.inv(master_action)
        expected = relative_action @ move @ np.linalg.inv(relative_action)
        if not np.allclose(matrix, expected, atol=tolerance, rtol=0):
            raise ValueError("Transported seed motions violate group conjugacy")
    return matrices


def transported_reference(contract, plan, motions):
    """Rebuild from the immutable initial reference and revalidate all limits."""
    if (
        plan.get("method") != METHOD
        or transport_fingerprint(contract) != plan["base_contract_sha256"]
    ):
        raise ValueError(
            "Reference transport plan is not bound to this initial contract"
        )
    matrices = validate_group_motions(
        plan, motions, tolerance=contract["limits"]["geometry_tolerance"]
    )
    out = copy.deepcopy(contract)
    transforms = []
    for record, binding in zip(
        out["residues"], plan["residue_transforms"], strict=True
    ):
        transform = interpolate_se3(
            matrices[binding["left"]], matrices[binding["right"]], binding["fraction"]
        )
        transforms.append(transform)
        record["reference_ca"] = (
            transform[:3, :3] @ np.asarray(record["reference_ca"]) + transform[:3, 3]
        ).tolist()
        record["reference_backbone"] = (
            np.asarray(record["reference_backbone"]) @ transform[:3, :3].T
            + transform[:3, 3]
        ).tolist()
    return validate_scaffold_contract(out), np.asarray(transforms)


def certify_reference_transition(before, after):
    """Certify the straight homotopy between two accepted reference polylines."""
    a, _, chains, _ = contract_arrays(before)
    b, _, other_chains, _ = contract_arrays(after)
    if [x.tolist() for x in chains] != [x.tolist() for x in other_chains]:
        raise ValueError("Reference transport changed peptide connectivity")
    epsilon = [float(np.linalg.norm(b[ids] - a[ids], axis=-1).max()) for ids in chains]
    checks = []
    minimum = before["limits"]["minimum_interchain_segment_distance"]
    for i, ids in enumerate(chains):
        for j in range(i + 1, len(chains)):
            other = chains[j]
            delta = float(
                _segment_distances(
                    a[ids[:-1]], a[ids[1:]], a[other[:-1]], a[other[1:]]
                ).min()
            )
            bound = delta - epsilon[i] - epsilon[j]
            checks.append(
                {
                    "chains": [i, j],
                    "lower_bound_angstrom": bound,
                    "passed": bound >= minimum,
                }
            )
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "scope": "CA interchain segment separation during each linear reference transition; not noisy sampler paths or self-knot certification",
    }


def replay_reference_transport(contract, plan, diagnostics):
    if diagnostics.get("plan_sha256") != transport_fingerprint(plan):
        raise ValueError("Result transport plan fingerprint differs from frozen input")
    current = contract
    transforms = np.repeat(np.eye(4)[None], len(contract["residues"]), axis=0)
    motions = {key: np.eye(4).tolist() for key in plan["groups"]}
    for record in diagnostics.get("accepted_transitions", []):
        candidate, matrices = transported_reference(
            contract, plan, record["group_transforms"]
        )
        if not certify_reference_transition(current, candidate)["passed"]:
            raise ValueError(
                "Result reference transition lacks the required segment clearance"
            )
        current, transforms, motions = candidate, matrices, record["group_transforms"]
    if diagnostics.get("final_group_transforms") != motions:
        raise ValueError(
            "Final seed motions differ from the accepted transport history"
        )
    return current, transforms
