"""Bounded joint seed-pose and peptide inverse kinematics on the full assembly.

Every optimization variable is either a whole-seed rigid transform or a peptide
torsion. Symmetry copies are derived, never independent variables. The objective
is a geometric least-squares construction problem, not a physical folding energy.
Independent preparation and output audits remain the acceptance authority.
"""

from __future__ import annotations

import copy
from itertools import groupby

import numpy as np

from rfd3_mosaic.scaffold_builder import (
    _ChainModel,
    _geometry_report,
    _point_jacobian,
    _reference_torsions,
    _resized_torsions,
)


def _landmarks(old_secondary, new_secondary):
    """Injective discrete block correspondence; no interpolated Cartesian atoms."""
    old = [(kind, len(list(run))) for kind, run in groupby(old_secondary)]
    new = [(kind, len(list(run))) for kind, run in groupby(new_secondary)]
    if [x[0] for x in old] != [x[0] for x in new]:
        raise ValueError(
            "Resizing requires matching ordered secondary-structure blocks"
        )
    result, a, b = {}, 0, 0
    for (_, m), (_, n) in zip(old, new, strict=True):
        count = min(m, n)
        left = np.rint(np.linspace(0, m - 1, count)).astype(int)
        right = np.rint(np.linspace(0, n - 1, count)).astype(int)
        result.update(
            (int(a + i), int(b + j)) for i, j in zip(left, right, strict=True)
        )
        a, b = a + m, b + n
    return result


def solve_assembly(
    *, source, patterns, runs, aligned, extra, spec, candidate_audit=None
):
    """Return an audited ASU witness and source rigid motions, or unresolved.

    A single objective includes ALL runs, fixed fragments, symmetry neighbours,
    explicit same-chain support and reference landmarks. Failed global audits
    consume another bounded start; they never publish a runnable task.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix, vstack
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    options = dict(spec.get("closure") or {})
    search = dict(spec.get("construction") or {})
    allowed = {
        "attempts",
        "max_nfev",
        "maximum_pose_translation",
        "maximum_pose_rotation_deg",
        "helix_torsion_scale_deg",
    }
    if set(search) - allowed:
        raise ValueError(
            f"Unknown construction options: {sorted(set(search) - allowed)}"
        )
    attempts = search.get("attempts", options.get("attempts", 2))
    max_nfev = search.get("max_nfev", options.get("max_nfev", 150))
    if (
        type(attempts) is not int
        or not 1 <= attempts <= 32
        or type(max_nfev) is not int
        or not 1 <= max_nfev <= 5000
    ):
        raise ValueError("construction requires attempts 1..32 and max_nfev 1..5000")
    closure_tol = float(options.get("closure_tolerance", 0.005))
    shape_bound = float(
        options.get(
            "maximum_reference_ca_deviation", spec["limits"]["maximum_ca_deviation"]
        )
    )
    clash = float(options.get("clash_distance", 2.0))
    margin = float(options.get("clash_clearance_margin", 0.1))
    contact_tol = float(options.get("maximum_reference_contact_error", 2.0))
    if not all(
        np.isfinite(x) and x > 0 for x in (closure_tol, shape_bound, clash, contact_tol)
    ):
        raise ValueError("Joint construction geometry bounds must be finite positive")
    shape_weight = float(options.get("reference_shape_weight", 1 / shape_bound**2))
    clash_weight = float(options.get("nonlocal_clash_weight", 1 / clash**2))
    if (
        not np.isfinite(shape_weight)
        or shape_weight <= 0
        or not np.isfinite(clash_weight)
        or clash_weight <= 0
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError(
            "Joint construction requires positive reference shape weight and nonnegative clash margin"
        )
    for key in ("bond_tolerance", "angle_tolerance_deg", "reference_contact_cutoff"):
        if key in options and (
            isinstance(options[key], bool)
            or not np.isfinite(options[key])
            or options[key] <= 0
        ):
            raise ValueError(f"closure.{key} must be finite positive")
    sep = options.get("reference_sequence_separation", 8)
    if type(sep) is not int or sep < 2:
        raise ValueError("reference_sequence_separation must be an integer >=2")
    if type(options.get("seed", 0)) is not int or options.get("seed", 0) < 0:
        raise ValueError("closure.seed must be a nonnegative integer")
    matrices = [
        np.asarray(extra["registry_transform_matrices"][k])
        for k in extra["registry_transform_order"]
    ]
    groups = {g["group_id"]: g for g in extra["motif_constraint_groups"]}
    offsets = np.cumsum([0] + [len(p) for p in patterns])
    count = int(offsets[-1])
    chain_ids = np.concatenate([np.full(len(p), e) for e, p in enumerate(patterns)])
    fixed_locations, centers, seed_radii = {}, [], []
    for orbit_index, orbit in enumerate(extra["motif_constraint_orbits"]):
        cloud = []
        for member in groups[orbit["master_group_id"]]["members"]:
            matrix = matrices[member["sym_transform_id"]]
            for component in member["src_components"]:
                fixed_locations[component] = (orbit_index, matrix)
                cloud.extend(
                    np.asarray(list(source[component]["atoms"].values()))
                    @ matrix[:3, :3].T
                    + matrix[:3, 3]
                )
        centers.append(np.asarray(cloud).mean(axis=0))
        seed_radii.append(
            float(np.linalg.norm(np.asarray(cloud) - centers[-1], axis=-1).max())
        )
    master_count = len(centers)
    missing = {c for p in patterns for c in p if c is not None} - set(fixed_locations)
    if missing:
        raise ValueError(
            f"Unbound fixed components in joint construction: {sorted(missing)}"
        )
    max_translation = float(
        search.get("maximum_pose_translation", spec["maximum_template_seed_rmsd"])
    )
    # A rotation by theta displaces a point at radius R by 2R sin(theta/2).
    # Reuse the declared spatial scale for a default angular search bound,
    # not a claimed universal fold-compatibility angle.
    derived_rotation = 2 * np.arcsin(
        min(1.0, max_translation / (2 * max(max(seed_radii), 1e-12)))
    )
    max_rotation = np.deg2rad(
        float(search.get("maximum_pose_rotation_deg", np.rad2deg(derived_rotation)))
    )
    if not all(np.isfinite(x) and x >= 0 for x in (max_translation, max_rotation)):
        raise ValueError("Pose search bounds must be finite nonnegative")
    pose_axes = [
        (k, axis, max_translation if axis < 3 else max_rotation)
        for k in range(master_count)
        for axis in range(6)
        if (max_translation if axis < 3 else max_rotation) > 0
    ]
    pose_count = len(pose_axes)
    variables = [0.0] * pose_count
    run_specs, landmark_indices, landmark_targets = [], [], []
    pairs, distances = [], []
    for entity, entity_runs in enumerate(runs):
        for r, (start, end, old_start, old_end) in enumerate(entity_runs):
            old = aligned[entity][0][old_start - 1 : old_end + 1]
            length = end - start
            if length == len(old) - 2:
                initial = _reference_torsions(old)
                correspondence = {i: i for i in range(length)}
            else:
                try:
                    old_ss = spec["reference_secondary_structure"][entity][r]
                    new_ss = spec["secondary_structure"][entity][r]
                except (KeyError, IndexError, TypeError) as error:
                    raise ValueError(
                        "Changed lengths require per-run reference_secondary_structure and secondary_structure"
                    ) from error
                if (
                    len(old_ss) != len(old) - 2
                    or len(new_ss) != length
                    or set(old_ss + new_ss) - {"H", "L"}
                ):
                    raise ValueError(
                        "H/L assignments must exactly match reference and requested lengths"
                    )
                initial, _ = _resized_torsions(old, old_ss, new_ss)
                correspondence = _landmarks(old_ss, new_ss)
            region = slice(len(variables), len(variables) + len(initial))
            variables.extend(initial)
            run_specs.append((entity, start, end, region))
            for old_i, new_i in correspondence.items():
                landmark_indices.append(int(offsets[entity]) + start + new_i)
                landmark_targets.append(old[old_i + 1, 1])
            sep = int(options.get("reference_sequence_separation", 8))
            cutoff = float(options.get("reference_contact_cutoff", 8.0))
            for i, new_i in correspondence.items():
                for j, new_j in correspondence.items():
                    if j - i < sep or new_j - new_i < sep:
                        continue
                    d = np.linalg.norm(old[i + 1, 1] - old[j + 1, 1])
                    if d <= cutoff:
                        pairs.append(
                            (
                                int(offsets[entity]) + start + new_i,
                                int(offsets[entity]) + start + new_j,
                            )
                        )
                        distances.append(d)
    x0 = np.asarray(variables)
    targets = np.asarray(landmark_targets)
    landmarks = np.asarray(landmark_indices)
    contact_pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    contact_targets = np.asarray(distances)
    for block in spec["helix_blocks"]:
        entity, start, end = (block.get(k) for k in ("entity", "start", "end"))
        if (
            type(entity) is not int
            or not 0 <= entity < len(patterns)
            or type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= len(patterns[entity])
        ):
            raise ValueError(
                "Helix block must identify a valid one-based ASU residue interval"
            )
    blocks = {
        b["id"]: np.arange(
            offsets[b["entity"]] + b["start"] - 1, offsets[b["entity"]] + b["end"]
        )
        for b in spec["helix_blocks"]
    }
    if len(blocks) != len(spec["helix_blocks"]):
        raise ValueError("Helix block IDs must be unique across ASU entities")
    for edge in spec["support_edges"]:
        if edge["left"] not in blocks or edge["right"] not in blocks:
            raise ValueError("Support edge references unknown helix block")
    partners = {name: set() for name in blocks}
    for edge in spec["support_edges"]:
        partners[edge["left"]].update(blocks[edge["right"]])
        partners[edge["right"]].update(blocks[edge["left"]])
    h_positions = {int(i) for ids in blocks.values() for i in ids}
    h_columns = []
    for entity, start, end, region in run_specs:
        for i in range(end - start):
            if offsets[entity] + start + i in h_positions:
                h_columns.extend((region.start + 2 * i, region.start + 2 * i + 1))
    h_columns = np.asarray(h_columns, dtype=int)
    h_scale = np.deg2rad(float(search.get("helix_torsion_scale_deg", 30.0)))
    if not np.isfinite(h_scale) or h_scale <= 0:
        raise ValueError("helix_torsion_scale_deg must be finite positive")

    def forward(x, jacobian=False):
        params = np.zeros((master_count, 6))
        for value, (k, axis, _) in zip(x[:pose_count], pose_axes, strict=True):
            params[k, axis] = value
        rotations = Rotation.from_rotvec(params[:, 3:]).as_matrix()
        transformed = {}
        for component, (k, matrix) in fixed_locations.items():
            record = copy.deepcopy(source[component])
            for atom, point in record["atoms"].items():
                master = np.asarray(point) @ matrix[:3, :3].T + matrix[:3, 3]
                moved = (
                    (master - centers[k]) @ rotations[k].T + centers[k] + params[k, :3]
                )
                record["atoms"][atom] = (moved - matrix[:3, 3]) @ matrix[:3, :3]
            transformed[component] = record
        xyz = np.zeros((count, 4, 3))
        for entity, pattern in enumerate(patterns):
            for i, component in enumerate(pattern):
                if component is not None:
                    xyz[offsets[entity] + i] = [
                        transformed[component]["atoms"][a]
                        for a in ("N", "CA", "C", "O")
                    ]
        derivative = np.zeros((*xyz.shape, len(x))) if jacobian else None
        closures, closure_jac = [], []
        for entity, start, end, region in run_specs:
            a, b = int(offsets[entity]) + start, int(offsets[entity]) + end
            model = _ChainModel.from_anchors(xyz[a - 1], xyz[b], end - start)
            local, origins, axes, affected = model.forward(x[region])
            xyz[a:b] = local[1:-1]
            closures.append(local[-1, :3] - xyz[b, :3])
            if jacobian:
                local_jac = _point_jacobian(
                    local, origins, axes, affected, np.arange(local.size // 3)
                ).reshape(*local.shape, region.stop - region.start)
                derivative[a:b, :, :, region] = local_jac[1:-1]
                cjac = np.zeros((3, 3, len(x)))
                cjac[:, :, region] = local_jac[-1, :3]
                closure_jac.append(cjac)
        closures = np.asarray(closures)
        if jacobian:
            closure_jac = np.asarray(closure_jac)
            # Only the few rigid-pose columns use numerical differentiation;
            # every torsion column uses the exact kinematic Jacobian.
            for column in range(pose_count):
                h = 1e-6
                plus, minus = x.copy(), x.copy()
                plus[column] += h
                minus[column] -= h
                xp, cp, _, _ = forward(plus)
                xm, cm, _, _ = forward(minus)
                derivative[..., column] = (xp - xm) / (2 * h)
                closure_jac[..., column] = (cp - cm) / (2 * h)
        return xyz, closures, transformed, (derivative, closure_jac)

    cached_x = cached = None

    def evaluate(x):
        nonlocal cached_x, cached
        if cached_x is not None and np.array_equal(x, cached_x):
            return cached
        xyz, closure, _, (jac, cjac) = forward(x, True)
        residuals = [(closure / closure_tol).reshape(-1)]
        derivatives = [csr_matrix(cjac.reshape(-1, len(x)) / closure_tol)]
        ca, cj = xyz[:, 1], jac[:, 1]
        residuals.append(
            (np.sqrt(shape_weight) * (ca[landmarks] - targets)).reshape(-1)
        )
        derivatives.append(
            csr_matrix(np.sqrt(shape_weight) * cj[landmarks].reshape(-1, len(x)))
        )
        if len(h_columns):
            delta = (x[h_columns] - x0[h_columns]) / 2
            residuals.append(2 * np.sin(delta) / h_scale)
            derivatives.append(
                csr_matrix(
                    (np.cos(delta) / h_scale, (np.arange(len(h_columns)), h_columns)),
                    shape=(len(h_columns), len(x)),
                )
            )

        def distance_terms(a, b, targets_, scale, hinge=False):
            difference = ca[a] - ca[b]
            lengths = np.linalg.norm(difference, axis=-1)
            signed = lengths - targets_
            residuals.append(scale * np.maximum(signed, 0) if hinge else scale * signed)
            dj = np.einsum(
                "ni,nij->nj",
                difference / np.maximum(lengths[:, None], 1e-12),
                cj[a] - cj[b],
            )
            derivatives.append(
                csr_matrix(scale * dj * (signed > 0)[:, None] if hinge else scale * dj)
            )

        if len(contact_pairs):
            distance_terms(
                contact_pairs[:, 0],
                contact_pairs[:, 1],
                contact_targets,
                1 / contact_tol,
            )
        # Coverage is required on EACH declared edge, as in the final contract.
        # The longest unsupported window uses the union of declared partners.
        for edge in spec["support_edges"]:
            for left, right in (
                (edge["left"], edge["right"]),
                (edge["right"], edge["left"]),
            ):
                ids, other = blocks[left], blocks[right]
                d = np.linalg.norm(ca[ids, None] - ca[other], axis=-1)
                nearest = other[d.argmin(axis=1)]
                k = int(
                    np.ceil(spec["limits"]["minimum_helix_contact_fraction"] * len(ids))
                )
                selected = np.argsort(d.min(axis=1))[:k]
                distance_terms(
                    ids[selected],
                    nearest[selected],
                    spec["limits"]["contact_distance"],
                    1.0,
                    hinge=True,
                )
        for name, partner in partners.items():
            if not partner:
                continue
            ids, other = blocks[name], np.asarray(sorted(partner))
            d = np.linalg.norm(ca[ids, None] - ca[other], axis=-1)
            nearest = other[d.argmin(axis=1)]
            lengths = d.min(axis=1)
            if "maximum_unsupported_run" in spec["limits"]:
                width = spec["limits"]["maximum_unsupported_run"] + 1
                for window in range(len(ids) - width + 1):
                    at = window + int(np.argmin(lengths[window : window + width]))
                    # Attraction only when the closest contact exceeds the cutoff.
                    distance_terms(
                        ids[at : at + 1],
                        nearest[at : at + 1],
                        spec["limits"]["contact_distance"],
                        1.0,
                        hinge=True,
                    )
        # Fixed slots per ASU residue / full-assembly residue pair; the nearest
        # atom pair is evaluated only inside a KD-tree neighbourhood.
        full = np.concatenate([xyz @ m[:3, :3].T + m[:3, 3] for m in matrices])
        full_jac = np.concatenate(
            [np.einsum("ab,nkbt->nkat", m[:3, :3], jac) for m in matrices]
        )
        n_full = len(full)
        r = np.zeros(count * n_full)
        rows, columns, values = [], [], []
        nearest_pairs = {}
        flat = full.reshape(-1, 3)
        for a, b in cKDTree(flat).query_pairs(clash + margin):
            ia, ib = a // 4, b // 4
            if ia >= count or ia == ib:
                continue
            if ib < count and chain_ids[ia] == chain_ids[ib] and abs(ia - ib) < 2:
                continue
            slot = ia * n_full + ib
            delta = flat[a] - flat[b]
            d = np.linalg.norm(delta)
            if slot not in nearest_pairs or d < nearest_pairs[slot][0]:
                nearest_pairs[slot] = (d, delta, a, b)
        for slot, (d, delta, a, b) in nearest_pairs.items():
            r[slot] = np.sqrt(clash_weight) * (clash + margin - d)
            dj = (
                -np.sqrt(clash_weight)
                * (delta / max(d, 1e-12))
                @ (full_jac[a // 4, a % 4] - full_jac[b // 4, b % 4])
            )
            nonzero = np.flatnonzero(dj)
            rows.extend([slot] * len(nonzero))
            columns.extend(nonzero)
            values.extend(dj[nonzero])
        residuals.append(r)
        derivatives.append(
            csr_matrix((values, (rows, columns)), shape=(len(r), len(x)))
        )
        cached_x = x.copy()
        cached = (np.concatenate(residuals), vstack(derivatives, format="csr"))
        return cached

    lower, upper = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    for i, (_, _, limit) in enumerate(pose_axes):
        lower[i], upper[i] = -limit / np.sqrt(3), limit / np.sqrt(3)
    rng = np.random.default_rng(options.get("seed", 0))
    report = {
        "status": "unresolved",
        "method": "joint_full_assembly_internal_coordinates_and_seed_rigid_poses",
        "attempt_budget": attempts,
        "max_nfev_per_attempt": max_nfev,
        "attempts": [],
        "maximum_pose_translation_angstrom": max_translation,
        "maximum_pose_rotation_degrees": float(np.rad2deg(max_rotation)),
        "rotation_bound_source": "explicit"
        if "maximum_pose_rotation_deg" in search
        else "2 asin(translation_bound/(2 maximum_seed_radius))",
        "nonlocal_clash_weight_per_angstrom_squared": clash_weight,
        "pose_bounds": "axis box inscribed in the declared translation/rotation norm balls",
        "reference_landmark_count": len(landmarks),
        "shape_bound_angstrom": shape_bound,
        "reference_mapping": "injective discrete H/L block landmarks; no XYZ interpolation",
        "helix_torsion_scale_deg": float(np.rad2deg(h_scale)),
        "pose_group_order": [
            o["master_group_id"] for o in extra["motif_constraint_orbits"]
        ],
        "objective_parameters": {
            "closure_tolerance_angstrom": closure_tol,
            "reference_contact_tolerance_angstrom": contact_tol,
            "reference_shape_weight_per_angstrom_squared": shape_weight,
            "clash_distance_angstrom": clash,
            "clash_margin_angstrom": margin,
            "reference_contact_cutoff_angstrom": options.get(
                "reference_contact_cutoff", 8.0
            ),
            "reference_sequence_separation": sep,
            "bond_tolerance_angstrom": options.get("bond_tolerance", 0.04),
            "angle_tolerance_degrees": options.get("angle_tolerance_deg", 8.0),
            "seed": options.get("seed", 0),
            "support_limits": spec["limits"],
            "energy_semantics": "geometric residual least squares, not physical folding energy",
        },
        "generated_lengths": [end - start for _, start, end, _ in run_specs],
    }
    for attempt in range(attempts):
        trial = x0.copy()
        if attempt:
            trial[pose_count:] += rng.normal(0, np.deg2rad(15), len(trial) - pose_count)
        result = least_squares(
            lambda x: evaluate(x)[0],
            trial,
            jac=lambda x: evaluate(x)[1],
            bounds=(lower, upper),
            tr_solver="lsmr",
            max_nfev=max_nfev,
            tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 10 * len(x0)},
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        xyz, closure, transformed, _ = forward(result.x)
        closure_error = float(np.linalg.norm(closure, axis=-1).max(initial=0))
        shape_error = float(
            np.linalg.norm(xyz[landmarks, 1] - targets, axis=-1).max(initial=0)
        )
        geometry = []
        for entity, start, end, _ in run_specs:
            local = xyz[offsets[entity] + start - 1 : offsets[entity] + end + 1]
            geometry.append(
                _geometry_report(
                    local,
                    bond_tolerance=options.get("bond_tolerance", 0.04),
                    angle_tolerance_deg=options.get("angle_tolerance_deg", 8.0),
                    clash_distance=clash,
                )
            )
        asu = []
        for entity, pattern in enumerate(patterns):
            asu.append(
                [
                    copy.deepcopy(transformed[c])
                    if c is not None
                    else {
                        "name": "ALA",
                        "atoms": dict(
                            zip(
                                ("N", "CA", "C", "O"),
                                xyz[offsets[entity] + i],
                                strict=True,
                            )
                        ),
                    }
                    for i, c in enumerate(pattern)
                ]
            )
        audit = (
            candidate_audit(asu)
            if candidate_audit is not None
            else {"passed": False, "reason": "independent_assembly_audit_required"}
        )
        contact_error = (
            float(
                np.abs(
                    np.linalg.norm(
                        xyz[contact_pairs[:, 0], 1] - xyz[contact_pairs[:, 1], 1],
                        axis=-1,
                    )
                    - contact_targets
                ).max(initial=0)
            )
            if len(contact_pairs)
            else 0.0
        )
        accepted = (
            closure_error <= closure_tol
            and shape_error <= shape_bound
            and contact_error <= contact_tol
            and all(g["passed"] for g in geometry)
            and audit["passed"]
        )
        report["attempts"].append(
            {
                "nfev": result.nfev,
                "optimizer_success": result.success,
                "closure_error_angstrom": closure_error,
                "reference_landmark_error_angstrom": shape_error,
                "reference_contact_error_angstrom": contact_error,
                "geometry": geometry,
                "assembly_audit": audit,
                "accepted": bool(accepted),
            }
        )
        if accepted:
            report.update(
                status="verified_assembly_geometry",
                selected_attempt=attempt,
                pose_parameters=[
                    {"orbit": k, "axis": axis, "value": float(result.x[i])}
                    for i, (k, axis, _) in enumerate(pose_axes)
                ],
            )
            return asu, report
    report["reason"] = "no_verified_assembly_within_finite_joint_search_budget"
    return None, report
