"""Constrain clean partial predictions to the explicit, validated backbone prior.

The reference is available only for complete-template partial diffusion. This
is a hard feasibility filter, not an energy improvement or folding claim.
"""

import torch
import math


def project_explicit_scaffold_feasibility(coordinates, topology, *, projector,
                                         iterations=16, minimum_fraction=1e-3):
    from rfd3_mosaic.validation.scaffold_contract import audit_scaffold_contract

    if type(iterations) is not int or not 0 <= iterations <= 64 or not math.isfinite(minimum_fraction) or not 0 < minimum_fraction < 1:
        raise ValueError("Invalid feasibility search limits")

    reference = topology.scaffold_contract
    if reference is None or reference.backbone_atom_indices is None:
        raise ValueError("Feasibility filtering requires an explicit v2 backbone reference")
    if coordinates.ndim != 3 or coordinates.shape[0] != 1 or not torch.isfinite(coordinates).all():
        raise ValueError("Feasibility filtering requires finite [1, atoms, 3] coordinates")
    contract = reference.contract
    ids = reference.backbone_atom_indices
    residues = contract["residues"]

    def evaluate(value):
        bb = value[0, ids].detach().double().cpu().numpy()
        audit = audit_scaffold_contract(
            contract=contract, coordinates=bb[:, 1],
            chain_ids=[r["chain_id"] for r in residues],
            residue_numbers=[r["residue_number"] for r in residues],
            fixed_mask=[r["fixed"] for r in residues],
            backbone_coordinates=bb,
            residue_names=[r["residue_name"] for r in residues],
        )
        # Keep the declared 1A envelope itself, rather than spending its
        # independent-audit roundoff allowance during each clean update.
        passed = audit["passed"] and audit["maximum_ca_deviation_angstrom"] <= contract["limits"]["maximum_ca_deviation"]
        return passed, audit

    source = projector(coordinates)
    passed, initial = evaluate(source)
    if passed:
        return source, {"applied": False, "fraction_retained": 1.0, "passed": True}

    # Try the smallest residue translation that restores only the CA envelope
    # before falling back to a full-backbone reference path. Revalidate all
    # physical constraints, since a CA-only clip can damage peptide geometry.
    source_ca = source[0, reference.ca_atom_indices]
    target_ca = reference.reference_ca.to(dtype=source.dtype)
    displacement = source_ca - target_ca
    norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    limit = contract["limits"]["maximum_ca_deviation"] - 2e-5
    clipped_ca = target_ca + displacement * (limit / norm.clamp_min(1e-12)).clamp(max=1)
    token_ids = topology.atom_to_token[reference.ca_atom_indices]
    local_shift = source.new_zeros((len(topology.generated_token_mask), 3))
    local_shift[token_ids] = clipped_ca - source_ca
    local_shift[~topology.generated_token_mask] = 0
    local = projector(source + local_shift[topology.atom_to_token][None])
    local_passed, local_audit = evaluate(local)
    if local_passed:
        return local.detach(), {
            "applied": True, "fraction_retained": None, "method": "local_CA_envelope_translation",
            "passed": True, "maximum_atom_displacement": float(torch.linalg.vector_norm(local-source,dim=-1).max()),
            "initial_max_ca_deviation": initial["maximum_ca_deviation_angstrom"],
            "final_max_ca_deviation": local_audit["maximum_ca_deviation_angstrom"],
            "initial_backbone_passed": initial["generated_backbone"]["passed"],
            "final_backbone_passed": local_audit["generated_backbone"]["passed"],
        }

    # All non-backbone model slots follow their residue's CA displacement,
    # retaining the learned local side-chain arrangement. N/CA/C/O use the
    # explicitly supplied prior. Fixed atoms are restored by the projector.
    source_ca = source[0, reference.ca_atom_indices]
    target_ca = reference.reference_ca.to(dtype=source.dtype)
    token_shift = source.new_zeros((len(topology.generated_token_mask), 3))
    token_ids = topology.atom_to_token[reference.ca_atom_indices]
    token_shift[token_ids] = target_ca - source_ca
    anchor = source + token_shift[topology.atom_to_token][None]
    anchor[:, ids] = reference.reference_backbone.to(dtype=source.dtype)[None]
    anchor = projector(anchor)
    anchor_passed, _ = evaluate(anchor)
    if not anchor_passed:
        raise ValueError("Declared backbone prior is infeasible after the actual hard projector")

    # Feasibility is nonconvex (angles and actual-alpha support). Start at
    # the prediction and scan downwards before refining a feasible interval;
    # bisection from [0,1] alone can miss the near-prediction feasible branch.
    low, high = 0.0, 1.0
    best = anchor
    for index in range(1, 33):
        fraction = 1.0 - index / 32.0
        candidate = projector(anchor + fraction * (source - anchor))
        valid, _ = evaluate(candidate)
        if valid:
            low, high, best = fraction, fraction + 1.0 / 32.0, candidate
            break
    for _ in range(iterations):
        fraction = (low + high) * 0.5
        candidate = projector(anchor + fraction * (source - anchor))
        valid, _ = evaluate(candidate)
        if valid:
            best, low = candidate, fraction
        else:
            high = fraction
    if low < minimum_fraction:
        raise ValueError("No nontrivial clean prediction satisfies the declared full backbone contract")
    safe_fraction = low * 0.999
    safe_candidate = projector(anchor + safe_fraction * (source - anchor))
    safe_passed, _ = evaluate(safe_candidate)
    if safe_passed:
        best, low = safe_candidate, safe_fraction
    # Recheck the actual stored candidate, including hard projection. No
    # reporting-only tolerance compensation or mutation of the reference.
    passed, final = evaluate(best)
    if not passed:
        raise RuntimeError("Feasibility filter returned an invalid stored candidate")
    return best.detach(), {
        "applied": True, "fraction_retained": low, "passed": True,
        "method": "reference_backbone_feasible_branch",
        "maximum_atom_displacement": float(torch.linalg.vector_norm(best-source, dim=-1).max()),
        "initial_max_ca_deviation": initial["maximum_ca_deviation_angstrom"],
        "final_max_ca_deviation": final["maximum_ca_deviation_angstrom"],
        "initial_backbone_passed": initial["generated_backbone"]["passed"],
        "final_backbone_passed": final["generated_backbone"]["passed"],
        "scope": "explicit reference CA/backbone feasibility; side-chain energy and folding not certified",
    }


def backbone_physical_nonregression_guard(coordinates, topology):
    """Use actual N/CA/C/O and actual-alpha support after hard projection.

    Unlike reference-position proxies, these requirements are invariant to
    moving the reference frame. Thus the same check applies to pose proposals.
    A feasible baseline must stay feasible; existing violations may improve
    but never introduce another failing bond, angle, or nonlocal atom pair.
    """
    from rfd3_mosaic.validation.generated_backbone import audit_generated_backbone

    reference = topology.scaffold_contract

    def measure(value):
        bb = value[0, reference.backbone_atom_indices].detach().double().cpu().numpy()
        return audit_generated_backbone(
            contract=reference.contract, backbone_coordinates=bb,
            residue_names=[r["residue_name"] for r in reference.contract["residues"]],
        )

    def deficits(audit):
        result = {}
        for item in audit["geometry_failures"]:
            key = (item["kind"], item["residue_index"])
            tolerance = audit["geometry_checks"][item["kind"]]["tolerance"]
            result[key] = abs(item["observed"] - item["target"]) - tolerance
        cutoff = reference.contract["backbone_policy"]["minimum_nonlocal_backbone_distance_angstrom"]
        for item in audit["nonlocal_backbone_clashes"]:
            key = ("nonlocal_backbone", item["left_residue_index"], item["right_residue_index"], item["left_atom"], item["right_atom"])
            result[key] = cutoff - item["distance_angstrom"]
        return result

    before = measure(coordinates)
    old = deficits(before)

    def validate(candidate):
        after = measure(candidate)
        new = deficits(after)
        if before["passed"]:
            passed = after["passed"]
        else:
            passed = (
                not before["nonlocal_backbone_clashes_truncated"]
                and not after["nonlocal_backbone_clashes_truncated"]
                and all(key in old and value <= old[key] + 1e-6 for key, value in new.items())
                and (after["helix_support_passed"] or before["helix_failures"] == after["helix_failures"])
            )
        return {
            "rule": "full_backbone_physical_nonregression", "passed": bool(passed),
            "baseline_passed": before["passed"], "candidate_passed": after["passed"],
            "baseline_geometry_failures": before["geometry_failure_count"],
            "candidate_geometry_failures": after["geometry_failure_count"],
            "baseline_backbone_clashes": before["nonlocal_backbone_clash_count"],
            "candidate_backbone_clashes": after["nonlocal_backbone_clash_count"],
            "candidate_helix_support_passed": after["helix_support_passed"],
            "satisfied_constraints_tolerance": 0.0,
        }

    return validate
