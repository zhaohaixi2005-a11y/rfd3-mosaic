"""Reference-free N/CA/C/O nonregression for real protein interface trials."""
import numpy as np
import torch


def interface_backbone_geometry_guard(coordinates, features):
    """Bind actual backbone slots, preserving valid geometry after projection.

    CA-only callers have no full-backbone binding and keep their existing guard.
    Unknown generated identities are checked against ALA/GLY/PRO bond targets;
    their intersection prevents a generic template from excusing a real sequence
    violation. This checks geometry only, without an input scaffold or helix prior.
    """
    encoded = features.get("ref_atom_name_chars")
    if encoded is None:
        return None
    from rfd3_mosaic.validation.generated_backbone import audit_generated_backbone, default_backbone_policy

    if encoded.ndim == 3 and encoded.shape[-2:] == (4, 64):
        name_codes = encoded.argmax(-1)
    elif encoded.ndim == 2 and encoded.shape[-1] == 256:
        name_codes = encoded.reshape(-1, 4, 64).argmax(-1)
    elif encoded.ndim == 2 and encoded.shape[-1] == 4:
        name_codes = encoded
    else:
        raise ValueError("Interface atom names require native encoded [L,4], flattened one-hot [L,256] or one-hot [L,4,64] features")
    names = ["".join(chr(int(i) + 32) for i in row).strip() for row in name_codes.detach().cpu().tolist()]
    atom_tokens = features["atom_to_token_map"].detach().cpu().tolist()
    virtual = features.get("is_virtual", torch.zeros(len(names), dtype=torch.bool)).detach().cpu().tolist()
    protein = features.get("is_protein", torch.ones(len(features["asym_id"]), dtype=torch.bool)).detach().cpu().tolist()
    fixed = features["is_motif_atom_with_fixed_coord"].detach().cpu().tolist()
    positions = features["residue_index"].detach().cpu().tolist()
    chain_ids = features["asym_id"].detach().cpu().tolist()
    binding = {}
    for index, name in enumerate(names):
        token = atom_tokens[index]
        if protein[token] and not virtual[index] and name in ("N", "CA", "C", "O"):
            if name in binding.setdefault(token, {}):
                raise ValueError("Interface backbone binding contains duplicate atom names")
            binding[token][name] = index
    tokens = [t for t in sorted(binding, key=lambda t: (chain_ids[t], positions[t]))]
    if not tokens:
        return None
    if any(len(binding[t]) != 4 for t in tokens):
        raise ValueError("Interface geometry requires complete real N/CA/C/O binding")
    indices = torch.tensor([[binding[t][n] for n in ("N", "CA", "C", "O")] for t in tokens], device=coordinates.device)
    residues = [{"chain_id": str(chain_ids[t]), "residue_number": int(positions[t]),
                 "fixed": all(fixed[i] for i in binding[t].values())} for t in tokens]
    if all(r["fixed"] for r in residues):
        return None
    identities = ["<G>"] * len(protein)
    if "restype" in features:
        from atomworks.ml.encoding_definitions import AF3SequenceEncoding
        restype = features["restype"]
        codes = restype.argmax(-1) if restype.ndim == 2 else restype
        if codes.ndim != 1 or len(codes) != len(protein):
            raise ValueError("Interface residue identities require encoded [T] or one-hot [T,K] features")
        identities = AF3SequenceEncoding().decode(codes.detach().cpu().numpy()).tolist()
    standard = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
    unknown = [identities[t] not in standard for t in tokens]
    cases = ["ALA", "GLY", "PRO"] if any(unknown) else ["known"]
    contract = {"residues": residues, "backbone_policy": default_backbone_policy(),
                "helix_blocks": [], "support_edges": [],
                "limits": {"contact_distance": 8., "geometry_tolerance": .001,
                           "minimum_helix_contact_fraction": .5, "maximum_unsupported_run": 8}}

    def measure(value):
        bb = value[0, indices].detach().double().cpu().numpy()
        return {case: audit_generated_backbone(contract=contract, backbone_coordinates=bb,
                 residue_names=[case if flag else identities[t] for t, flag in zip(tokens, unknown)]) for case in cases}

    def deficits(audits):
        result = {}
        for case, audit in audits.items():
            for item in audit["geometry_failures"]:
                tolerance = audit["geometry_checks"][item["kind"]]["tolerance"]
                result[(case, item["kind"], item["residue_index"])] = abs(item["observed"] - item["target"]) - tolerance
        audit = next(iter(audits.values()))
        for item in audit["nonlocal_backbone_clashes"]:
            result[("clash", item["left_residue_index"], item["right_residue_index"], item["left_atom"], item["right_atom"])] = 2. - item["distance_angstrom"]
        return result

    before = measure(coordinates)
    old = deficits(before)
    baseline_passed = all(a["geometry_passed"] for a in before.values())

    def validate(candidate):
        if candidate.shape != coordinates.shape or candidate.device != coordinates.device or not torch.isfinite(candidate).all():
            return {"rule": "interface_backbone_geometry_nonregression", "passed": False, "reason": "invalid_coordinates"}
        after = measure(candidate)
        new = deficits(after)
        candidate_passed = all(a["geometry_passed"] for a in after.values())
        passed = candidate_passed if baseline_passed else (
            not any(a["nonlocal_backbone_clashes_truncated"] for a in [*before.values(), *after.values()])
            and all(key in old and value <= old[key] + 1e-6 for key, value in new.items()))
        return {"rule": "interface_backbone_geometry_nonregression", "passed": bool(passed),
                "baseline_geometry_passed": baseline_passed, "candidate_geometry_passed": candidate_passed,
                "baseline_geometry_failures_by_target": {k:a["geometry_failure_count"] for k,a in before.items()},
                "candidate_geometry_failures_by_target": {k:a["geometry_failure_count"] for k,a in after.items()},
                "baseline_backbone_clashes": next(iter(before.values()))["nonlocal_backbone_clash_count"],
                "candidate_backbone_clashes": next(iter(after.values()))["nonlocal_backbone_clash_count"],
                "sequence_scope": "known identities plus ALA/GLY/PRO target intersection for unknown generic protein tokens",
                "scope": "actual projected N/CA/C/O geometry only; no reference scaffold or helix/folding guarantee",
                "satisfied_constraints_tolerance": 0.0}

    return validate
