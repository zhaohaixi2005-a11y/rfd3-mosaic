"""Prepare a task-level complete scaffold on CPU before partial diffusion.

A template is an explicit geometric prior, not a folding predictor. Selection
is conditional on the materialized contig, joint seed pose, closed backbone,
declared intrachain block contacts and interchain separation. No GPU search or
hidden threshold fitting is performed here. A failed finite closure search is
reported as unresolved, not as proof that the requested fold cannot exist.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfd3_mosaic.assembly_compiler import compile_experiment_assembly
from rfd3_mosaic.pose_tasks import _rotation_xyz_degrees
from rfd3_mosaic.scaffold_input import (
    _compiled_contig_chains,
    _components,
    _residues,
    apply_scaffold_input,
    compiled_scaffold_contract_sha256,
    scaffold_transport_plan,
)
from rfd3_mosaic.schema import UserDesignSpec, load_user_design
from rfd3_mosaic.seed_stabilizer import _fit_transform
from rfd3_mosaic.validation.generated_backbone import default_backbone_policy
from rfd3_mosaic.validation.scaffold_contract import validate_scaffold_contract

_BB = ("N", "CA", "C", "O")


class ScaffoldConstructionUnresolved(ValueError):
    """A bounded search failed; retain its evidence without publishing a task."""

    def __init__(self, entity: int, run: int, report: dict):
        self.report = {
            "schema_version": 1,
            "status": "unresolved",
            "entity": entity,
            "run": run,
            "closure": report,
            "gpu_jobs_submitted": 0,
            "task": None,
        }
        super().__init__(
            f"CPU scaffold construction unresolved ({'full assembly' if entity < 0 else f'entity {entity}, run {run}'}): {report.get('reason', 'no verified witness')}"
        )


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be explicit, positive and finite")
    return float(value)


def _chain_name(index: int) -> str:
    result = ""
    while index >= 0:
        result = chr(65 + index % 26) + result
        index = index // 26 - 1
    return result


def _write_cif(path: Path, chains: list[list[dict]]) -> None:
    fields = (
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "auth_seq_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    )
    lines = [
        "data_mosaic_complete_scaffold",
        "#",
        "loop_",
        *["_atom_site." + x for x in fields],
    ]
    serial = 0
    for index, chain in enumerate(chains):
        label = _chain_name(index)
        for number, residue in enumerate(chain, 1):
            for atom, xyz in residue["atoms"].items():
                serial += 1
                name = residue["name"]
                if not re.fullmatch(r"[A-Z0-9]{3}", name) or not re.fullmatch(
                    r"[A-Za-z0-9]+", atom
                ):
                    raise ValueError(
                        "Unsupported protein residue/atom name in complete scaffold"
                    )
                lines.append(
                    " ".join(
                        (
                            "ATOM",
                            str(serial),
                            atom[0],
                            atom,
                            ".",
                            name,
                            label,
                            str(index + 1),
                            str(number),
                            "?",
                            *(f"{v:.8f}" for v in xyz),
                            "1.00",
                            "0.00",
                            str(number),
                            name,
                            label,
                            atom,
                            "1",
                        )
                    )
                )
    path.write_text("\n".join(lines) + "\n#\n", encoding="utf-8")


def _full_backbone_clearance(chains: list[list[dict]], minimum: float) -> dict:
    """Check all non-neighbour backbone atoms, including other seeds/copies."""
    from scipy.spatial import cKDTree

    xyz, identities = [], []
    for chain_index, chain in enumerate(chains):
        for number, residue in enumerate(chain):
            for atom in _BB:
                xyz.append(residue["atoms"][atom])
                identities.append((chain_index, number, atom))
    xyz = np.asarray(xyz)
    violations = []
    for left, right in cKDTree(xyz).query_pairs(minimum):
        a, b = identities[left], identities[right]
        if a[0] == b[0] and abs(a[1] - b[1]) < 2:
            continue
        distance = float(np.linalg.norm(xyz[left] - xyz[right]))
        if distance < minimum:
            violations.append({"left": a, "right": b, "distance_angstrom": distance})
    report = {
        "passed": not violations,
        "minimum_allowed_angstrom": minimum,
        "atom_scope": "N,CA,C,O; excluding same-chain identical/adjacent residues",
        "violation_count": len(violations),
        "violations": sorted(violations, key=lambda x: x["distance_angstrom"])[:20],
    }
    if violations:
        raise ValueError(
            f"Complete scaffold has nonlocal backbone clashes: {json.dumps(report)}"
        )
    return report


def _compile(design: UserDesignSpec, directory: Path) -> tuple[dict, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "design.yaml"
    config.write_text(
        yaml.safe_dump(
            design.model_dump(mode="json", exclude_none=True), sort_keys=False
        )
    )
    compiled = compile_experiment_assembly(
        {"kind": "user_design", "config": str(config), "example_id": design.name},
        directory / "compiled",
        project_directory=design.input.parent,
        experiment_name=design.name,
    )
    return json.loads(compiled.input_path.read_text()), compiled.input_path.parent


def _layout(pattern: list[str | None], reference_length: int, declared_lengths=None):
    """Map fixed spans and generated runs without silently changing contig lengths."""
    runs = []
    start = 0
    while start < len(pattern):
        end = start + 1
        while end < len(pattern) and (pattern[end] is None) == (pattern[start] is None):
            end += 1
        runs.append((start, end, pattern[start] is None))
        start = end
    generated = [(a, b) for a, b, g in runs if g]
    if not generated or pattern[0] is None or pattern[-1] is None:
        raise ValueError(
            "prepare-scaffold currently requires generated runs bounded by two fixed fragments; terminal-growth mode remains available in ordinary runs"
        )
    if declared_lengths is None:
        if reference_length == len(pattern):
            lengths = [b - a for a, b in generated]
        elif len(generated) == 1:
            lengths = [reference_length - sum(x is not None for x in pattern)]
        else:
            raise ValueError(
                "Multiple resized runs require reference_generated_lengths for each entity"
            )
    else:
        lengths = declared_lengths
    if (
        not isinstance(lengths, list)
        or len(lengths) != len(generated)
        or any(type(n) is not int or n < 1 for n in lengths)
    ):
        raise ValueError(
            "reference_generated_lengths must contain one positive integer per generated run"
        )
    mapping, aligned_runs, cursor, run_index = {}, [], 0, 0
    for a, b, g in runs:
        length = lengths[run_index] if g else b - a
        if g:
            aligned_runs.append((a, b, cursor, cursor + length))
            run_index += 1
        else:
            mapping.update((i, cursor + i - a) for i in range(a, b))
        cursor += length
    if cursor != reference_length:
        raise ValueError(
            "Reference lengths and fixed spans do not cover the complete reference chain"
        )
    return mapping, aligned_runs


def _freeze_fitted_poses(design, extra, source, patterns, maps, aligned, maximum_rmsd):
    """Fit each entire joint seed; never fit its interface halves independently."""
    samples = extra["initialization_samples"]
    registry = [
        np.asarray(extra["registry_transform_matrices"][key])
        for key in extra["registry_transform_order"]
    ]
    locations = {
        component: (entity, ordinal)
        for entity, chain in enumerate(patterns)
        for ordinal, component in enumerate(chain)
        if component is not None
    }
    groups = {group["group_id"]: group for group in extra["motif_constraint_groups"]}
    payload = design.model_dump(mode="json", exclude_none=True)
    poses, reports = {}, []
    for orbit in extra["motif_constraint_orbits"]:
        group = groups[orbit["master_group_id"]]
        key = group["coupling_group_id"]
        original, target = [], []
        for member in group["members"]:
            transform = member["sym_transform_id"]
            matrix = registry[transform]
            for component in member["src_components"]:
                entity, position = locations[component]
                reference = aligned[entity][transform][maps[entity][position]]
                for slot, atom in enumerate(_BB):
                    original.append(
                        source[component]["atoms"][atom] @ matrix[:3, :3].T
                        + matrix[:3, 3]
                    )
                    target.append(reference[slot])
        rotation, translation, rmsd = _fit_transform(
            np.asarray(original), np.asarray(target)
        )
        if rmsd > maximum_rmsd:
            raise ValueError(
                f"Template joint seed fit {key} RMSD {rmsd:.4f} exceeds explicit limit {maximum_rmsd}; reference is not a compatible pose prior"
            )
        sample = samples[key]
        total_rotation = rotation @ np.asarray(sample["rotation_matrix"])
        center = rotation @ np.asarray(sample["target_center"]) + translation
        symmetry = design.symmetry
        axis = (
            np.array([0.0, 0.0, 1.0])
            if isinstance(symmetry, str)
            else np.asarray(symmetry.axis, dtype=float)
        )
        origin = (
            np.zeros(3)
            if isinstance(symmetry, str)
            else np.asarray(symmetry.center, dtype=float)
        )
        axis /= np.linalg.norm(axis)
        axial = float(np.dot(center - origin, axis))
        radial = center - origin - axial * axis
        radius = float(np.linalg.norm(radial))
        radial_direction = (
            radial / radius
            if radius > 1e-10
            else np.asarray(sample["radial_direction"])
        )
        poses[key] = {
            "radius": {"minimum": radius, "maximum": radius},
            "axial_offset": {"minimum": axial, "maximum": axial},
            "radial_direction": radial_direction.tolist(),
            "orientation": {
                "method": "fixed",
                "rotation_deg": _rotation_xyz_degrees(total_rotation),
            },
        }
        reports.append(
            {
                "motion_group": key,
                "joint_backbone_rmsd_angstrom": rmsd,
                "method": "one proper rigid fit of every fragment in the joint seed",
            }
        )
    sampling = payload["sampling"]
    sampling.pop("scaffold_artifact", None)
    if len(poses) == 1 and not design.sampling.initial_poses:
        sampling.pop("initial_poses", None)
        sampling["initial_pose"] = next(iter(poses.values()))
    else:
        # Public component and native motion IDs differ. Resolve by their
        # explicitly recorded sampling seeds, as prepare-poses does.
        remapped = {}
        for public, old in design.sampling.initial_poses.items():
            matches = [k for k, s in samples.items() if s["random_seed"] == old.seed]
            if len(matches) != 1 or matches[0] not in poses:
                raise ValueError(
                    "Multiple seed poses require unique explicit initial_poses seeds"
                )
            remapped[public] = poses[matches[0]]
        if len(remapped) != len(poses):
            raise ValueError(
                "Every joint seed requires an explicit initial_poses entry"
            )
        sampling.pop("initial_pose", None)
        sampling["initial_poses"] = remapped
    return UserDesignSpec.model_validate(payload), reports


def _remap_extra(extra, patterns, groups_count):
    """Relabel selectors into transform-major complete chains, preserving graph."""
    count = len(patterns)
    locations = {
        component: (entity, ordinal + 1)
        for entity, chain in enumerate(patterns)
        for ordinal, component in enumerate(chain)
        if component is not None
    }

    def components(values, transform=0):
        expanded = [x for value in values for x in _components(value)]
        return [
            f"{_chain_name(transform * count + locations[x][0])}{locations[x][1]}"
            for x in expanded
        ]

    def selection(text):
        return ",".join(components(_components(text)))

    result = {
        key: copy.deepcopy(extra.get(key, []))
        for key in (
            "motif_constraint_groups",
            "motif_constraint_orbits",
            "assembly_interface_relations",
            "asu_scaffold_segments",
            "asu_terminal_extensions",
        )
    }
    for group in result["motif_constraint_groups"]:
        for member in group["members"]:
            old = member["src_components"]
            member["src_components"] = components(old, member["sym_transform_id"])
            member["correspondence_components"] = components(old)
    for orbit in result["motif_constraint_orbits"]:
        if "source_components" in orbit:
            orbit["source_components"] = components(orbit["source_components"])
    from rfd3_mosaic.scaffold_input import remap_scaffold_interface_relations

    result["assembly_interface_relations"] = remap_scaffold_interface_relations(
        extra.get("assembly_interface_relations", []),
        extra["motif_constraint_groups"],
        result["motif_constraint_groups"],
    )
    for segment in result["asu_scaffold_segments"]:
        for key in ("from_selector", "to_selector"):
            if segment.get(key):
                segment[key] = selection(segment[key])
        if segment.get("path_selectors"):
            segment["path_selectors"] = [
                selection(x) for x in segment["path_selectors"]
            ]
        segment["contig_chains"] = [
            ",".join(
                selection(token) if re.match(r"[A-Za-z]", token) else token
                for token in chain.split(",")
            )
            for chain in segment["contig_chains"]
        ]
    layout = [
        {"transform_index": g, "entity_id": entity, "is_asu": g == 0}
        for g in range(groups_count)
        for entity in range(count)
    ]
    result.update(
        preexpanded_chain_layout=layout,
        asu_chain_count=len(layout),
        asu_polymer_chain_count=len(layout),
        generated_coordinate_initialization="complete_scaffold_partial_diffusion",
    )
    return result, components


def _expand_asu(asu, matrices):
    return [
        [
            {
                "name": r["name"],
                "atoms": {
                    a: np.asarray(xyz) @ matrix[:3, :3].T + matrix[:3, 3]
                    for a, xyz in r["atoms"].items()
                },
            }
            for r in chain
        ]
        for matrix in matrices
        for chain in asu
    ]


def _scaffold_contract(full, patterns, spec, groups_count):
    """One v2 contract is used during construction, native handoff and output."""
    contract = {
        "schema_version": 2,
        "residues": [],
        "helix_blocks": [],
        "support_edges": [],
        "limits": copy.deepcopy(spec["limits"]),
        "backbone_policy": copy.deepcopy(
            spec.get("backbone_policy", default_backbone_policy())
        ),
    }
    if "maximum_unsupported_run" not in contract["limits"]:
        raise ValueError(
            "Complete-scaffold v2 requires an explicit limits.maximum_unsupported_run; see COMPLETE_SCAFFOLD.zh-CN.md"
        )
    offset = 0
    for g in range(groups_count):
        for entity, pattern in enumerate(patterns):
            chain_id = _chain_name(g * len(patterns) + entity)
            chain = full[g * len(patterns) + entity]
            contract["residues"].extend(
                {
                    "chain_id": chain_id,
                    "residue_number": i + 1,
                    "fixed": pattern[i] is not None,
                    "reference_ca": np.asarray(r["atoms"]["CA"]).tolist(),
                    "reference_backbone": np.asarray(
                        [r["atoms"][a] for a in _BB]
                    ).tolist(),
                    "residue_name": r["name"],
                }
                for i, r in enumerate(chain)
            )
            for block in spec["helix_blocks"]:
                if (
                    set(block) != {"entity", "id", "start", "end"}
                    or type(block["entity"]) is not int
                    or not 0 <= block["entity"] < len(patterns)
                ):
                    raise ValueError(
                        "helix_blocks require entity, id and inclusive one-based start/end"
                    )
                if block["entity"] != entity:
                    continue
                if not (
                    type(block["start"]) is int
                    and type(block["end"]) is int
                    and 1 <= block["start"] < block["end"] <= len(pattern)
                ):
                    raise ValueError("Invalid helix block range")
                contract["helix_blocks"].append(
                    {
                        "id": f"copy{g}:{block['id']}",
                        "residue_indices": list(
                            range(offset + block["start"] - 1, offset + block["end"])
                        ),
                    }
                )
            offset += len(chain)
        contract["support_edges"].extend(
            {"left": f"copy{g}:{e['left']}", "right": f"copy{g}:{e['right']}"}
            for e in spec["support_edges"]
        )
    return validate_scaffold_contract(contract)


def prepare_scaffold_task(
    config: Path, blueprint: Path, output_directory: Path, *, progress=None
) -> dict:
    """Create one immutable task and explain every rejection without inference."""
    emit = progress or (lambda message: None)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared scaffold task: {output}")
    design = load_user_design(config)
    if design.sampling.scaffold_artifact is not None:
        raise ValueError(
            "Prepare from an ordinary design, not an already bound scaffold task"
        )
    if design.sampling.execution_backend != "explicit_all_copy":
        raise ValueError(
            "Complete scaffold requires explicit_all_copy; ordinary local modes remain unchanged"
        )
    blueprint = Path(blueprint).expanduser().resolve()
    spec = yaml.safe_load(blueprint.read_text())
    required = {
        "schema_version",
        "template",
        "chains",
        "partial_t",
        "limits",
        "helix_blocks",
        "support_edges",
        "maximum_template_seed_rmsd",
        "maximum_template_symmetry_rmsd",
    }
    optional = {
        "closure",
        "reference_generated_lengths",
        "secondary_structure",
        "reference_secondary_structure",
        "construction",
        "backbone_policy",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) - required - optional
        or required - set(spec)
        or spec["schema_version"] != 1
    ):
        raise ValueError(
            f"Scaffold blueprint requires {sorted(required)}; optional {sorted(optional)}"
        )
    for key in (
        "partial_t",
        "maximum_template_seed_rmsd",
        "maximum_template_symmetry_rmsd",
    ):
        _positive(spec[key], key)
    template = Path(spec["template"]).expanduser()
    if not template.is_absolute():
        template = blueprint.parent / template
    template = template.resolve()
    residues = _residues(template)
    template_chains = {}
    for residue in residues.values():
        if not set(_BB) <= set(residue["atoms"]):
            raise ValueError("Every template residue requires N/CA/C/O")
        template_chains.setdefault(residue["chain"], []).append(residue)
    with tempfile.TemporaryDirectory(prefix="mosaic-scaffold-") as temporary:
        work = Path(temporary)
        emit("Compile exact contig and fixed-seed graph")
        payload, compiled = _compile(design, work / "initial")
        native = next(iter(payload.values()))
        extra = native["extra"]
        if not re.fullmatch(r"[CD][2-9]\d*|[CD]1\d+", native["symmetry"]["id"]):
            raise ValueError(
                "Complete-scaffold preparation currently supports full Cn/Dn, n>=2"
            )
        if extra.get("symmetry_action_kind") != "regular_full_group" or extra.get(
            "preexpanded_chain_layout"
        ):
            raise ValueError(
                "Complete-scaffold preparation requires an ordinary full-group action"
            )
        if any(
            o.get("mobility_mode") not in {"fixed", "orbit_rigid"}
            for o in extra["motif_constraint_orbits"]
        ):
            raise ValueError(
                "Complete-scaffold preparation requires fixed or bounded rigid seed orbits"
            )
        patterns = _compiled_contig_chains(native["contig"])
        order = extra["registry_transform_order"]
        matrices = [
            np.asarray(extra["registry_transform_matrices"][key]) for key in order
        ]
        selections = spec["chains"]
        if (
            not isinstance(selections, list)
            or len(selections) != len(patterns)
            or any(
                not isinstance(row, list) or len(row) != len(order)
                for row in selections
            )
            or sorted(x for row in selections for x in row) != sorted(template_chains)
        ):
            raise ValueError(
                "chains must map each ASU polymer path to every registry copy and cover template chains exactly once"
            )
        reference = [
            [
                np.asarray(
                    [[r["atoms"][a] for a in _BB] for r in template_chains[name]]
                )
                for name in row
            ]
            for row in selections
        ]
        maps, runs = [], []
        for entity, pattern in enumerate(patterns):
            declared = spec.get("reference_generated_lengths")
            mapping, pairs = _layout(
                pattern,
                len(reference[entity][0]),
                None if declared is None else declared[entity],
            )
            maps.append(mapping)
            runs.append(pairs)
        from rfd3_mosaic.scaffold_pose import align_template_to_registry

        emit("Align complete template symmetry and fit whole joint seeds")
        aligned, symmetry_report = align_template_to_registry(
            reference,
            matrices,
            maximum_symmetry_rmsd=spec["maximum_template_symmetry_rmsd"],
        )
        if aligned is None:
            raise ValueError(
                f"Template symmetry alignment unresolved: {json.dumps(symmetry_report)}"
            )
        source = _residues(compiled / native["input"])
        # A reference sequence-fixed motif must be the same sequence, even
        # though its approximate coordinates are subsequently replaced.
        masked_reference = (
            set(_components(native["select_unfixed_sequence"]))
            if native.get("select_unfixed_sequence")
            else set()
        )
        for entity, mapping in enumerate(maps):
            for position, reference_position in mapping.items():
                if (
                    patterns[entity][position] not in masked_reference
                    and source[patterns[entity][position]]["name"]
                    != template_chains[selections[entity][0]][reference_position][
                        "name"
                    ]
                ):
                    raise ValueError(
                        "Template fixed-fragment sequence does not match compiled seed correspondence"
                    )
        frozen, pose_report = _freeze_fitted_poses(
            design,
            extra,
            source,
            patterns,
            maps,
            aligned,
            spec["maximum_template_seed_rmsd"],
        )
        payload, compiled = _compile(frozen, work / "fitted")
        native = next(iter(payload.values()))
        extra = native["extra"]
        if _compiled_contig_chains(native["contig"]) != patterns:
            raise ValueError(
                "Pose fitting changed the materialized contig; use explicit fixed lengths"
            )
        source = _residues(compiled / native["input"])
        closure_options = dict(spec.get("closure") or {})
        allowed = {
            "attempts",
            "max_nfev",
            "seed",
            "closure_tolerance",
            "bond_tolerance",
            "angle_tolerance_deg",
            "clash_distance",
            "reference_contact_cutoff",
            "reference_sequence_separation",
            "maximum_reference_contact_error",
            "reference_shape_weight",
            "maximum_reference_ca_deviation",
            "nonlocal_clash_weight",
            "clash_clearance_margin",
        }
        if set(closure_options) - allowed:
            raise ValueError(
                f"Unknown closure options {sorted(set(closure_options) - allowed)}"
            )
        from rfd3_mosaic.scaffold_assembly import solve_assembly

        def audit_candidate(candidate):
            try:
                expanded = _expand_asu(candidate, matrices)
                clearance = _full_backbone_clearance(
                    expanded, closure_options.get("clash_distance", 2.0)
                )
                _scaffold_contract(expanded, patterns, spec, len(matrices))
                return {"passed": True, "full_backbone_clearance": clearance}
            except ValueError as error:
                return {"passed": False, "reason": str(error)}

        emit(
            "Jointly solve seed rigid poses and all generated regions against the complete assembly"
        )
        asu, construction_report = solve_assembly(
            source=source,
            patterns=patterns,
            runs=runs,
            aligned=aligned,
            extra=extra,
            spec=spec,
            candidate_audit=audit_candidate,
        )
        if asu is None:
            raise ScaffoldConstructionUnresolved(-1, -1, construction_report)
        full = _expand_asu(asu, matrices)
        # Recompile the optimized entire seeds through the same public pose
        # representation; no independent movement of interface halves.
        solved_reference = [
            [
                np.asarray(
                    [
                        [r["atoms"][a] for a in _BB]
                        for r in full[g * len(patterns) + entity]
                    ]
                )
                for g in range(len(matrices))
            ]
            for entity in range(len(patterns))
        ]
        identity_maps = [
            {i: i for i, component in enumerate(p) if component is not None}
            for p in patterns
        ]
        frozen, solved_pose_report = _freeze_fitted_poses(
            frozen,
            extra,
            source,
            patterns,
            identity_maps,
            solved_reference,
            spec["maximum_template_seed_rmsd"],
        )
        payload, compiled = _compile(frozen, work / "solved")
        native = next(iter(payload.values()))
        extra = native["extra"]
        if _compiled_contig_chains(native["contig"]) != patterns:
            raise ValueError("Joint construction changed the materialized contig")
        clearance_report = _full_backbone_clearance(
            full, closure_options.get("clash_distance", 2.0)
        )
        remapped, components = _remap_extra(extra, patterns, len(matrices))
        remapped["mosaic_scaffold_contract"] = _scaffold_contract(
            full, patterns, spec, len(matrices)
        )
        closure_reports = construction_report["attempts"]
        structure = work / "complete_scaffold.cif"
        _write_cif(structure, full)
        fixed = {}
        for g in range(len(matrices)):
            for selector, atom_names in native["select_fixed_atoms"].items():
                fixed[",".join(components(_components(selector), g))] = atom_names
        masked_original = (
            set(_components(native["select_unfixed_sequence"]))
            if native.get("select_unfixed_sequence")
            else set()
        )
        unfixed = [
            f"{_chain_name(g * len(patterns) + entity)}{i + 1}"
            for g in range(len(matrices))
            for entity, pattern in enumerate(patterns)
            for i, component in enumerate(pattern)
            if component is None or component in masked_original
        ]
        symmetry = copy.deepcopy(native["symmetry"])
        symmetry.update(
            use_declared_frames=True,
            declared_transform_order=order,
            declared_transform_matrices=extra["registry_transform_matrices"],
            declared_preexpanded_chain_layout=remapped["preexpanded_chain_layout"],
        )
        if any(
            o["mobility_mode"] == "orbit_rigid"
            for o in remapped["motif_constraint_orbits"]
        ):
            remapped["mosaic_reference_transport"] = scaffold_transport_plan(
                {"select_fixed_atoms": fixed, "extra": remapped},
                extra,
                _residues(structure),
            )
        artifact = {
            "schema_version": 1,
            "compiled_contract_sha256": compiled_scaffold_contract_sha256(payload),
            "structure_path": structure.name,
            "structure_sha256": _sha(structure),
            "partial_t": spec["partial_t"],
            "native_input": {
                "contig": ",/0,".join(
                    f"{_chain_name(i)}1-{len(chain)}" for i, chain in enumerate(full)
                ),
                "select_fixed_atoms": fixed,
                "select_unfixed_sequence": ",".join(unfixed),
                "symmetry": symmetry,
                "extra": remapped,
            },
        }
        artifact_path = work / "scaffold_artifact.json"
        _json(artifact_path, artifact)
        emit(
            "Verify full scaffold, exact seed atoms, explicit masks and native symmetry handoff"
        )
        apply_scaffold_input(
            payload, artifact_path=artifact_path, output_directory=compiled
        )
        # Stage next to the destination so publishing the complete directory
        # is one filesystem rename. Interrupted writes never expose a task YAML.
        destination = output
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".mosaic-prepare-", dir=output.parent
        ) as publication:
            output = Path(publication) / "task"
            # Publish only after all independent construction/contract checks.
            output.mkdir()
            shutil.copyfile(structure, output / structure.name)
            shutil.copyfile(artifact_path, output / artifact_path.name)
            shutil.copyfile(blueprint, output / "source_blueprint.yaml")
            seed_name = "seed_input" + "".join(design.input.suffixes)
            shutil.copyfile(design.input, output / seed_name)
            template_name = "reference_template" + "".join(template.suffixes)
            shutil.copyfile(template, output / template_name)
            replay_blueprint = copy.deepcopy(spec)
            replay_blueprint["template"] = template_name
            (output / "blueprint.yaml").write_text(
                yaml.safe_dump(replay_blueprint, sort_keys=False)
            )
            task = frozen.model_dump(mode="json", exclude_none=True)
            task["input"] = seed_name
            task["sampling"]["scaffold_artifact"] = artifact_path.name
            task_path = output / "design.yaml"
            task_path.write_text(yaml.safe_dump(task, sort_keys=False))
            report = {
                "schema_version": 1,
                "status": "prepared",
                "task": str(destination / "design.yaml"),
                "template": str(template),
                "template_sha256": _sha(template),
                "source_design": str(Path(config).resolve()),
                "source_design_sha256": _sha(Path(config)),
                "structure_sha256": _sha(structure),
                "blueprint_sha256": _sha(blueprint),
                "symmetry_fit": symmetry_report,
                "joint_seed_fits": pose_report,
                "closure": closure_reports,
                "joint_construction": construction_report,
                "solved_seed_fits": solved_pose_report,
                "limits": spec["limits"],
                "partial_t": spec["partial_t"],
                "full_backbone_clearance": clearance_report,
                "designs_share_one_initial_pose": True,
                "gpu_jobs_submitted": 0,
                "scope": "Geometric scaffold witness and partial-diffusion input; not folding, coiled-coil identity, sequence designability or generation-yield certification",
            }
            _json(output / "preparation_report.json", report)
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite prepared scaffold task: {destination}"
                )
            output.rename(destination)
            return report
