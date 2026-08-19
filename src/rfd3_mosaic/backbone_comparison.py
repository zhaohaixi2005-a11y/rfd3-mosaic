"""Paper-aligned backbone comparison for interface-seeded campaigns.

The comparison intentionally stops at the diffusion/backbone stage.  It does
not turn missing sequence-design, refolding or experimental measurements into
backbone failures.  Every reported field therefore names both its source and
its comparison stage.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from rfd3_mosaic.rfd3_batch_screen import (
    _ca_coordinates_by_chain,
    cyclic_ring_descriptors,
    interchain_packing_descriptors,
)
from rfd3_mosaic.rfd3_seed_audit import _derive_structure_path
from rfd3_mosaic.run_index import read_run_record
from rfd3_mosaic.structure import AtomRecord, read_structure_atoms


PROTOCOL_ID = "hoyeung_lhd101_backbone_v1"
PAPER_DOI = "10.64898/2026.07.02.736098"
PAPER_URL = "https://www.biorxiv.org/content/10.64898/2026.07.02.736098v1"


@dataclass(frozen=True)
class ComparisonArtifacts:
    """Files written by one campaign comparison."""

    json_path: Path
    csv_path: Path
    markdown_path: Path


def paper_backbone_protocol() -> dict[str, Any]:
    """Return only claims explicitly reported by the paper and supplement."""

    return {
        "protocol_id": PROTOCOL_ID,
        "paper": {
            "title": (
                "A generalizable interface-seeded framework for de novo "
                "design of functional oligomers"
            ),
            "doi": PAPER_DOI,
            "url": PAPER_URL,
            "version_posted": "2026-07-03",
        },
        "matched_generation_case": {
            "interface_seed": "LHD101",
            "source_pdb": "7MWR",
            "symmetry": "C3",
            "diffusion_timesteps": 50,
            "generated_length_range": [70, 100],
            "diversity_analysis_backbones": 1000,
        },
        "paper_generation_method": {
            "initial_orientation": "sample full rotational space",
            "initial_radius": "sample user-defined range",
            "runtime_motion": "radial motif dragging after each diffusion step",
            "typical_backbones_per_objective": [5000, 10000],
        },
        "paper_backbone_screen": {
            "metrics": ["loop_percentage", "radius_of_gyration"],
            "selection": "lowest 50th percentile",
            "reported_fraction_remaining_approximately": 0.10,
            "warning": (
                "The supplement does not publish an absolute loop or Rg "
                "threshold, nor enough detail to reconstruct every additional "
                "structural filter. Mosaic reports cohort percentiles without "
                "inventing a cutoff."
            ),
        },
        "paper_diversity": {
            "method": "Foldseek easy-cluster",
            "definition": "number_of_unique_clusters / number_of_backbones",
            "reported_orientation_sampling_gain": ">13-fold",
            "raw_cluster_assignments_publicly_available": False,
        },
        "paper_orientation_descriptors": {
            "interface_residue_cutoff_angstrom": 6.0,
            "flatness": "interface-vector angle to the ring xy-plane",
            "twist": "chain-COM vector angle to the ring xz-plane",
        },
        "out_of_scope_for_this_report": [
            "SolubleMPNN sequence design",
            "AF2/AF3 monomer or multimer prediction",
            "Rosetta ddG, buried SASA and shape complementarity",
            "expression, SEC, SEC-MALS and structure determination",
        ],
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return value


def _local_path(path: str | Path, run_directory: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = run_directory / candidate
    if candidate.is_file():
        return candidate.resolve()
    local = run_directory / candidate.name
    return local.resolve() if local.is_file() else candidate.resolve()


def _run_directories(
    manifest: Mapping[str, Any],
    *,
    run_root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    directories: list[Path] = []
    unavailable: list[dict[str, Any]] = []
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Campaign manifest contains no shard records")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Every campaign shard record must be a mapping")
        job_id = record.get("job_id")
        if job_id is None:
            unavailable.append(
                {
                    "shard_index": record.get("shard_index"),
                    "reason": "not submitted or no job ID recorded",
                }
            )
            continue
        indexed = read_run_record(run_root, str(job_id))
        if indexed is None:
            unavailable.append(
                {"job_id": str(job_id), "reason": "run index is missing"}
            )
            continue
        run_value = indexed.get("run_directory")
        run_directory = Path(str(run_value)).expanduser().resolve()
        if not run_directory.is_dir():
            unavailable.append(
                {
                    "job_id": str(job_id),
                    "reason": "run directory is not available",
                    "run_directory": str(run_directory),
                }
            )
            continue
        directories.append(run_directory)
    return directories, unavailable


def _audit_payload(
    design: Mapping[str, Any],
    *,
    name: str,
    run_directory: Path,
) -> dict[str, Any] | None:
    for raw in design.get("reports") or []:
        path = _local_path(str(raw), run_directory)
        if path.name == name and path.is_file():
            return _load_mapping(path)
    return None


def _seed_metrics(
    design: Mapping[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    constraint = _audit_payload(
        design,
        name="constraint_orbit_audit.json",
        run_directory=run_directory,
    )
    if constraint is not None:
        summary = constraint.get("summary") or {}
        return {
            "audit": "constraint_orbit_audit.json",
            "passed": bool(constraint.get("passed")),
            "rmsd_angstrom": summary.get("maximum_acceptance_rmsd"),
            "atom_completeness": summary.get("atom_completeness"),
        }
    seed = _audit_payload(
        design,
        name="seed_integrity_audit.json",
        run_directory=run_directory,
    )
    if seed is not None:
        summary = seed.get("summary") or {}
        return {
            "audit": "seed_integrity_audit.json",
            "passed": bool(seed.get("passed")),
            "rmsd_angstrom": (
                summary.get("maximum_all_atom_rmsd")
                or summary.get("maximum_ca_rmsd")
            ),
            "atom_completeness": summary.get("minimum_atom_completeness"),
        }
    return {
        "audit": None,
        "passed": None,
        "rmsd_angstrom": None,
        "atom_completeness": None,
    }


def _heavy_coordinates_by_residue(
    atoms: Iterable[AtomRecord],
    chain_id: str,
) -> dict[tuple[int, str], np.ndarray]:
    grouped: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    for atom in atoms:
        element = atom.element.upper()
        atom_name = atom.atom_name.lstrip("0123456789").upper()
        if (
            atom.record_type != "ATOM"
            or atom.chain_id != chain_id
            or element.startswith("H")
            or atom_name.startswith("H")
        ):
            continue
        grouped.setdefault(
            (atom.residue_number, atom.insertion_code), []
        ).append(atom.coordinate)
    return {
        residue: np.asarray(coordinates, dtype=float)
        for residue, coordinates in grouped.items()
    }


def _ca_by_residue(
    atoms: Iterable[AtomRecord],
    chain_id: str,
) -> dict[tuple[int, str], np.ndarray]:
    return {
        (atom.residue_number, atom.insertion_code): np.asarray(
            atom.coordinate, dtype=float
        )
        for atom in atoms
        if atom.record_type == "ATOM"
        and atom.chain_id == chain_id
        and atom.atom_name.upper() == "CA"
    }


def flatness_twist_descriptors(
    atoms: tuple[AtomRecord, ...],
    ring: Mapping[str, Any],
    *,
    contact_cutoff: float = 6.0,
) -> dict[str, Any]:
    """Reproduce the paper's backbone-level Flatness/Twist definitions.

    The authors' claimed analysis repository is not currently public.  The
    implementation follows the written method and records that provenance in
    the result instead of claiming byte-identical reproduction of their code.
    """

    order = list(ring.get("angular_chain_order") or ())
    if not ring.get("available") or len(order) < 2:
        return {"available": False, "reason": "ring frame is unavailable"}
    left = min(order)
    index = order.index(left)
    right = order[(index + 1) % len(order)]
    left_heavy = _heavy_coordinates_by_residue(atoms, left)
    right_heavy = _heavy_coordinates_by_residue(atoms, right)
    left_ca = _ca_by_residue(atoms, left)
    right_ca = _ca_by_residue(atoms, right)
    contacting_left: set[tuple[int, str]] = set()
    contacting_right: set[tuple[int, str]] = set()
    for left_id, left_coordinates in left_heavy.items():
        for right_id, right_coordinates in right_heavy.items():
            minimum = float(
                np.min(
                    np.linalg.norm(
                        left_coordinates[:, None, :]
                        - right_coordinates[None, :, :],
                        axis=-1,
                    )
                )
            )
            if minimum <= contact_cutoff:
                contacting_left.add(left_id)
                contacting_right.add(right_id)
    left_points = np.asarray(
        [left_ca[item] for item in sorted(contacting_left) if item in left_ca],
        dtype=float,
    )
    right_points = np.asarray(
        [right_ca[item] for item in sorted(contacting_right) if item in right_ca],
        dtype=float,
    )
    if not len(left_points) or not len(right_points):
        return {
            "available": False,
            "reason": "no neighbouring interface residues within cutoff",
            "left_chain": left,
            "right_chain": right,
        }

    center = np.asarray(ring["center"], dtype=float)
    z_axis = np.asarray(ring["axis"], dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    left_center = np.mean(np.asarray(list(left_ca.values())), axis=0)
    right_center = np.mean(np.asarray(list(right_ca.values())), axis=0)
    x_axis = left_center - center
    x_axis -= np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    differences = right_points[None, :, :] - left_points[:, None, :]
    distances = np.linalg.norm(differences, axis=-1)
    pair_index = np.unravel_index(int(np.argmax(distances)), distances.shape)
    interface_vector = differences[pair_index]
    interface_local = np.asarray(
        [
            np.dot(interface_vector, x_axis),
            np.dot(interface_vector, y_axis),
            np.dot(interface_vector, z_axis),
        ]
    )
    com_vector = right_center - left_center
    com_local = np.asarray(
        [
            np.dot(com_vector, x_axis),
            np.dot(com_vector, y_axis),
            np.dot(com_vector, z_axis),
        ]
    )
    flatness = math.degrees(
        math.atan2(
            abs(float(interface_local[2])),
            float(np.linalg.norm(interface_local[:2])),
        )
    )
    twist = math.degrees(
        math.atan2(
            abs(float(com_local[1])),
            float(np.linalg.norm(com_local[[0, 2]])),
        )
    )
    return {
        "available": True,
        "method": "paper_written_definition_reimplemented",
        "contact_cutoff_angstrom": contact_cutoff,
        "left_chain": left,
        "right_chain": right,
        "left_interface_residues": len(left_points),
        "right_interface_residues": len(right_points),
        "flatness_degrees": flatness,
        "twist_degrees": twist,
    }


def _write_stride_pdb(
    path: Path,
    atoms: tuple[AtomRecord, ...],
    chain_id: str,
) -> None:
    lines = []
    serial = 1
    for atom in atoms:
        if atom.record_type != "ATOM" or atom.chain_id != chain_id:
            continue
        x, y, z = atom.coordinate
        element = (atom.element or atom.atom_name[:1]).upper()[:2]
        lines.append(
            f"ATOM  {serial:5d} {atom.atom_name:>4s} "
            f"{atom.residue_name:>3s} A{atom.residue_number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
            f"{element:>2s}\n"
        )
        serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines), encoding="utf-8")


def stride_loop_metrics(
    atoms: tuple[AtomRecord, ...],
    *,
    chain_id: str,
    stride_executable: str | None,
) -> dict[str, Any]:
    if stride_executable is None:
        return {
            "available": False,
            "reason": "STRIDE executable was not supplied",
        }
    executable = shutil.which(stride_executable) or (
        str(Path(stride_executable).resolve())
        if Path(stride_executable).is_file()
        else None
    )
    if executable is None:
        return {
            "available": False,
            "reason": f"STRIDE executable not found: {stride_executable}",
        }
    with tempfile.TemporaryDirectory(prefix="mosaic-stride-") as directory:
        pdb_path = Path(directory) / "asu.pdb"
        _write_stride_pdb(pdb_path, atoms, chain_id)
        completed = subprocess.run(
            [executable, str(pdb_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": (completed.stderr or completed.stdout).strip()
            or f"STRIDE exited {completed.returncode}",
        }
    counts = {key: 0 for key in ("H", "E", "C", "T", "G", "I", "B")}
    for line in completed.stdout.splitlines():
        if not line.startswith("ASG"):
            continue
        fields = line.split()
        if len(fields) > 5 and fields[5].upper() in counts:
            counts[fields[5].upper()] += 1
    total = sum(counts.values())
    if not total:
        return {"available": False, "reason": "STRIDE returned no assignments"}
    percentages = {key: 100.0 * value / total for key, value in counts.items()}
    return {
        "available": True,
        "residue_count": total,
        "class_percentages": percentages,
        "helix_percentage": sum(percentages[key] for key in ("H", "G", "I")),
        "extended_percentage": sum(percentages[key] for key in ("E", "B")),
        "loop_percentage": percentages["C"] + percentages["T"],
        "loop_definition": "STRIDE C (coil) + T (turn)",
    }


def _one_design(
    design: Mapping[str, Any],
    *,
    run_directory: Path,
    expected_order: int,
    stride_executable: str | None,
) -> dict[str, Any]:
    result_json = _local_path(str(design["result_json"]), run_directory)
    structure = _derive_structure_path(result_json)
    atoms = read_structure_atoms(structure)
    ca_by_chain = _ca_coordinates_by_chain(atoms)
    ring = cyclic_ring_descriptors(ca_by_chain, expected_order)
    packing = (
        interchain_packing_descriptors(
            ca_by_chain,
            list(ring["angular_chain_order"]),
        )
        if ring.get("available")
        else {"available": False, "reason": "ring descriptors unavailable"}
    )
    orientation = flatness_twist_descriptors(atoms, ring)
    representative_chain = (
        min(ring["angular_chain_order"])
        if ring.get("available")
        else min(ca_by_chain)
    )
    stride = stride_loop_metrics(
        atoms,
        chain_id=representative_chain,
        stride_executable=stride_executable,
    )
    scaffold = _audit_payload(
        design,
        name="scaffold_validity_audit.json",
        run_directory=run_directory,
    )
    scaffold_summary = (scaffold or {}).get("summary") or {}
    guidance = _audit_payload(
        design,
        name="graph_interface_guidance_audit.json",
        run_directory=run_directory,
    )
    guidance_summary = (guidance or {}).get("summary") or {}
    seed = _seed_metrics(design, run_directory)
    return {
        "job_id": run_directory.name,
        "design_index": design.get("design_index"),
        "design_id": design.get("design_id"),
        "structure": str(structure),
        "worker_accepted": bool(design.get("accepted")),
        "rejection_reason": design.get("rejection_reason"),
        "seed": seed,
        "scaffold_audit_passed": (
            bool(scaffold.get("passed")) if scaffold is not None else None
        ),
        "packing_guidance": {
            "audit": (
                "graph_interface_guidance_audit.json"
                if guidance is not None
                else None
            ),
            "passed": (
                bool(guidance.get("passed"))
                if guidance is not None
                else None
            ),
            "final_targets_satisfied": guidance_summary.get(
                "final_proxy_targets_satisfied"
            ),
            "final_metrics": guidance_summary.get(
                "final_packing_metrics"
            ),
        },
        "chain_break_count": scaffold_summary.get("chain_break_count"),
        "ca_clash_count": scaffold_summary.get("ca_clash_count"),
        "symmetry_coordinate_rmsd": scaffold_summary.get(
            "maximum_symmetry_coordinate_rmsd"
        ),
        "maximum_chain_ca_radius_of_gyration": scaffold_summary.get(
            "maximum_chain_ca_radius_of_gyration"
        ),
        "ring": ring,
        "packing": packing,
        "orientation": orientation,
        "secondary_structure": stride,
        "paper_backbone_filter": "pending_cohort_percentiles",
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _quantiles(values: Iterable[Any]) -> dict[str, float] | None:
    finite = [_number(value) for value in values]
    array = np.asarray([value for value in finite if value is not None])
    if not len(array):
        return None
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _apply_cohort_filter(records: list[dict[str, Any]]) -> dict[str, Any]:
    rg_values = [
        _number(record["maximum_chain_ca_radius_of_gyration"])
        for record in records
    ]
    loop_values = [
        _number(record["secondary_structure"].get("loop_percentage"))
        if record["secondary_structure"].get("available")
        else None
        for record in records
    ]
    finite_rg = [value for value in rg_values if value is not None]
    finite_loop = [value for value in loop_values if value is not None]
    rg_median = float(np.median(finite_rg)) if finite_rg else None
    loop_median = float(np.median(finite_loop)) if finite_loop else None
    for record, rg, loop in zip(records, rg_values, loop_values, strict=True):
        if rg_median is None:
            record["paper_backbone_filter"] = "unavailable"
        elif loop_median is None:
            record["paper_backbone_filter"] = (
                "rg_lowest_half_only" if rg <= rg_median else "rejected_by_rg"
            )
        else:
            record["paper_backbone_filter"] = (
                "passed_reported_two_metric_percentiles"
                if rg <= rg_median and loop <= loop_median
                else "rejected_by_reported_two_metric_percentiles"
            )
    return {
        "radius_of_gyration_median": rg_median,
        "loop_percentage_median": loop_median,
        "complete_two_metric_filter_available": loop_median is not None,
        "two_metric_pass_count": sum(
            record["paper_backbone_filter"]
            == "passed_reported_two_metric_percentiles"
            for record in records
        ),
        "rg_only_lowest_half_count": sum(
            record["paper_backbone_filter"] == "rg_lowest_half_only"
            for record in records
        ),
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    requested: int,
    produced: int,
    unavailable_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "requested_designs": requested,
        "produced_designs": produced,
        "analyzed_designs": len(records),
        "unavailable_shards": len(unavailable_shards),
        "generation_complete": produced == requested and len(records) == produced,
        "worker_accepted_count": sum(record["worker_accepted"] for record in records),
        "seed_preserved_count": sum(record["seed"]["passed"] is True for record in records),
        "scaffold_passed_count": sum(
            record["scaffold_audit_passed"] is True for record in records
        ),
        "packing_guidance_passed_count": sum(
            record["packing_guidance"]["passed"] is True
            for record in records
        ),
        "packing_targets_satisfied_count": sum(
            record["packing_guidance"]["final_targets_satisfied"] is True
            for record in records
        ),
        "continuous_count": sum(record["chain_break_count"] == 0 for record in records),
        "clash_free_count": sum(record["ca_clash_count"] == 0 for record in records),
        "radius_of_gyration": _quantiles(
            record["maximum_chain_ca_radius_of_gyration"] for record in records
        ),
        "seed_rmsd_angstrom": _quantiles(
            record["seed"]["rmsd_angstrom"] for record in records
        ),
        "neighbor_ca_contacts": _quantiles(
            record["packing"].get("mean_neighbor_ca_contacts")
            for record in records
        ),
        "flatness_degrees": _quantiles(
            record["orientation"].get("flatness_degrees")
            for record in records
        ),
        "twist_degrees": _quantiles(
            record["orientation"].get("twist_degrees") for record in records
        ),
        "loop_percentage": _quantiles(
            record["secondary_structure"].get("loop_percentage")
            for record in records
        ),
        "foldseek_diversity": {
            "available": False,
            "reason": (
                "Foldseek cluster assignments were not supplied. Run "
                "Foldseek easy-cluster on the accepted structures, then add "
                "clusters/structures to obtain the paper-defined fraction."
            ),
        },
    }


def _flat_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": record["job_id"],
        "design_index": record["design_index"],
        "design_id": record["design_id"],
        "worker_accepted": record["worker_accepted"],
        "seed_passed": record["seed"]["passed"],
        "seed_rmsd_angstrom": record["seed"]["rmsd_angstrom"],
        "scaffold_passed": record["scaffold_audit_passed"],
        "chain_breaks": record["chain_break_count"],
        "ca_clashes": record["ca_clash_count"],
        "max_chain_ca_rg": record["maximum_chain_ca_radius_of_gyration"],
        "mean_neighbor_ca_contacts": record["packing"].get(
            "mean_neighbor_ca_contacts"
        ),
        "flatness_degrees": record["orientation"].get("flatness_degrees"),
        "twist_degrees": record["orientation"].get("twist_degrees"),
        "loop_percentage": record["secondary_structure"].get("loop_percentage"),
        "paper_backbone_filter": record["paper_backbone_filter"],
        "structure": record["structure"],
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    paper_filter = payload["cohort_filter"]
    lines = [
        "# Ho-Yeung LHD101 backbone comparison",
        "",
        f"Protocol: `{PROTOCOL_ID}`  ",
        f"Paper: [{PAPER_DOI}]({PAPER_URL})",
        "",
        "This report compares only the RFdiffusion/RFD3 backbone-generation "
        "stage. Sequence design, refolding, Rosetta and experiments are not "
        "counted as failures.",
        "",
        "## Campaign outcome",
        "",
        f"- Requested: {summary['requested_designs']}",
        f"- Produced: {summary['produced_designs']}",
        f"- Analyzed: {summary['analyzed_designs']}",
        f"- Mosaic strict audit passes: {summary['worker_accepted_count']}",
        f"- Supplied seed preserved: {summary['seed_preserved_count']}",
        "- Generated-scaffold packing audit passes: "
        f"{summary['packing_guidance_passed_count']}",
        "- Generated-scaffold final packing targets satisfied: "
        f"{summary['packing_targets_satisfied_count']}",
        f"- Continuous backbones: {summary['continuous_count']}",
        f"- CA-clash-free backbones: {summary['clash_free_count']}",
        "",
        "## Paper-aligned backbone screen",
        "",
        "The paper retained the lowest 50th percentile by loop percentage "
        "and radius of gyration; it did not publish absolute cutoffs.",
        "",
        f"- Cohort Rg median: {paper_filter['radius_of_gyration_median']}",
        f"- Cohort loop median: {paper_filter['loop_percentage_median']}",
        "- Complete loop+Rg filter available: "
        f"{paper_filter['complete_two_metric_filter_available']}",
        "- Two-metric pass count: "
        f"{paper_filter['two_metric_pass_count']}",
        "",
        "## Fair-comparison limits",
        "",
        "- The authors report Foldseek diversity as clusters/backbones and "
        "a >13-fold gain from orientation sampling, but their raw cluster "
        "assignments are not in the preprint supplement.",
        "- The paper-linked `interface_seeded_oligomers` repository returned "
        "404 when this protocol was prepared; exact Flatness/Twist source "
        "code was therefore unavailable. Mosaic records a transparent "
        "reimplementation of the written definition.",
        "- ProteinMPNN/AF2/AF3/Rosetta and experimental success are outside "
        "this backbone-only report.",
        "",
    ]
    return "\n".join(lines)


def compare_hoyeung_backbone_campaign(
    campaign_manifest: str | Path,
    *,
    output_directory: str | Path,
    run_root: str | Path,
    stride_executable: str | None = None,
    expected_order: int = 3,
) -> ComparisonArtifacts:
    """Analyze all completed shards from a frozen Mosaic campaign manifest."""

    manifest_path = Path(campaign_manifest).expanduser().resolve()
    manifest = _load_mapping(manifest_path)
    requested = int(manifest.get("total_designs", 0))
    if requested < 1:
        raise ValueError("Campaign manifest has no positive total_designs")
    root = Path(run_root).expanduser().resolve()
    run_directories, unavailable = _run_directories(manifest, run_root=root)
    records: list[dict[str, Any]] = []
    produced = 0
    run_summaries: list[dict[str, Any]] = []
    for run_directory in run_directories:
        summary_path = run_directory / "experiment_summary.json"
        if not summary_path.is_file():
            unavailable.append(
                {
                    "job_id": run_directory.name,
                    "reason": "experiment summary is missing",
                }
            )
            continue
        summary = _load_mapping(summary_path)
        produced += int(summary.get("produced_designs") or 0)
        design_results = summary.get("design_results") or []
        if not isinstance(design_results, list):
            raise ValueError(f"design_results must be a list in {summary_path}")
        for design in design_results:
            if not isinstance(design, Mapping):
                raise ValueError(f"Invalid design result in {summary_path}")
            records.append(
                _one_design(
                    design,
                    run_directory=run_directory,
                    expected_order=expected_order,
                    stride_executable=stride_executable,
                )
            )
        run_summaries.append(
            {
                "job_id": run_directory.name,
                "run_directory": str(run_directory),
                "status": summary.get("status"),
                "requested_designs": summary.get("requested_designs"),
                "produced_designs": summary.get("produced_designs"),
                "accepted_designs": summary.get("accepted_designs"),
            }
        )
    cohort_filter = _apply_cohort_filter(records)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": paper_backbone_protocol(),
        "campaign_manifest": str(manifest_path),
        "run_root": str(root),
        "summary": _summary(
            records,
            requested=requested,
            produced=produced,
            unavailable_shards=unavailable,
        ),
        "cohort_filter": cohort_filter,
        "unavailable_shards": unavailable,
        "runs": run_summaries,
        "records": records,
    }
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "hoyeung_backbone_comparison.json"
    csv_path = output / "hoyeung_backbone_metrics.csv"
    markdown_path = output / "hoyeung_backbone_comparison.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat = [_flat_record(record) for record in records]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(flat[0]) if flat else ["design_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return ComparisonArtifacts(
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )


__all__ = [
    "ComparisonArtifacts",
    "PROTOCOL_ID",
    "compare_hoyeung_backbone_campaign",
    "flatness_twist_descriptors",
    "paper_backbone_protocol",
    "stride_loop_metrics",
]
