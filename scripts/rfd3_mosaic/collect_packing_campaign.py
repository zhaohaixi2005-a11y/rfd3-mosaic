#!/usr/bin/env python3
"""Collect locked/guided packing evidence from one frozen campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _run_directory(record: dict[str, Any], run_root: Path) -> Path | None:
    candidates = []
    if record.get("run_index"):
        candidates.append(Path(str(record["run_index"])))
    if record.get("job_id"):
        candidates.append(
            run_root / ".rfd3-mosaic" / "jobs" / f"{record['job_id']}.json"
        )
    for index_path in candidates:
        if not index_path.is_file():
            continue
        index = _load(index_path)
        raw = index.get("run_directory")
        if raw:
            directory = Path(str(raw))
            if directory.is_dir():
                return directory
    return None


def _first_metric(
    payload: dict[str, Any],
    *keys: str,
) -> Any:
    value: Any = payload.get("summary", payload)
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _minimum_reciprocal_edge_value(
    payload: dict[str, Any],
    left_key: str,
    right_key: str,
) -> int | None:
    """Return the worst physical-edge value across reciprocal sides."""

    left = payload.get(left_key)
    right = payload.get(right_key)
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    if not left or len(left) != len(right):
        return None
    return min(
        min(int(left_value), int(right_value))
        for left_value, right_value in zip(left, right, strict=True)
    )


def _result_records(run: Path) -> list[dict[str, Any]]:
    results = []
    audit_roots = sorted((run / "audits").glob("*"))
    for audit_root in audit_roots:
        if not audit_root.is_dir():
            continue
        graph_path = audit_root / "graph_interface_guidance_audit.json"
        relation_path = audit_root / "assembly_interface_relation_audit.json"
        scaffold_path = audit_root / "scaffold_validity_audit.json"
        graph = _load(graph_path) if graph_path.is_file() else {}
        relation = _load(relation_path) if relation_path.is_file() else {}
        scaffold = _load(scaffold_path) if scaffold_path.is_file() else {}
        screening_path = audit_root / "screening_advice.json"
        screening = _load(screening_path) if screening_path.is_file() else {}
        graph_summary = graph.get("summary", graph)
        final_proxy = graph_summary.get("final_packing_metrics") or {}
        relation_summary = relation.get("summary", relation)
        interfaces = relation.get("interfaces") or []
        posthoc_coverage = [
            min(
                int(item.get("contact_residue_count_left", 0)),
                int(item.get("contact_residue_count_right", 0)),
            )
            for item in interfaces
        ]
        posthoc_continuity = [
            min(
                int(
                    item.get(
                        "maximum_contiguous_contact_residues_left", 0
                    )
                ),
                int(
                    item.get(
                        "maximum_contiguous_contact_residues_right", 0
                    )
                ),
            )
            for item in interfaces
        ]
        results.append(
            {
                "result_id": audit_root.name,
                "generated": True,
                "runtime_contract_met": graph.get("passed"),
                "packing_targets_satisfied": graph_summary.get(
                    "quality_targets_satisfied",
                    graph_summary.get("final_proxy_targets_satisfied"),
                ),
                "posthoc_interface_targets_satisfied": relation.get(
                    "passed"
                ),
                "scaffold_checks_satisfied": scaffold.get("passed"),
                "contract_status": screening.get("contract_status"),
                "recommendation": screening.get("recommendation"),
                "satisfied_edges": relation_summary.get(
                    "satisfied_required_edge_instance_count"
                ),
                "required_edges": relation_summary.get(
                    "required_edge_instance_count"
                ),
                "packing_energy": _first_metric(
                    graph, "final_packing_metrics", "energy"
                ),
                "shape_loss": _first_metric(
                    graph, "final_packing_metrics", "shape"
                ),
                "minimum_edge_distance": _first_metric(
                    graph,
                    "final_packing_metrics",
                    "minimum_edge_distance",
                ),
                "runtime_ca_contact_residues_per_side": (
                    _minimum_reciprocal_edge_value(
                        final_proxy,
                        "covered_left_residues",
                        "covered_right_residues",
                    )
                ),
                "runtime_ca_contiguous_residues_per_side": (
                    _minimum_reciprocal_edge_value(
                        final_proxy,
                        "contiguous_left_residues",
                        "contiguous_right_residues",
                    )
                ),
                "posthoc_contact_residues_per_side": (
                    min(posthoc_coverage) if posthoc_coverage else None
                ),
                "posthoc_contiguous_residues_per_side": (
                    min(posthoc_continuity) if posthoc_continuity else None
                ),
                "hard_clashes": sum(
                    int(item.get("hard_clashes_below_2_0A", 0))
                    for item in interfaces
                ),
            }
        )
    return results


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Interface packing campaign summary",
        "",
        f"- revision: `{summary.get('git_revision')}`",
        f"- requested outputs: {summary['requested_output_count']}",
        f"- generated outputs: {summary['generated_output_count']}",
        "- runtime contracts met: "
        f"{summary['runtime_contract_met_output_count']}",
        "- advisory packing targets satisfied: "
        f"{summary['packing_targets_satisfied_output_count']}",
        f"- recommended for next stage: {summary['recommended_output_count']}",
        f"- review advised: {summary['review_output_count']}",
        "",
        "The CA-window columns are the differentiable runtime objective. The "
        "post-hoc columns are a stricter backbone-heavy-atom observation. "
        "Neither is an experimental success/failure verdict.",
        "",
        "| mode | job/run | result | contract | advice | edges | runtime CA coverage | runtime CA contiguous | post-hoc coverage | post-hoc contiguous | shape | min edge (A) | hard clashes |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary["records"]:
        identity = record.get("job_id") or Path(
            str(record.get("run_directory") or "pending")
        ).name
        if not record["results"]:
            lines.append(
                f"| {record['mode']} | {identity} | pending | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        for result in record["results"]:
            edges = f"{result['satisfied_edges']}/{result['required_edges']}"
            lines.append(
                f"| {record['mode']} | {identity} | {result['result_id']} | "
                f"{result['contract_status'] or '-'} | "
                f"{result['recommendation'] or '-'} | {edges} | "
                f"{result['runtime_ca_contact_residues_per_side']} | "
                f"{result['runtime_ca_contiguous_residues_per_side']} | "
                f"{result['posthoc_contact_residues_per_side']} | "
                f"{result['posthoc_contiguous_residues_per_side']} | "
                f"{result['shape_loss']} | {result['minimum_edge_distance']} | "
                f"{result['hard_clashes']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run-root", type=Path)
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.expanduser().resolve()
    campaign = _load(manifest_path)
    run_root = (
        arguments.run_root.expanduser().resolve()
        if arguments.run_root is not None
        else Path(str(campaign["run_root"]))
    )
    records = []
    for source in campaign.get("records", []):
        run = _run_directory(source, run_root)
        records.append(
            {
                "mode": source.get("mode"),
                "seed": source.get("seed"),
                "profile": source.get("profile"),
                "job_id": source.get("job_id"),
                "run_directory": str(run) if run is not None else None,
                "results": _result_records(run) if run is not None else [],
            }
        )
    observed = sum(len(record["results"]) for record in records)
    runtime_contract_met = sum(
        result["runtime_contract_met"] is True
        for record in records
        for result in record["results"]
    )
    packing_targets_satisfied = sum(
        result["packing_targets_satisfied"] is True
        for record in records
        for result in record["results"]
    )
    recommended = sum(
        result["recommendation"] == "recommended_for_next_stage"
        for record in records
        for result in record["results"]
    )
    reviewed = sum(
        result["recommendation"]
        in {"review_contract", "review_advisory_metrics"}
        for record in records
        for result in record["results"]
    )
    summary = {
        "schema_version": 2,
        "git_revision": campaign.get("git_revision"),
        "campaign_manifest": str(manifest_path),
        "requested_output_count": campaign.get("requested_output_count", 0),
        "generated_output_count": observed,
        "runtime_contract_met_output_count": runtime_contract_met,
        "packing_targets_satisfied_output_count": (
            packing_targets_satisfied
        ),
        "recommended_output_count": recommended,
        "review_output_count": reviewed,
        "complete": observed == campaign.get("requested_output_count", 0),
        "records": records,
    }
    json_path = manifest_path.parent / "packing_campaign_summary.json"
    markdown_path = manifest_path.parent / "packing_campaign_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"summary JSON: {json_path}")
    print(f"summary Markdown: {markdown_path}")
    print(
        f"generated outputs: {observed}/{summary['requested_output_count']}; "
        f"runtime contracts met: {runtime_contract_met}; "
        f"advisory packing targets satisfied: {packing_targets_satisfied}; "
        f"recommended: {recommended}; review: {reviewed}"
    )


if __name__ == "__main__":
    main()
