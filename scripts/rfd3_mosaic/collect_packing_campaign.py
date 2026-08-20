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
        relation_summary = relation.get("summary", relation)
        interfaces = relation.get("interfaces") or []
        results.append(
            {
                "result_id": audit_root.name,
                "accepted": bool(
                    graph.get("passed")
                    and relation.get("passed")
                    and scaffold.get("passed")
                ),
                "graph_guidance_passed": graph.get("passed"),
                "interface_relation_passed": relation.get("passed"),
                "scaffold_passed": scaffold.get("passed"),
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
                "maximum_contact_residues_per_side": max(
                    (
                        min(
                            int(item.get("contact_residue_count_left", 0)),
                            int(item.get("contact_residue_count_right", 0)),
                        )
                        for item in interfaces
                    ),
                    default=0,
                ),
                "maximum_contiguous_residues_per_side": max(
                    (
                        min(
                            int(
                                item.get(
                                    "maximum_contiguous_contact_residues_left",
                                    0,
                                )
                            ),
                            int(
                                item.get(
                                    "maximum_contiguous_contact_residues_right",
                                    0,
                                )
                            ),
                        )
                        for item in interfaces
                    ),
                    default=0,
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
        f"- observed outputs: {summary['observed_output_count']}",
        f"- scientifically accepted: {summary['accepted_output_count']}",
        "",
        "| mode | job/run | result | accepted | edges | coverage | contiguous | shape | min edge (A) | hard clashes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary["records"]:
        identity = record.get("job_id") or Path(
            str(record.get("run_directory") or "pending")
        ).name
        if not record["results"]:
            lines.append(
                f"| {record['mode']} | {identity} | pending | no | - | - | - | - | - | - |"
            )
            continue
        for result in record["results"]:
            edges = f"{result['satisfied_edges']}/{result['required_edges']}"
            lines.append(
                f"| {record['mode']} | {identity} | {result['result_id']} | "
                f"{'yes' if result['accepted'] else 'no'} | {edges} | "
                f"{result['maximum_contact_residues_per_side']} | "
                f"{result['maximum_contiguous_residues_per_side']} | "
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
    accepted = sum(
        result["accepted"]
        for record in records
        for result in record["results"]
    )
    summary = {
        "schema_version": 1,
        "git_revision": campaign.get("git_revision"),
        "campaign_manifest": str(manifest_path),
        "requested_output_count": campaign.get("requested_output_count", 0),
        "observed_output_count": observed,
        "accepted_output_count": accepted,
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
        f"outputs: {observed}/{summary['requested_output_count']}; "
        f"scientifically accepted: {accepted}"
    )


if __name__ == "__main__":
    main()
