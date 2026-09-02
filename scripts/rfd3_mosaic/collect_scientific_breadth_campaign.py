#!/usr/bin/env python3
"""Collect job/run locations for one scientific-breadth campaign.

The collector is deliberately non-destructive. It reads the persistent job
index, counts generated coordinate outputs and audits, and writes a stable
transfer list. Running it repeatedly while jobs are queued or active updates
only files beside the campaign manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _coordinate_outputs(run: Path) -> tuple[Path, ...]:
    mirror = run / "generated_structures_cif"
    if mirror.is_dir():
        candidates = (*mirror.glob("*.cif"), *mirror.glob("*.cif.gz"))
        by_structure = {
            path.name.removesuffix(".gz"): path for path in sorted(candidates)
        }
        return tuple(by_structure[name] for name in sorted(by_structure))
    return tuple(
        sorted(
            path
            for path in (*run.glob("*model_0.cif"), *run.glob("*model_0.cif.gz"))
            if "denoised" not in path.name and "noisy" not in path.name
        )
    )


def _indexed_run(
    record: dict[str, Any], run_root: Path
) -> tuple[dict[str, Any], Path | None]:
    job_id = record.get("job_id")
    if not job_id:
        return {}, None
    index_path = run_root / ".rfd3-mosaic" / "jobs" / f"{job_id}.json"
    if not index_path.is_file():
        return {}, None
    index = _load(index_path)
    raw_run = index.get("run_directory")
    run = Path(str(raw_run)).resolve() if raw_run else None
    return index, run if run is not None and run.is_dir() else None


def _audit_results(run: Path | None) -> tuple[dict[str, Any], ...]:
    """Summarize every design-level audit bundle without changing it."""

    if run is None or not (run / "audits").is_dir():
        return ()
    results = []
    for root in sorted((run / "audits").iterdir()):
        if not root.is_dir():
            continue
        audit_paths = sorted(root.glob("*_audit.json"))
        audit_states = []
        for path in audit_paths:
            payload = _load(path)
            audit_states.append(
                {
                    "name": path.stem,
                    "passed": payload.get("passed"),
                }
            )
        screening_path = root / "screening_advice.json"
        screening = _load(screening_path) if screening_path.is_file() else {}
        failed_audits = [
            item["name"] for item in audit_states if item["passed"] is False
        ]
        results.append(
            {
                "result_id": root.name,
                "audit_count": len(audit_states),
                "all_reported_audits_passed": bool(audit_states)
                and not failed_audits
                and all(item["passed"] is True for item in audit_states),
                "failed_audits": failed_audits,
                "contract_status": screening.get("contract_status"),
                "recommendation": screening.get("recommendation"),
                "audits": audit_states,
            }
        )
    return tuple(results)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Scientific breadth campaign collection status",
        "",
        f"- campaign: `{summary['campaign']}`",
        f"- source revision: `{summary.get('source_revision')}`",
        f"- jobs located: {summary['located_job_count']}/{summary['job_count']}",
        f"- outputs observed: {summary['observed_output_count']}/{summary['requested_output_count']}",
        f"- audit directories observed: {summary['audit_directory_count']}",
        f"- complete audit bundles passing: {summary['audit_passed_output_count']}",
        f"- screening contracts met: {summary['contract_met_output_count']}",
        f"- recommended for next stage: {summary['recommended_output_count']}",
        "",
        "| Case | Job | Scheduler/index state | Outputs | Requested | Audits | Run directory |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in summary["records"]:
        lines.append(
            f"| {record['case']} | {record.get('job_id') or '-'} | "
            f"{record.get('state') or 'not-indexed'} | "
            f"{record['observed_outputs']} | {record['requested_designs']} | "
            f"{record['audit_directories']} | "
            f"`{record.get('run_directory') or '-'}` |"
        )
    lines.extend(
        [
            "",
            "`transfer_paths.txt` contains the absolute campaign manifest and "
            "every located run directory. Re-run this collector after the jobs "
            "finish before transferring data.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load(manifest_path)
    run_root = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else Path(str(manifest["run_root"])).expanduser().resolve()
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else manifest_path.parent
    )
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    # The campaign directory contains the immutable manifest and frozen YAMLs.
    transfer_paths = {manifest_path.parent}
    for source in manifest.get("records", []):
        index, run = _indexed_run(source, run_root)
        outputs = _coordinate_outputs(run) if run is not None else ()
        audit_results = _audit_results(run)
        audit_count = len(audit_results)
        if run is not None:
            transfer_paths.add(run)
        records.append(
            {
                "case": source.get("case"),
                "job_id": source.get("job_id"),
                "state": index.get("state"),
                "requested_designs": int(source.get("requested_designs", 0)),
                "observed_outputs": len(outputs),
                "audit_directories": audit_count,
                "audit_passed_outputs": sum(
                    result["all_reported_audits_passed"] for result in audit_results
                ),
                "contract_met_outputs": sum(
                    result["contract_status"] == "met" for result in audit_results
                ),
                "recommended_outputs": sum(
                    result["recommendation"] == "recommended_for_next_stage"
                    for result in audit_results
                ),
                "run_directory": str(run) if run is not None else None,
                "results": audit_results,
            }
        )

    summary = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "campaign": manifest.get("campaign"),
        "source_revision": manifest.get("source_revision"),
        "run_root": str(run_root),
        "job_count": len(records),
        "located_job_count": sum(
            record["run_directory"] is not None for record in records
        ),
        "requested_output_count": sum(
            record["requested_designs"] for record in records
        ),
        "observed_output_count": sum(record["observed_outputs"] for record in records),
        "audit_directory_count": sum(record["audit_directories"] for record in records),
        "audit_passed_output_count": sum(
            record["audit_passed_outputs"] for record in records
        ),
        "contract_met_output_count": sum(
            record["contract_met_outputs"] for record in records
        ),
        "recommended_output_count": sum(
            record["recommended_outputs"] for record in records
        ),
        "complete": (
            sum(record["observed_outputs"] for record in records)
            == sum(record["requested_designs"] for record in records)
        ),
        "records": records,
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "COLLECTION_STATUS.md").write_text(_markdown(summary), encoding="utf-8")
    (output / "transfer_paths.txt").write_text(
        "\n".join(str(path) for path in sorted(transfer_paths)) + "\n",
        encoding="utf-8",
    )
    print(f"collection: {output}")
    print(
        "outputs: "
        f"{summary['observed_output_count']}/"
        f"{summary['requested_output_count']}"
    )


if __name__ == "__main__":
    main()
