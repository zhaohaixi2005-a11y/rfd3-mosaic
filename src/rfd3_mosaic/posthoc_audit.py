"""Re-audit an existing RFD3-Mosaic result without rerunning diffusion."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rfd3_mosaic.result_auditing import (
    find_compiled_input,
    find_result_jsons,
    gate_result_audits,
    infer_existing_run_audits,
    run_result_audits,
    utc_now,
)
from rfd3_mosaic.run_index import update_run_state


@dataclass(frozen=True)
class PosthocAuditResult:
    run_directory: Path
    passed: bool
    reports: tuple[Path, ...]
    result_json: Path
    result_jsons: tuple[Path, ...]
    error: str | None = None


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _update_index(
    config: dict[str, Any],
    run_directory: Path,
    *,
    state: str,
    error: str | None,
) -> None:
    try:
        output = config["output"]
        update_run_state(
            root=output["root"],
            job_id=run_directory.name,
            state=state,
            experiment=str(config["name"]),
            campaign=str(output["campaign"]),
            run_directory=run_directory,
            error=error,
        )
    except (KeyError, OSError, TypeError, ValueError) as index_error:
        print(
            "WARNING: could not update the RFD3-Mosaic run index: "
            f"{index_error}",
            flush=True,
        )


def audit_existing_run(
    run_directory: str | Path,
    *,
    python: str = sys.executable,
) -> PosthocAuditResult:
    """Reconstruct and execute the complete frozen post-inference audit set.

    Existing report files are overwritten intentionally: this command applies
    the currently installed audit implementation to immutable run inputs and
    model outputs.  RFD3 inference is never invoked.
    """

    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    config_path = root / "resolved_config.yaml"
    summary_path = root / "experiment_summary.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Required frozen run artifact is missing: {config_path}"
        )
    config = _load_mapping(config_path)
    result_jsons = find_result_jsons(root)
    previous: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot read existing worker summary {summary_path}: {error}"
            ) from error
        if not isinstance(previous, dict):
            raise ValueError(f"Expected a JSON mapping in {summary_path}")

    compiled_by_result = {
        str(record.get("result_json")): Path(str(record["compiled_input"]))
        for record in previous.get("design_results", [])
        if isinstance(record, dict) and record.get("compiled_input")
    }
    default_input = find_compiled_input(root)

    def input_for_result(result_json: Path) -> Path:
        return compiled_by_result.get(str(result_json), default_input)

    def report_directory(result_json: Path) -> Path:
        if len(result_jsons) == 1:
            return root
        design_id = result_json.stem.removesuffix("_model_0")
        return root / "audits" / design_id

    started_at = utc_now()
    reports_list: list[Path] = []
    planned_reports_list: list[Path] = []
    mobility_paths: list[Path] = []
    failures: list[Exception] = []
    for result_json in result_jsons:
        try:
            input_path = input_for_result(result_json)
            audits = infer_existing_run_audits(
                run_directory=root,
                rfd3_input=input_path,
                resolved_config=config,
            )
            planned_reports_list.extend(
                report_directory(result_json) / report_name
                for report_name in (
                    *(audit.report_name for audit in audits),
                    "scaffold_validity_audit.json",
                )
            )
            outcome = run_result_audits(
                run_directory=root,
                rfd3_input=input_path,
                result_json=result_json,
                semantic_audits=audits,
                output_directory=report_directory(result_json),
                python=python,
            )
            reports_list.extend(outcome.reports)
            if outcome.mobility_trajectory is not None:
                mobility_paths.append(outcome.mobility_trajectory)
            gate_result_audits(outcome.reports, python=python)
        except Exception as error:  # Preserve a complete fail-closed record.
            failures.append(error)

    reports = (
        tuple(reports_list)
        if reports_list
        else tuple(planned_reports_list)
    )
    mobility = mobility_paths[0] if len(mobility_paths) == 1 else None
    failure = failures[0] if failures else None
    passed = failure is None
    summary = dict(previous)
    prior_status = previous.get("status")
    prior_error = previous.get("error")
    prior_error_type = previous.get("error_type")
    summary.update(
        {
            "status": "completed" if passed else "failed",
            "experiment": config.get("name")
            or previous.get("experiment"),
            "topology": (config.get("topology") or {}).get("kind"),
            "result_json": (
                str(result_jsons[0]) if len(result_jsons) == 1 else None
            ),
            "result_jsons": [str(path) for path in result_jsons],
            "reports": [str(path) for path in reports],
            "mobility_trajectory": str(mobility) if mobility else None,
            "mobility_trajectories": [
                str(path) for path in mobility_paths
            ],
            "posthoc_audit": {
                "schema_version": 1,
                "started_at": started_at,
                "completed_at": utc_now(),
                "passed": passed,
                "reports": [str(path) for path in reports],
                "inference_rerun": False,
                "result_count": len(result_jsons),
                "previous_worker_status": prior_status,
                "previous_error_type": prior_error_type,
                "previous_error": prior_error,
            },
        }
    )
    if passed:
        summary.pop("error", None)
        summary.pop("error_type", None)
    else:
        summary["error_type"] = type(failure).__name__
        summary["error"] = str(failure)
    _write_json(summary_path, summary)
    _update_index(
        config,
        root,
        state="completed" if passed else "failed",
        error=None if passed else str(failure),
    )
    return PosthocAuditResult(
        run_directory=root,
        passed=passed,
        reports=reports,
        result_json=result_jsons[0],
        result_jsons=result_jsons,
        error=None if passed else str(failure),
    )


__all__ = ["PosthocAuditResult", "audit_existing_run"]
