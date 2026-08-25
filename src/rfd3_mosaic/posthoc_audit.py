"""Re-audit an existing RFD3-Mosaic result without rerunning diffusion."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rfd3_mosaic.advisory_screening import write_advisory_screening
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


def _materialize_result_compiled_input(
    *,
    merged_input: Path,
    result_json: Path,
    run_directory: Path,
) -> Path:
    """Return one exact compiled example for a multi-example result.

    Early multi-input Mosaic runs retained only the merged engine input.  A
    result filename embeds its RFD3 example id, so post-hoc audit can recover
    the immutable one-example contract without rerunning inference.
    """

    payload = json.loads(merged_input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected compiled RFD3 examples in {merged_input}")
    if len(payload) == 1:
        return merged_input
    matching = [
        example_id
        for example_id in payload
        if example_id in result_json.stem
    ]
    if len(matching) != 1:
        raise ValueError(
            "Cannot map result to exactly one compiled RFD3 example: "
            f"result={result_json.name}, matches={matching}"
        )
    example_id = matching[0]
    example = payload[example_id]
    if not isinstance(example, dict):
        raise ValueError(
            f"Compiled RFD3 example {example_id!r} is not an object"
        )
    destination_directory = run_directory / "input" / "posthoc_examples"
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{example_id}.json"
    destination.write_text(
        json.dumps({example_id: example}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


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
    reuse_reports: bool = False,
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
        recorded = compiled_by_result.get(str(result_json))
        if recorded is not None and recorded.is_file():
            return recorded
        return _materialize_result_compiled_input(
            merged_input=default_input,
            result_json=result_json,
            run_directory=root,
        )

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
    check_flags: list[str] = []
    screening_paths: list[Path] = []
    reaudited_designs: list[dict[str, Any]] = []
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
            output_directory = report_directory(result_json)
            if reuse_reports:
                outcome_reports = tuple(
                    output_directory / report_name
                    for report_name in (
                        *(audit.report_name for audit in audits),
                        "scaffold_validity_audit.json",
                    )
                )
                missing = [path for path in outcome_reports if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "Cannot reuse missing result audit reports: "
                        + ", ".join(str(path) for path in missing)
                    )
                trajectory = output_directory / "mobility_trajectory.json"
                outcome_mobility = trajectory if trajectory.is_file() else None
            else:
                outcome = run_result_audits(
                    run_directory=root,
                    rfd3_input=input_path,
                    result_json=result_json,
                    semantic_audits=audits,
                    output_directory=output_directory,
                    python=python,
                )
                outcome_reports = outcome.reports
                outcome_mobility = outcome.mobility_trajectory
            reports_list.extend(outcome_reports)
            if outcome_mobility is not None:
                mobility_paths.append(outcome_mobility)
            required_audits_met = True
            rejection_reason = None
            try:
                gate_result_audits(outcome_reports, python=python)
            except RuntimeError as error:
                # A generated coordinate file is not an execution failure.
                # Preserve the legacy aggregate check while the advisory
                # record distinguishes contracts from scientific proxies.
                check_flags.append(str(error))
                required_audits_met = False
                rejection_reason = str(error)
            screening = config.get("sampling", {}).get("screening") or {}
            screening_path = (
                report_directory(result_json) / "screening_advice.json"
            )
            screening_payload = write_advisory_screening(
                screening_path,
                outcome_reports,
                mode=str(screening.get("mode", "advisory")),
                protocol=str(screening.get("protocol", "auto")),
            )
            screening_paths.append(screening_path)
            reaudited_designs.append(
                {
                    "result_json": str(result_json),
                    "compiled_input": str(input_path),
                    "reports": [str(path) for path in outcome_reports],
                    "mobility_trajectory": (
                        str(outcome_mobility)
                        if outcome_mobility is not None
                        else None
                    ),
                    "screening_advice": str(screening_path),
                    "screening": screening_payload,
                    "accepted": required_audits_met,
                    "rejection_reason": rejection_reason,
                }
            )
        except Exception as error:  # Audit execution itself remains strict.
            failures.append(error)

    reports = (
        tuple(reports_list)
        if reports_list
        else tuple(planned_reports_list)
    )
    mobility = mobility_paths[0] if len(mobility_paths) == 1 else None
    failure = failures[0] if failures else None
    execution_completed = failure is None
    passed = bool(execution_completed and not check_flags)
    summary = dict(previous)
    prior_status = previous.get("status")
    prior_error = previous.get("error")
    prior_error_type = previous.get("error_type")
    if execution_completed:
        previous_designs = [
            dict(record)
            for record in previous.get("design_results", [])
            if isinstance(record, dict)
        ]

        def previous_design_for(
            result_json: str,
            design_index: int,
        ) -> dict[str, Any]:
            exact = [
                record
                for record in previous_designs
                if str(record.get("result_json")) == result_json
            ]
            if len(exact) == 1:
                return exact[0]
            result_name = Path(result_json).name
            by_name = [
                record
                for record in previous_designs
                if Path(str(record.get("result_json", ""))).name
                == result_name
            ]
            if len(by_name) == 1:
                return by_name[0]
            return {
                "design_index": design_index,
                "design_id": Path(result_json).stem.removesuffix("_model_0"),
            }

        refreshed_designs: list[dict[str, Any]] = []
        for design_index, audited in enumerate(reaudited_designs):
            result_json = str(audited["result_json"])
            record = previous_design_for(result_json, design_index)
            screening_payload = audited["screening"]
            contract_met = screening_payload["contract_status"] in {
                "met",
                "not_evaluated",
            }
            record.update(
                {
                    "result_json": result_json,
                    "compiled_input": audited["compiled_input"],
                    "generated": True,
                    "contract_met": contract_met,
                    "recommendation": screening_payload["recommendation"],
                    "screening_advice": audited["screening_advice"],
                    "accepted": audited["accepted"],
                    "rejection_reason": audited["rejection_reason"],
                    "reports": audited["reports"],
                    "mobility_trajectory": audited["mobility_trajectory"],
                }
            )
            refreshed_designs.append(record)

        contract_met_count = sum(
            bool(record["contract_met"]) for record in refreshed_designs
        )
        accepted_count = sum(
            bool(record["accepted"]) for record in refreshed_designs
        )
        recommended_count = sum(
            record["recommendation"] == "recommended_for_next_stage"
            for record in refreshed_designs
        )
        summary.update(
            {
                "produced_designs": len(refreshed_designs),
                "generated_designs": len(refreshed_designs),
                "contract_met_designs": contract_met_count,
                "contract_flagged_designs": (
                    len(refreshed_designs) - contract_met_count
                ),
                "recommended_designs": recommended_count,
                "review_designs": len(refreshed_designs) - recommended_count,
                "accepted_designs": accepted_count,
                "rejected_designs": len(refreshed_designs) - accepted_count,
                "design_results": refreshed_designs,
            }
        )
    prior_posthoc = previous.get("posthoc_audit")
    prior_execution_status = prior_status
    if (
        prior_status == "failed"
        and isinstance(prior_posthoc, dict)
        and prior_posthoc.get("execution_completed") is False
    ):
        recovered = prior_posthoc.get("previous_worker_status")
        if recovered in {"completed", "running", "failed"}:
            prior_execution_status = recovered
    posthoc_record = {
        "schema_version": 2,
        "started_at": started_at,
        "completed_at": utc_now(),
        "passed": passed,
        "execution_completed": execution_completed,
        "semantics": "advisory_non_destructive",
        "check_flags": check_flags,
        "screening_advice": [str(path) for path in screening_paths],
        "reports": [str(path) for path in reports],
        "inference_rerun": False,
        "reports_reused": reuse_reports,
        "result_count": len(result_jsons),
        "previous_worker_status": prior_execution_status,
        "previous_error_type": prior_error_type,
        "previous_error": prior_error,
        "audit_error_type": type(failure).__name__ if failure else None,
        "audit_error": str(failure) if failure else None,
    }
    if execution_completed:
        summary.update(
            {
                "status": "completed",
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
                "posthoc_audit": posthoc_record,
            }
        )
        summary.pop("error", None)
        summary.pop("error_type", None)
    else:
        # Audit execution is diagnostic work performed after inference.  A
        # killed or resource-limited audit must not rewrite a previously
        # generated run as an inference failure or discard its last valid
        # report set and design counters.
        summary.update(
            {
                "status": prior_execution_status or "completed",
                "experiment": config.get("name")
                or previous.get("experiment"),
                "topology": (config.get("topology") or {}).get("kind"),
                "posthoc_audit": posthoc_record,
            }
        )
        if summary["status"] == "completed":
            summary.pop("error", None)
            summary.pop("error_type", None)
    _write_json(summary_path, summary)
    indexed_state = str(summary["status"])
    if indexed_state not in {"completed", "running", "failed"}:
        indexed_state = "completed" if result_jsons else "failed"
    _update_index(
        config,
        root,
        state=indexed_state,
        error=(
            str(failure)
            if failure is not None and indexed_state == "failed"
            else None
        ),
    )
    return PosthocAuditResult(
        run_directory=root,
        passed=passed,
        reports=reports,
        result_json=result_jsons[0],
        result_jsons=result_jsons,
        error=None if execution_completed else str(failure),
    )


__all__ = ["PosthocAuditResult", "audit_existing_run"]
