"""Evidence-scoped, non-destructive advice for generated backbones.

The module deliberately does not define a universal protein-design score.
It reads existing Mosaic audits, separates execution/geometry contracts from
task-dependent quality observations, and writes an explanation that keeps
every generated coordinate file available to the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

_HARD_REPORTS = frozenset(
    {
        "constraint_orbit_audit.json",
        "seed_integrity_audit.json",
        "component_mobility_audit.json",
        "cylindrical_coordinate_audit.json",
    }
)
_ADVISORY_REPORTS = frozenset(
    {
        "assembly_interface_relation_audit.json",
    }
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _flag(
    *,
    code: str,
    report: Path,
    message: str,
    observed: Any = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "code": code,
        "report": str(report),
        "message": message,
    }
    if observed is not None:
        record["observed"] = observed
    return record


def _false_summary_flags(
    summary: Mapping[str, Any],
    keys: Iterable[str],
    *,
    report: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    return [
        _flag(
            code=f"{prefix}.{key}",
            report=report,
            message=f"Audit diagnostic {key} was not satisfied.",
            observed=summary.get(key),
        )
        for key in keys
        if key in summary and summary.get(key) is not True
    ]


def build_advisory_screening(
    reports: Iterable[str | Path],
    *,
    mode: str = "advisory",
    protocol: str = "auto",
) -> dict[str, Any]:
    """Return a recommendation without deleting or rejecting an output.

    A contract flag means that an explicit invariant such as exact supplied
    geometry, chain continuity, or declared symmetry needs inspection.  An
    advisory flag means that a task-dependent proxy (packing, compactness,
    clash burden, interface coverage, or shape target) was not satisfied.
    Neither category asserts experimental failure or designability.
    """

    paths = tuple(Path(path).resolve() for path in reports)
    if mode not in {"off", "advisory"}:
        raise ValueError("screening mode must be off or advisory")
    if protocol not in {"auto", "generic_backbone", "hoyeung_lhd101"}:
        raise ValueError(f"Unsupported screening protocol: {protocol}")
    if mode == "off":
        return {
            "schema_version": 1,
            "mode": mode,
            "protocol": protocol,
            "generated_output_retained": True,
            "contract_status": "not_evaluated",
            "recommendation": "not_screened",
            "contract_flags": [],
            "advisory_flags": [],
            "reports": [str(path) for path in paths],
        }

    contract_flags: list[dict[str, Any]] = []
    advisory_flags: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_object(path)
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        passed = payload.get("passed") is True

        if path.name in _HARD_REPORTS:
            if not passed:
                contract_flags.append(
                    _flag(
                        code=f"contract.{path.stem}",
                        report=path,
                        message=(
                            "An explicit geometry/runtime contract was not "
                            "satisfied; inspect the source audit."
                        ),
                    )
                )
            continue

        if path.name in _ADVISORY_REPORTS:
            if not passed:
                advisory_flags.append(
                    _flag(
                        code=f"advisory.{path.stem}",
                        report=path,
                        message=(
                            "A task-dependent interface or packing target "
                            "was not satisfied."
                        ),
                    )
                )
            continue

        if path.name == "graph_interface_guidance_audit.json":
            contract_count = len(contract_flags)
            advisory_count = len(advisory_flags)
            contract_flags.extend(
                _false_summary_flags(
                    summary,
                    (
                        "runtime_active",
                        "identifier_contract_valid",
                        "patch_identity_contract_valid",
                        "adaptive_phase_contract_valid",
                        "capacity_preflight_contract_valid",
                        "final_proxy_contract_valid",
                        "execution_contract_valid",
                    ),
                    report=path,
                    prefix="contract.interface_guidance",
                )
            )
            controller_proxy_satisfied = summary.get(
                "controller_proxy_targets_satisfied",
                summary.get("final_proxy_targets_satisfied"),
            )
            if controller_proxy_satisfied is False:
                advisory_flags.append(
                    _flag(
                        code="advisory.interface_guidance.controller_proxies",
                        report=path,
                        message=(
                            "The generated-interface controller proxy bundle "
                            "was not reached. This is not a published "
                            "interface-quality verdict."
                        ),
                        observed=summary.get("final_packing_metrics"),
                    )
                )
            if (
                not passed
                and len(contract_flags) == contract_count
                and len(advisory_flags) == advisory_count
            ):
                advisory_flags.append(
                    _flag(
                        code="advisory.interface_guidance.audit_flagged",
                        report=path,
                        message="The legacy interface-guidance audit was flagged.",
                    )
                )
            continue

        if path.name == "scaffold_validity_audit.json":
            contract_count = len(contract_flags)
            advisory_count = len(advisory_flags)
            contract_flags.extend(
                _false_summary_flags(
                    summary,
                    ("passed_continuity", "passed_symmetry"),
                    report=path,
                    prefix="contract.scaffold",
                )
            )
            advisory_flags.extend(
                _false_summary_flags(
                    summary,
                    (
                        "passed_clashes",
                        "passed_compactness",
                        "passed_assembly_shape",
                    ),
                    report=path,
                    prefix="advisory.scaffold",
                )
            )
            # Older reports may only carry a top-level decision.  Preserve
            # that information as advice rather than silently upgrading it
            # into a hard failure.
            if (
                not passed
                and len(contract_flags) == contract_count
                and len(advisory_flags) == advisory_count
            ):
                advisory_flags.append(
                    _flag(
                        code="advisory.scaffold.audit_flagged",
                        report=path,
                        message="The legacy scaffold audit was flagged.",
                    )
                )
            continue

        if path.name == "scaffold_core_guidance_audit.json":
            contract_flags.extend(
                _false_summary_flags(
                    summary,
                    (
                        "runtime_active",
                        "config_contract_valid",
                        "step_contract_valid",
                        "final_metric_contract_valid",
                        "safety_contract_valid",
                    ),
                    report=path,
                    prefix="contract.scaffold_core",
                )
            )
            targets_satisfied = summary.get(
                "declared_quality_targets_satisfied",
                summary.get("scientific_quality_satisfied"),
            )
            if targets_satisfied is False:
                quality_flag = _flag(
                    code=(
                        "contract.scaffold_core.declared_quality_targets"
                        if summary.get("quality_required") is True
                        else "advisory.scaffold_core.controller_proxies"
                    ),
                    report=path,
                    message=(
                        "The explicitly required scaffold proxy targets "
                        "were not satisfied."
                        if summary.get("quality_required") is True
                        else "Optional compactness/tertiary-support controller "
                        "reference values were not reached."
                    ),
                    observed=summary.get("final_metrics"),
                )
                if summary.get("quality_required") is True:
                    contract_flags.append(quality_flag)
                else:
                    advisory_flags.append(quality_flag)
            continue

        if not passed:
            advisory_flags.append(
                _flag(
                    code=f"advisory.unclassified.{path.stem}",
                    report=path,
                    message=(
                        "An unclassified audit was flagged; Mosaic reports "
                        "it as advice rather than inventing a hard gate."
                    ),
                )
            )

    contract_status = "flagged" if contract_flags else "met"
    if contract_flags:
        recommendation = "review_contract"
    elif advisory_flags:
        recommendation = "review_advisory_metrics"
    else:
        recommendation = "recommended_for_next_stage"
    return {
        "schema_version": 1,
        "mode": mode,
        "protocol": protocol,
        "generated_output_retained": True,
        "contract_status": contract_status,
        "recommendation": recommendation,
        "contract_flags": contract_flags,
        "advisory_flags": advisory_flags,
        "reports": [str(path) for path in paths],
        "interpretation": (
            "This is a backbone-only recommendation, not an experimental "
            "success/failure label. Sequence design and refolding remain "
            "necessary to assess designability."
        ),
        "evidence": {
            "policy": "docs/rfd3_mosaic/BACKBONE_EVALUATION_EVIDENCE.md",
            "metric_provenance": ("docs/rfd3_mosaic/STRUCTURE_METRIC_PROVENANCE.md"),
            "cohort_note": (
                "Ho-Yeung loop/Rg selection and diversity analysis require "
                "a campaign cohort and are not inferred from one structure."
            ),
        },
    }


def write_advisory_screening(
    output: str | Path,
    reports: Iterable[str | Path],
    *,
    mode: str = "advisory",
    protocol: str = "auto",
) -> dict[str, Any]:
    payload = build_advisory_screening(
        reports,
        mode=mode,
        protocol=protocol,
    )
    path = Path(output)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = ["build_advisory_screening", "write_advisory_screening"]
