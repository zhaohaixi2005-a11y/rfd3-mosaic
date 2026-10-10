"""Content binding for cached audit reports, independent of run location."""

from __future__ import annotations

import importlib.util
import json
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rfd3_mosaic.provenance.software import sha256_file, source_tree_sha256

if TYPE_CHECKING:
    from rfd3_mosaic.assembly_compiler import CompiledAudit

BINDING_NAME = "audit_evidence.json"


def _auditor_identity() -> dict[str, Any]:
    roots = {"rfd3_mosaic": Path(__file__).resolve().parent}
    for name in ("rfd3", "foundry"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            roots[name] = Path(spec.origin).resolve().parent
    versions = {}
    for name in ("numpy", "scipy", "biotite", "atomworks", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"source_sha256": source_tree_sha256(roots), "packages": versions}


def _audit_dependencies(audits: tuple[CompiledAudit, ...]) -> list[dict[str, Any]]:
    plans = []
    for audit in audits:
        arguments = dict(audit.input_arguments)
        records = {}
        for flag, value in audit.input_arguments:
            path = Path(value)
            if path.is_file():
                records[flag] = {"sha256": sha256_file(path)}
            elif flag != "--base-directory":
                records[flag] = {"value": value}
        if audit.module == "rfd3_mosaic.rfd3_seed_audit" and "--config" in arguments:
            from rfd3_mosaic.rfd3_seed_audit import _resolve_source

            config = Path(arguments["--config"])
            payload = yaml.safe_load(config.read_text(encoding="utf-8"))
            fragments = payload["interface_seed"].get("fragments", {})
            base = arguments.get("--base-directory")
            records["fragment_sources"] = {
                name: sha256_file(_resolve_source(
                    fragment["source"], config_path=config,
                    base_directory=Path(base) if base else None,
                ))
                for name, fragment in sorted(fragments.items())
            }
        plans.append({
            "module": audit.module, "report_name": audit.report_name,
            "arguments": records,
        })
    return plans


def audit_evidence(
    *, compiled_input: Path, result_json: Path, reports: tuple[Path, ...],
    semantic_audits: tuple[CompiledAudit, ...] = (),
) -> dict[str, Any]:
    """Fingerprint the immutable contract, output and exact report set.

    Relative evidence identities keep a whole-run relocation valid.  Structure
    variants are recorded separately so replacing CIF with PDB or adding a
    conflicting companion cannot retain a cached pass.
    """
    structures = {
        suffix: sha256_file(path)
        for suffix in (".cif.gz", ".cif", ".pdb")
        if (path := result_json.with_suffix(suffix)).is_file()
    }
    payload = json.loads(compiled_input.read_text(encoding="utf-8"))
    inputs = {}
    for example_id, example in payload.items():
        source = example.get("input") if isinstance(example, dict) else None
        if source:
            path = Path(source)
            if not path.is_absolute():
                path = compiled_input.parent / path
            inputs[example_id] = sha256_file(path) if path.is_file() else None
    return {
        "schema_version": 2,
        "auditor": _auditor_identity(),
        "semantic_audits": _audit_dependencies(semantic_audits),
        "compiled_input_sha256": sha256_file(compiled_input),
        "input_structure_sha256": inputs,
        "result_metadata_sha256": sha256_file(result_json),
        "result_structure_sha256": structures,
        "reports": {path.name: sha256_file(path) for path in reports},
    }


def verify_audit_evidence(
    *, directory: Path, compiled_input: Path, result_json: Path,
    reports: tuple[Path, ...],
    semantic_audits: tuple[CompiledAudit, ...] = (),
) -> None:
    """Refuse reuse when reports have no binding or any bound content changed."""
    binding = directory / BINDING_NAME
    if not binding.is_file():
        raise ValueError(
            "Cached audit reports lack content-bound evidence; rerun audit "
            "without --reuse-reports (diffusion is not rerun)"
        )
    recorded = json.loads(binding.read_text(encoding="utf-8"))
    observed = audit_evidence(
        compiled_input=compiled_input, result_json=result_json, reports=reports,
        semantic_audits=semantic_audits,
    )
    if recorded != observed:
        raise ValueError(
            "Cached audit evidence changed: input, result or reports no longer "
            "match; rerun audit without --reuse-reports"
        )
