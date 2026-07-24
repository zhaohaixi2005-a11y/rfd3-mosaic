"""Command-line scaffold geometry audit for RFD3 output structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfd3_mosaic.rfd3_seed_audit import _derive_structure_path
from rfd3_mosaic.structure import read_structure_atoms
from rfd3_mosaic.validation.scaffold_validity import audit_scaffold_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-chain-ca-rg", type=float, default=25.0)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()

    result_json = arguments.result_json.resolve()
    structure = (
        arguments.result_structure.resolve()
        if arguments.result_structure
        else _derive_structure_path(result_json)
    )
    report = audit_scaffold_geometry(
        read_structure_atoms(structure),
        max_chain_ca_rg=arguments.max_chain_ca_rg,
    )
    report["inputs"] = {
        "result_json": str(result_json),
        "result_structure": str(structure),
    }
    output = arguments.output.resolve()
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(f"Scaffold audit: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"chains:              {summary['chain_count']}")
    print(f"chain breaks:        {summary['chain_break_count']}")
    print(
        "maximum chain CA Rg: "
        f"{summary['maximum_chain_ca_radius_of_gyration']:.3f} A"
    )
    print(f"CA clashes:          {summary['ca_clash_count']}")
    print(f"report: {output}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
