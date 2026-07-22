"""Construct and audit an RFD3 atom array without loading a checkpoint."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _expected_multiplicity(symmetry_id: str) -> int:
    normalized = symmetry_id.strip().upper()
    if normalized.startswith("C") and normalized[1:].isdigit():
        order = int(normalized[1:])
        if order >= 2:
            return order
    if normalized.startswith("D") and normalized[1:].isdigit():
        order = int(normalized[1:])
        if order >= 2:
            return 2 * order
    raise ValueError(
        f"Prevalidation currently supports native Cn/Dn symmetry, got "
        f"{symmetry_id!r}"
    )


def _validate_report(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected = report["expected_multiplicity"]
    expected_asu_chains = report.get("expected_asu_chain_count", 1)
    expected_chain_count = expected * expected_asu_chains
    if report["chain_count"] != expected_chain_count:
        failures.append(
            f"expected {expected_chain_count} chains, observed "
            f"{report['chain_count']}"
        )
    if report["symmetry_transform_ids"] != list(range(expected)):
        failures.append(
            "symmetry transform IDs do not cover every native copy"
        )
    residue_counts = list(report["residues_per_chain"].values())
    residue_count_frequencies = Counter(residue_counts)
    if not residue_counts or any(
        frequency % expected != 0
        for frequency in residue_count_frequencies.values()
    ):
        failures.append(
            "chain residue counts do not repeat across every symmetry copy"
        )
    if report["motif_atom_count"] <= 0:
        failures.append("RFD3 did not recognize any motif atoms")
    if report["fixed_coordinate_atom_count"] <= 0:
        failures.append("RFD3 did not recognize any fixed motif coordinates")
    if report["fixed_coordinate_atom_count"] != report["motif_atom_count"]:
        failures.append(
            "not every motif atom has fixed coordinates "
            f"({report['fixed_coordinate_atom_count']}/"
            f"{report['motif_atom_count']})"
        )
    if report["fixed_sequence_atom_count"] != report["motif_atom_count"]:
        failures.append(
            "not every motif atom has fixed sequence identity "
            f"({report['fixed_sequence_atom_count']}/"
            f"{report['motif_atom_count']})"
        )
    if report["asu_atom_count"] <= 0:
        failures.append("RFD3 did not mark an asymmetric unit")
    if report["symmetry_ids"] != [report["expected_symmetry_id"]]:
        failures.append("constructed symmetry annotation does not match input")
    return failures


def prevalidate_rfd3_input(
    input_path: str | Path,
    *,
    example_id: str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load one emitted specification and run RFD3's complete input builder."""

    # Imports stay lazy so schema/compiler users do not require an RFD3 runtime.
    import numpy as np
    from rfd3.inference.input_parsing import (
        DesignInputSpecification,
        ensure_input_is_abspath,
    )
    from rfd3.transforms.conditioning_base import get_motif_features

    path = Path(input_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("RFD3 input JSON must contain at least one example")
    selected_id = example_id or next(iter(payload))
    if selected_id not in payload:
        raise KeyError(f"Unknown RFD3 example ID {selected_id!r}")
    raw_spec = payload[selected_id]
    if not isinstance(raw_spec, dict):
        raise ValueError("Each RFD3 example must contain a JSON object")
    raw_spec = ensure_input_is_abspath(dict(raw_spec), path)
    design = DesignInputSpecification.safe_init(**raw_spec)
    atom_array, metadata = design.build(return_metadata=True)

    chain_ids = sorted(str(value) for value in np.unique(atom_array.chain_id))
    residues_per_chain: dict[str, int] = {}
    atoms_per_chain: dict[str, int] = {}
    for chain_id in chain_ids:
        mask = atom_array.chain_id == chain_id
        atoms_per_chain[chain_id] = int(mask.sum())
        residues_per_chain[chain_id] = len(
            {
                int(residue_id)
                for residue_id in atom_array.res_id[mask]
            }
        )

    categories = set(atom_array.get_annotation_categories())
    motif_mask = get_motif_features(atom_array)["is_motif_atom"].astype(bool)
    expected_symmetry_id = str(raw_spec["symmetry"]["id"]).upper()
    extra = raw_spec.get("extra") or {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pending",
        "input_path": str(path),
        "example_id": selected_id,
        "expected_symmetry_id": expected_symmetry_id,
        "expected_multiplicity": _expected_multiplicity(
            expected_symmetry_id
        ),
        "expected_asu_chain_count": int(extra.get("asu_chain_count", 1)),
        "atom_count": int(len(atom_array)),
        "chain_count": len(chain_ids),
        "chain_ids": chain_ids,
        "atoms_per_chain": atoms_per_chain,
        "residues_per_chain": residues_per_chain,
        "motif_atom_count": int(motif_mask.sum()),
        "fixed_coordinate_atom_count": int(
            atom_array.is_motif_atom_with_fixed_coord.astype(bool).sum()
        ),
        "fixed_sequence_atom_count": int(
            atom_array.is_motif_atom_with_fixed_seq.astype(bool).sum()
        ),
        "asu_atom_count": int(atom_array.is_sym_asu.astype(bool).sum()),
        "symmetry_ids": sorted(
            str(value) for value in np.unique(atom_array.symmetry_id)
        ),
        "symmetry_transform_ids": sorted(
            int(value) for value in np.unique(atom_array.sym_transform_id)
        ),
        "annotation_count": len(categories),
        "rfd3_metadata": metadata,
    }
    failures = _validate_report(report)
    report["status"] = "passed" if not failures else "failed"
    report["failures"] = failures

    destination = (
        Path(report_path)
        if report_path is not None
        else path.with_name("rfd3_prevalidation.json")
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(destination.resolve())
    if failures:
        raise ValueError(
            "RFD3 input construction failed semantic checks: "
            + "; ".join(failures)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and audit an RFD3 input without loading a checkpoint."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--example-id")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    report = prevalidate_rfd3_input(
        arguments.input,
        example_id=arguments.example_id,
        report_path=arguments.report,
    )
    print("RFD3 input construction: PASSED")
    print(f"example:    {report['example_id']}")
    print(f"chains:     {report['chain_count']} {report['chain_ids']}")
    print(f"residues:   {report['residues_per_chain']}")
    print(f"motif atoms:{report['motif_atom_count']}")
    print(f"fixed atoms:{report['fixed_coordinate_atom_count']}")
    print(f"transforms: {report['symmetry_transform_ids']}")
    print(f"report:     {report['report_path']}")


if __name__ == "__main__":
    main()
