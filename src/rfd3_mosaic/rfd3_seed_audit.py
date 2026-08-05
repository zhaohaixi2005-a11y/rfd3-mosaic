"""Audit preservation of a cross-chain Interface-Seed in RFD3 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from rfd3_mosaic.structure import (
    parse_atom_selection,
    read_pdb_atoms,
    read_structure_atoms,
    select_atoms,
)
from rfd3_mosaic.validation import (
    audit_interface_seed_pairs,
    infer_fragment_placements,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _derive_structure_path(result_json: Path) -> Path:
    base = result_json.with_suffix("")
    candidates = [
        Path(f"{base}.cif.gz"),
        Path(f"{base}.cif"),
        Path(f"{base}.pdb"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find result structure beside metadata JSON; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _derive_mapping_path(
    result_json: Path,
    result_payload: dict[str, Any],
) -> Path:
    specification = result_payload.get("specification", {})
    input_path = specification.get("input")
    candidates = [result_json.parent / "adapter" / "mapping.json"]
    if input_path:
        candidates.insert(0, Path(input_path).resolve().parent / "mapping.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find adapter mapping.json; pass --adapter-mapping"
    )


def _derive_config_path(
    result_payload: dict[str, Any],
) -> Path:
    config = (
        result_payload.get("specification", {})
        .get("extra", {})
        .get("interface_seed_config")
    )
    if not config:
        raise ValueError(
            "Result metadata does not record interface_seed_config; "
            "pass --config"
        )
    path = Path(config).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Recorded Interface-Seed config does not exist: {path}"
        )
    return path


def _resolve_source(
    configured_source: str,
    *,
    config_path: Path,
    base_directory: Path | None,
) -> Path:
    source = Path(configured_source)
    if source.is_absolute():
        return source
    candidates: list[Path] = []
    if base_directory is not None:
        candidates.append(base_directory / source)
    candidates.append(config_path.parent / source)
    candidates.extend(parent / source for parent in config_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve fragment source {configured_source!r}; "
        "pass --base-directory"
    )


def _load_references(
    config_path: Path,
    fragment_ids: set[str],
    *,
    base_directory: Path | None,
) -> dict[str, tuple]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = payload["interface_seed"]
    configured_fragments = seed["fragments"]
    missing = fragment_ids - set(configured_fragments)
    if missing:
        raise ValueError(
            f"Adapter mapping fragments are absent from config: {sorted(missing)}"
        )
    references = {}
    for fragment_id in sorted(fragment_ids):
        fragment = configured_fragments[fragment_id]
        source = _resolve_source(
            fragment["source"],
            config_path=config_path,
            base_directory=base_directory,
        )
        selection = parse_atom_selection(fragment["selection"])
        references[fragment_id] = select_atoms(
            read_pdb_atoms(source), selection
        )
    return references


def _derive_fragment_pairs(
    config_path: Path,
) -> tuple[tuple[str, str], ...]:
    """Resolve each configured interface to one explicit fragment pair."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = payload["interface_seed"]
    ports = seed.get("ports", {})
    interfaces = seed.get("interfaces", {})
    pairs: list[tuple[str, str]] = []
    for interface_id, interface in interfaces.items():
        left_port_id = interface["left_port"]
        right_port_id = interface["right_port"]
        try:
            left_fragments = ports[left_port_id]["fragments"]
            right_fragments = ports[right_port_id]["fragments"]
        except KeyError as error:
            raise ValueError(
                f"Interface {interface_id!r} references an unknown port"
            ) from error
        if len(left_fragments) != 1 or len(right_fragments) != 1:
            raise ValueError(
                f"Interface {interface_id!r} must resolve to exactly one "
                "fragment on each side for the seed-integrity audit"
            )
        pair = (str(left_fragments[0]), str(right_fragments[0]))
        if pair[0] == pair[1]:
            raise ValueError(
                f"Interface {interface_id!r} uses the same fragment twice"
            )
        pairs.append(pair)
    if not pairs:
        raise ValueError("Config does not define an interface to audit")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Config defines duplicate interface fragment pairs")
    return tuple(pairs)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that every configured two-fragment cross-chain "
            "Interface-Seed is rigidly preserved in an RFD3 result."
        )
    )
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--adapter-mapping", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-directory", type=Path)
    parser.add_argument(
        "--left-fragment",
        help="Audit one explicit pair instead of all configured interfaces.",
    )
    parser.add_argument(
        "--right-fragment",
        help="Audit one explicit pair instead of all configured interfaces.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--max-ca-rmsd", type=float, default=0.5)
    parser.add_argument("--max-all-atom-rmsd", type=float, default=0.75)
    parser.add_argument("--min-contact-retention", type=float, default=0.9)
    parser.add_argument("--min-atom-completeness", type=float, default=0.99)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit zero after writing the report.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    result_json = arguments.result_json.resolve()
    result_payload = _load_json(result_json)
    result_structure = (
        arguments.result_structure.resolve()
        if arguments.result_structure
        else _derive_structure_path(result_json)
    )
    mapping_path = (
        arguments.adapter_mapping.resolve()
        if arguments.adapter_mapping
        else _derive_mapping_path(result_json, result_payload)
    )
    config_path = (
        arguments.config.resolve()
        if arguments.config
        else _derive_config_path(result_payload)
    )
    output_path = (
        arguments.output.resolve()
        if arguments.output
        else result_json.with_name(
            f"{result_json.stem}_seed_integrity.json"
        )
    )

    placements = infer_fragment_placements(
        _load_json(mapping_path),
        result_payload.get("diffused_index_map", {}),
    )
    references = _load_references(
        config_path,
        set(placements),
        base_directory=(
            arguments.base_directory.resolve()
            if arguments.base_directory
            else None
        ),
    )
    if bool(arguments.left_fragment) != bool(arguments.right_fragment):
        raise ValueError(
            "--left-fragment and --right-fragment must be provided together"
        )
    fragment_pairs = (
        ((arguments.left_fragment, arguments.right_fragment),)
        if arguments.left_fragment
        else _derive_fragment_pairs(config_path)
    )
    report = audit_interface_seed_pairs(
        output_atoms=read_structure_atoms(result_structure),
        references=references,
        placements=placements,
        fragment_pairs=fragment_pairs,
        contact_cutoff=arguments.contact_cutoff,
        max_ca_rmsd=arguments.max_ca_rmsd,
        max_all_atom_rmsd=arguments.max_all_atom_rmsd,
        min_contact_retention=arguments.min_contact_retention,
        min_atom_completeness=arguments.min_atom_completeness,
    )
    report["inputs"] = {
        "result_json": str(result_json),
        "result_structure": str(result_structure),
        "adapter_mapping": str(mapping_path),
        "config": str(config_path),
        "rfd3_seed": result_payload.get("seed"),
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print(f"Interface-Seed audit: {'PASSED' if report['passed'] else 'FAILED'}")
    if "interface_seeds" in summary:
        print(
            "interface seeds:        "
            f"{summary['passed_interface_seeds']}/{summary['interface_seeds']}"
        )
    print(f"cross-chain seed pairs: {summary['seed_pairs']}")
    print(f"maximum CA RMSD:       {summary['maximum_ca_rmsd']:.4f} A")
    print(
        "maximum all-atom RMSD: "
        f"{summary['maximum_all_atom_rmsd']:.4f} A"
    )
    print(
        "minimum contact retention: "
        f"{summary['minimum_contact_retention']:.3f}"
    )
    for pair in report["seed_pairs"]:
        print(
            f"  left {pair['left_chain']} + right {pair['right_chain']}: "
            f"CA={pair['ca_rmsd']:.4f} A, "
            f"all={pair['all_atom_rmsd']:.4f} A, "
            f"contacts={pair['contact_retention']:.3f}, "
            f"{'PASS' if pair['passed'] else 'FAIL'}"
        )
    print(f"report: {output_path}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
