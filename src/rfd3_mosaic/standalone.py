"""Command-line entry point for the RFD3-independent compiler."""

import argparse
import json
from pathlib import Path

from rfd3_mosaic.output import compile_standalone


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Interface-Seed inputs without running RFD3."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--base-directory",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative fragment source paths.",
    )
    parser.add_argument(
        "--allow-infeasible",
        action="store_true",
        help=(
            "Emit diagnostics for clashing or unsatisfied candidates instead "
            "of rejecting them; intended for pose-search ranking."
        ),
    )
    parser.add_argument(
        "--pose-seed",
        type=int,
        help="Override the config seed for rigid pose sampling.",
    )
    arguments = parser.parse_args()
    outputs = compile_standalone(
        arguments.config,
        arguments.output_dir,
        base_directory=arguments.base_directory,
        strict_validation=not arguments.allow_infeasible,
        random_seed=arguments.pose_seed,
    )
    print(f"structure: {outputs.structure_path}")
    print(f"mapping:   {outputs.mapping_path}")
    print(f"manifest:  {outputs.manifest_path}")
    print(
        f"compiled {outputs.atom_count} atoms, "
        f"{outputs.residue_count} residues, "
        f"{outputs.chain_count} chains"
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    validation = manifest["validation"]
    clashes = validation["inter_group_clashes"]
    linkers = validation["scaffold_link_geometry"]
    cavities = validation["symmetry_cavities"]["orbits"]
    for orbit in cavities:
        symmetry_prefix = (
            "C" if orbit["symmetry_type"] == "cyclic" else "D"
        )
        print(
            f"symmetry: {symmetry_prefix}{orbit['symmetry_order']} "
            f"({orbit['copy_count']} copies)"
        )
        print(
            f"central void: {orbit['central_void_radius']:.3f} A; "
            f"axis clearance: {orbit['minimum_axis_clearance']:.3f} A"
        )
    feasible_count = sum(
        bool(link["within_maximum_contour"])
        for link in linkers["links"]
    )
    print(
        f"hard clashes: {clashes['total_hard_clashes']}; "
        f"link spans feasible: {feasible_count}/{len(linkers['links'])}"
    )
    objectives = validation["objectives"]
    print(
        f"objective penalty: {objectives['total_weighted_penalty']:.6g}; "
        f"required failures: {objectives['required_failure_count']}"
    )


if __name__ == "__main__":
    main()
