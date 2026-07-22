"""Command-line entry point for the static RFD3 input adapter."""

import argparse
from pathlib import Path

from rfd3_mosaic.output import compile_rfd3_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Interface-Seed inputs for native RFD3 symmetry."
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
        "--example-id",
        default="lhd101_c3_interface_seed",
    )
    parser.add_argument(
        "--pose-seed",
        type=int,
        help=(
            "Override the config random seed for rigid orientation/radius "
            "sampling; recorded in the generated provenance."
        ),
    )
    parser.add_argument(
        "--pose-candidate-manifest",
        type=Path,
        help=(
            "Rebuild the exact sampled pose recorded by a standalone "
            "candidate manifest and fail on config/structure SHA mismatch."
        ),
    )
    arguments = parser.parse_args()
    outputs = compile_rfd3_input(
        arguments.config,
        arguments.output_dir,
        base_directory=arguments.base_directory,
        example_id=arguments.example_id,
        pose_seed=arguments.pose_seed,
        pose_candidate_manifest=arguments.pose_candidate_manifest,
    )
    print(f"RFD3 input: {outputs.input_path}")
    print(f"structure:  {outputs.structure_path}")
    print(f"mapping:    {outputs.mapping_path}")
    print(f"manifest:   {outputs.manifest_path}")
    print(f"ASU contig: {outputs.contig}")


if __name__ == "__main__":
    main()
