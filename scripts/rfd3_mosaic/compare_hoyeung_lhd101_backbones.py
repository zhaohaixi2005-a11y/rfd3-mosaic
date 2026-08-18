#!/usr/bin/env python3
"""Build the paper-specific LHD101 backbone comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfd3_mosaic.backbone_comparison import (
    compare_hoyeung_backbone_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one completed Mosaic LHD101 C3 campaign with the "
            "RFdiffusion/backbone stage reported by Ho-Yeung et al."
        )
    )
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Run root containing .rfd3-mosaic/jobs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--stride",
        help=(
            "Optional STRIDE executable for the paper's loop-percentage "
            "screen. All Mosaic-native metrics work without it."
        ),
    )
    arguments = parser.parse_args()

    artifacts = compare_hoyeung_backbone_campaign(
        arguments.campaign_manifest,
        output_directory=arguments.output_dir,
        run_root=arguments.run_root,
        stride_executable=arguments.stride,
    )
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    print("Mosaic / Ho-Yeung LHD101 backbone comparison")
    print(
        "designs: "
        f"requested={summary['requested_designs']} "
        f"produced={summary['produced_designs']} "
        f"analyzed={summary['analyzed_designs']}"
    )
    print(
        "strict:  "
        f"{summary['worker_accepted_count']}/{summary['analyzed_designs']}"
    )
    print(f"JSON:    {artifacts.json_path}")
    print(f"CSV:     {artifacts.csv_path}")
    print(f"report:  {artifacts.markdown_path}")
    if not summary["generation_complete"]:
        raise SystemExit(
            "Campaign is incomplete; report was written without treating "
            "missing shards as failed structures."
        )


if __name__ == "__main__":
    main()
