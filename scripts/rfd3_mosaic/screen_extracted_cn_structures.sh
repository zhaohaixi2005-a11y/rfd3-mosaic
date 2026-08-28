#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${RFD3_PROJECT_DIR:-$(pwd)}"
RUN_BASE="${RFD3_RUN_BASE:-${RFD3_MOSAIC_RUN_ROOT:-$HOME/rfd3-mosaic-runs}}"
ORDERS="${RFD3_SCREEN_ORDERS:-5 6 7}"

read -r -a ORDER_VALUES <<<"$ORDERS"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"

for order in "${ORDER_VALUES[@]}"; do
    input_dir="$RUN_BASE/native_c${order}_full/extracted_cif"
    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: extracted structure directory does not exist: $input_dir" >&2
        exit 1
    fi

    echo "=== Screening C${order}: $input_dir ==="
    python -m rfd3_mosaic.rfd3_batch_screen \
        --input-dir "$input_dir" \
        --symmetry-order "$order"
done

python - "$RUN_BASE" "${ORDER_VALUES[@]}" <<'PY'
import json
import sys
from pathlib import Path


def display(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


run_base = Path(sys.argv[1])
orders = [int(value) for value in sys.argv[2:]]

print("\n=== Cn extracted-structure screen summary ===")
for order in orders:
    report_path = (
        run_base
        / f"native_c{order}_full"
        / "extracted_cif"
        / f"c{order}_batch_screen.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]

    seed_passes = sum(item["seed_passed"] is True for item in records)
    geometry_passes = sum(
        item["recomputed_scaffold_passed"] for item in records
    )
    continuous = sum(
        item["scaffold_summary"]["passed_continuity"]
        for item in records
    )
    clash_free = sum(
        item["scaffold_summary"]["ca_clash_count"] == 0
        for item in records
    )
    symmetry_available = sum(
        item["declared_symmetry_available"] for item in records
    )

    print(f"\nC{order}: {len(records)} structures")
    print(
        f"  seed audit passed:       {seed_passes}/{len(records)}"
    )
    print(
        f"  continuous chains:       {continuous}/{len(records)}"
    )
    print(
        f"  zero CA clashes:         {clash_free}/{len(records)}"
    )
    print(
        f"  declared symmetry found: {symmetry_available}/{len(records)}"
    )
    print(
        f"  geometry gate passed:    {geometry_passes}/{len(records)}"
    )
    print(
        f"  strict total passed:     "
        f"{report['strict_pass_count']}/{len(records)}"
    )
    print("  top candidates:")

    for item in records[: min(5, len(records))]:
        scaffold = item["scaffold_summary"]
        packing = item["packing"]
        ring = item["ring"]
        print(
            "    "
            f"rank={item['rank']} "
            f"job={item['job_id'] or 'unknown'} "
            f"pose={display(item.get('pose_seed'))} "
            f"diffusion={display(item.get('diffusion_seed'))} "
            f"strict={item['strict_passed']} "
            f"breaks={scaffold['chain_break_count']} "
            f"clashes={scaffold['ca_clash_count']} "
            f"neighbor_contacts="
            f"{display(packing.get('minimum_neighbor_ca_contacts'))} "
            f"min_interchain_CA="
            f"{display(packing.get('minimum_interchain_ca_distance'))}A "
            f"ring_radius="
            f"{display(ring.get('mean_chain_com_radius'))}A "
            f"axis_clearance="
            f"{display(ring.get('minimum_ca_axis_clearance'))}A "
            f"shape_aspect="
            f"{display(ring.get('ca_axial_to_radial_aspect_ratio'))}"
        )

print(
    "\nStrict pass is the acceptance gate. Packing and ring descriptors "
    "rank structures only after the gate; they are not calibrated "
    "biological success thresholds."
)
PY
