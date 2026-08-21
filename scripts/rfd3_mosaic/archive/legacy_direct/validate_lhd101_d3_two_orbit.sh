#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONFIG=${RFD3_D3_TWO_ORBIT_CONFIG:-$PROJECT_DIR/configs/rfd3_mosaic/dihedral/lhd101_d3_two_orbit_engineering.yaml}
OUTPUT_DIR=${1:-$PROJECT_DIR/runs/rfd3-mosaic/d3_two_orbit_engineering/prevalidation}
EXAMPLE_ID=lhd101_d3_two_orbit_engineering

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"

echo "=== D3 two-interface-orbit engineering validation ==="
echo "config: $CONFIG"
echo "output: $OUTPUT_DIR"

python -m unittest discover \
    -s tests/rfd3_mosaic/unit \
    -p 'test_rfd3_adapter.py' \
    -v
python -m unittest discover \
    -s tests/rfd3_mosaic/unit \
    -p 'test_seed_integrity.py' \
    -v

mkdir -p "$OUTPUT_DIR"
python -m rfd3_mosaic.rfd3_adapter \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR/adapter" \
    --base-directory "$PROJECT_DIR" \
    --example-id "$EXAMPLE_ID"

python -m rfd3_mosaic.rfd3_prevalidate \
    --input "$OUTPUT_DIR/adapter/rfd3_input.json" \
    --example-id "$EXAMPLE_ID" \
    --report "$OUTPUT_DIR/rfd3_prevalidation.json"

python - "$OUTPUT_DIR/adapter/rfd3_input.json" "$OUTPUT_DIR/rfd3_prevalidation.json" <<'PY'
import json
import sys
from pathlib import Path

input_payload = json.loads(Path(sys.argv[1]).read_text())
example = input_payload["lhd101_d3_two_orbit_engineering"]
report = json.loads(Path(sys.argv[2]).read_text())
extra = example["extra"]

print("=== Compiled multi-orbit summary ===")
print("symmetry:", example["symmetry"]["id"])
print("contig:", example["contig"])
print("ASU chains:", extra["asu_chain_count"])
print("ASU segments:", len(extra["asu_scaffold_segments"]))
print("constraint groups:", len(extra["motif_constraint_groups"]))
print("constraint orbits:", len(extra["motif_constraint_orbits"]))
print("prevalidation status:", report["status"])
PY
