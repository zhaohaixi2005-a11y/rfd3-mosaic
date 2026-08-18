#!/usr/bin/env bash
set -euo pipefail

mode=${1:-pilot}
if [[ "$mode" != "pilot" && "$mode" != "full" ]]; then
    echo "usage: $0 [pilot|full]"
    exit 2
fi

HOYEUNG_REPO=${HOYEUNG_REPO:?Set HOYEUNG_REPO to the original RFdiffusion_interfaceseed checkout}
RFDIFFUSION_PYTHON=${RFDIFFUSION_PYTHON:?Set RFDIFFUSION_PYTHON to its environment python}
RFDIFFUSION_MODEL_DIR=${RFDIFFUSION_MODEL_DIR:-$HOYEUNG_REPO/models}
RFD1_RUN_BASE=${RFD1_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/baselines}
SBATCH_SCRIPT=${RFD1_SBATCH_SCRIPT:-$(pwd)/scripts/rfd3_mosaic/hoyeung_lhd101_1000_array.sbatch}

test -f "$HOYEUNG_REPO/scripts/run_inference.py"
test -f "$HOYEUNG_REPO/examples/input_pdbs/7mwr_interface.pdb"
test -x "$RFDIFFUSION_PYTHON"
test -f "$RFDIFFUSION_MODEL_DIR/Base_ckpt.pt"
test -f "$SBATCH_SCRIPT"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ "$mode" == "pilot" ]]; then
    output_root="$RFD1_RUN_BASE/hoyeung-lhd101-pilot-$stamp"
    array_spec="0-0"
    per_task=1
    total=1
else
    output_root="$RFD1_RUN_BASE/hoyeung-lhd101-1000-$stamp"
    array_spec=${RFD1_ARRAY_SPEC:-0-99%16}
    per_task=10
    total=1000
fi
mkdir -p "$output_root"

git -C "$HOYEUNG_REPO" rev-parse HEAD >"$output_root/source_revision.txt"
sha256sum \
    "$HOYEUNG_REPO/examples/input_pdbs/7mwr_interface.pdb" \
    "$RFDIFFUSION_MODEL_DIR/Base_ckpt.pt" \
    "$HOYEUNG_REPO/scripts/run_inference.py" \
    >"$output_root/frozen_identities.sha256"

job_output=$(sbatch \
    --array="$array_spec" \
    --export=ALL,HOYEUNG_REPO="$HOYEUNG_REPO",RFDIFFUSION_PYTHON="$RFDIFFUSION_PYTHON",RFDIFFUSION_MODEL_DIR="$RFDIFFUSION_MODEL_DIR",RFD1_OUTPUT_ROOT="$output_root",RFD1_DESIGNS_PER_TASK="$per_task",RFD1_EXPECTED_TOTAL="$total" \
    "$SBATCH_SCRIPT")
printf '%s\n' "$job_output" | tee "$output_root/submission.txt"
echo "output: $output_root"
