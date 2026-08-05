#!/bin/bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE=${RFD3_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}
MANIFEST=${RFD3_POSE_CANDIDATE_MANIFEST:?Set RFD3_POSE_CANDIDATE_MANIFEST to one validated C5 candidate manifest}
DIFFUSION_SEED=${RFD3_AB_DIFFUSION_SEED:-42}
TIMESTEPS=${RFD3_AB_NUM_TIMESTEPS:-10}
NEIGHBOUR_RADIUS=${RFD3_AB_NEIGHBOUR_RADIUS:-1}
WALLTIME=${RFD3_AB_WALLTIME:-12:00:00}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_FILE=${RFD3_AB_JOB_FILE:-"$RUN_BASE/c5_local_backend_ab_${TIMESTAMP}.tsv"}

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: candidate manifest does not exist: $MANIFEST"
    exit 2
fi
if [[ ! "$DIFFUSION_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_AB_DIFFUSION_SEED must be a non-negative integer"
    exit 2
fi
if [[ ! "$TIMESTEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( TIMESTEPS < 2 || TIMESTEPS > 200 )); then
    echo "ERROR: RFD3_AB_NUM_TIMESTEPS must be an integer from 2 to 200"
    exit 2
fi
if [[ ! "$NEIGHBOUR_RADIUS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_AB_NEIGHBOUR_RADIUS must be a non-negative integer"
    exit 2
fi

mkdir -p "$(dirname "$JOB_FILE")"
printf 'backend\tdiffusion_seed\tsteps\tneighbour_radius\tjob_id\tmanifest\n' \
    >"$JOB_FILE"

cd "$PROJECT_DIR"
for BACKEND in explicit_all_copy local_neighbourhood; do
    JOB_ID=$(sbatch --parsable \
        --job-name="c5-ab-${BACKEND%%_*}" \
        --time="$WALLTIME" \
        --export="ALL,RFD3_CYCLIC_ORDER=5,RFD3_POSE_CANDIDATE_MANIFEST=${MANIFEST},RFD3_SEED=${DIFFUSION_SEED},RFD3_NUM_TIMESTEPS=${TIMESTEPS},RFD3_SYMMETRY_EXECUTION_BACKEND=${BACKEND},RFD3_SYMMETRY_NEIGHBOUR_RADIUS=${NEIGHBOUR_RADIUS}" \
        scripts/rfd3_mosaic/lhd101_cn_full_p100.sbatch)
    JOB_ID=${JOB_ID%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$BACKEND" \
        "$DIFFUSION_SEED" \
        "$TIMESTEPS" \
        "$NEIGHBOUR_RADIUS" \
        "$JOB_ID" \
        "$MANIFEST" | tee -a "$JOB_FILE"
done

echo "C5 explicit/local A/B jobs recorded in: $JOB_FILE"
