#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CYCLIC_ORDER OUTPUT_DIRECTORY"
    exit 2
fi

ORDER=$1
OUTPUT_DIR=$2
if [[ ! "$ORDER" =~ ^(5|6|7)$ ]]; then
    echo "ERROR: CYCLIC_ORDER must be 5, 6, or 7"
    exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SAMPLE_COUNT=${RFD3_POSE_SAMPLES:-512}
SEED_START=${RFD3_POSE_SEED_START:-3000}
MAX_SELECTED=${RFD3_QD_MAX_SELECTED:-16}
MIN_SO3_SEPARATION=${RFD3_QD_MIN_SO3_SEPARATION:-25}
QUALITY_POOL_FRACTION=${RFD3_QD_QUALITY_POOL_FRACTION:-0.25}
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/cyclic/lhd101_c${ORDER}.yaml"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"

python -m rfd3_mosaic.pose_ensemble \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --base-directory "$PROJECT_DIR" \
    --samples "$SAMPLE_COUNT" \
    --seed-start "$SEED_START" \
    --sampling-strategy latin_hypercube

python -m rfd3_mosaic.pose_qd \
    --ensemble "$OUTPUT_DIR/pose_ensemble.json" \
    --max-selected "$MAX_SELECTED" \
    --quality-pool-fraction "$QUALITY_POOL_FRACTION" \
    --min-orientation-separation-deg "$MIN_SO3_SEPARATION"

echo "C${ORDER} quality-diversity shortlist: $OUTPUT_DIR/pose_qd_shortlist.json"
