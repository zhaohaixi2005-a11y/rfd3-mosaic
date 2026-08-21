#!/bin/bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE=${RFD3_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}
PARTITION=${RFD3_H100_PARTITION:-lrz-hgx-h100-94x4}
POSES_PER_ORDER=${RFD3_SCREEN_POSES_PER_ORDER:-3}
NUM_TIMESTEPS=${RFD3_SCREEN_NUM_TIMESTEPS:-50}
WALLTIME=${RFD3_SCREEN_WALLTIME:-08:00:00}
SHORTLIST_TAG=${RFD3_SCREEN_SHORTLIST_TAG:-qd_v2}
DIFFUSION_SEEDS=${RFD3_SCREEN_DIFFUSION_SEEDS:-"42 43 44 45 46"}
ORDERS=${RFD3_SCREEN_ORDERS:-"5 6 7"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_FILE=${RFD3_SCREEN_JOB_FILE:-"$RUN_BASE/cn_h100_screen_${TIMESTAMP}.tsv"}

if [[ ! "$POSES_PER_ORDER" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RFD3_SCREEN_POSES_PER_ORDER must be a positive integer"
    exit 2
fi
if [[ ! "$NUM_TIMESTEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( NUM_TIMESTEPS < 2 || NUM_TIMESTEPS > 200 )); then
    echo "ERROR: RFD3_SCREEN_NUM_TIMESTEPS must be from 2 to 200"
    exit 2
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"

if [[ -f "$JOB_FILE" ]]; then
    EXPECTED_HEADER=$'order\tpose_rank\tpose_seed\tdiffusion_seed\tsteps\tjob_id\tmanifest'
    OBSERVED_HEADER=$(head -n 1 "$JOB_FILE")
    if [[ "$OBSERVED_HEADER" != "$EXPECTED_HEADER" ]]; then
        echo "ERROR: existing job file has an incompatible header: $JOB_FILE"
        exit 2
    fi
    echo "Resuming H100 screen from: $JOB_FILE"
else
    printf 'order\tpose_rank\tpose_seed\tdiffusion_seed\tsteps\tjob_id\tmanifest\n' \
        >"$JOB_FILE"
fi

for ORDER in $ORDERS; do
    if [[ ! "$ORDER" =~ ^(5|6|7)$ ]]; then
        echo "ERROR: screen order must be 5, 6, or 7; got $ORDER"
        exit 2
    fi
    SHORTLIST="$RUN_BASE/lhd101_c${ORDER}_${SHORTLIST_TAG}/pose_qd_shortlist.json"
    if [[ ! -f "$SHORTLIST" ]]; then
        echo "ERROR: shortlist does not exist: $SHORTLIST"
        exit 2
    fi

    while IFS=$'\t' read -r POSE_RANK POSE_SEED MANIFEST; do
        if [[ ! -f "$MANIFEST" ]]; then
            echo "ERROR: candidate manifest does not exist: $MANIFEST"
            exit 2
        fi
        for DIFFUSION_SEED in $DIFFUSION_SEEDS; do
            if [[ ! "$DIFFUSION_SEED" =~ ^[0-9]+$ ]]; then
                echo "ERROR: diffusion seeds must be non-negative integers"
                exit 2
            fi
            if awk -F '\t' \
                -v order="$ORDER" \
                -v pose_seed="$POSE_SEED" \
                -v diffusion_seed="$DIFFUSION_SEED" \
                -v steps="$NUM_TIMESTEPS" \
                'NR > 1 && $1 == order && $3 == pose_seed &&
                 $4 == diffusion_seed && $5 == steps { found = 1 }
                 END { exit(found ? 0 : 1) }' \
                "$JOB_FILE"; then
                echo "Already submitted: C${ORDER} pose=${POSE_SEED} diffusion_seed=${DIFFUSION_SEED} steps=${NUM_TIMESTEPS}"
                continue
            fi
            JOB_NAME="c${ORDER}p${POSE_SEED}d${DIFFUSION_SEED}t${NUM_TIMESTEPS}"
            if ! JOB_ID=$(sbatch --parsable \
                    --partition="$PARTITION" \
                    --job-name="$JOB_NAME" \
                    --time="$WALLTIME" \
                    --cpus-per-task=12 \
                    --export="ALL,RFD3_ACCELERATOR_LABEL=H100,RFD3_CYCLIC_ORDER=${ORDER},RFD3_NUM_TIMESTEPS=${NUM_TIMESTEPS},RFD3_SEED=${DIFFUSION_SEED},RFD3_POSE_CANDIDATE_MANIFEST=${MANIFEST}" \
                    scripts/rfd3_mosaic/archive/legacy_direct/lhd101_cn_full_p100.sbatch); then
                echo "Submission stopped. After QOS capacity is available, resume with:"
                echo "RFD3_SCREEN_JOB_FILE=$JOB_FILE bash scripts/rfd3_mosaic/archive/legacy_direct/submit_lhd101_cn_h100_screen.sh"
                exit 75
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$ORDER" \
                "$POSE_RANK" \
                "$POSE_SEED" \
                "$DIFFUSION_SEED" \
                "$NUM_TIMESTEPS" \
                "$JOB_ID" \
                "$MANIFEST" | tee -a "$JOB_FILE"
        done
    done < <(
        python - "$SHORTLIST" "$POSES_PER_ORDER" <<'PY'
import json
import sys
from pathlib import Path

shortlist = Path(sys.argv[1])
limit = int(sys.argv[2])
data = json.loads(shortlist.read_text())
selected = data["shortlist"][:limit]
if len(selected) < limit:
    raise SystemExit(
        f"{shortlist} contains only {len(selected)} poses; requested {limit}"
    )
for candidate in selected:
    manifest = Path(candidate["directory"]) / "manifest.json"
    print(
        candidate["ensemble_rank"],
        candidate["pose_seed"],
        manifest,
        sep="\t",
    )
PY
    )
done

echo "Submitted H100 screen jobs recorded in: $JOB_FILE"
