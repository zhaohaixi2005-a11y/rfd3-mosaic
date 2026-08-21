#!/bin/bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE=${RFD3_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}
ENSEMBLE=${RFD3_C5_ENSEMBLE:-"$RUN_BASE/lhd101_c5_mobile_lhs_v1/pose_ensemble.json"}
P100_PARTITIONS=${RFD3_P100_PARTITIONS:-lrz-dgx-1-p100x8,lrz-hpe-p100x4}
POSES_PER_BIN=${RFD3_LOW_TILT_POSES_PER_BIN:-1}
DIFFUSION_SEED=${RFD3_LOW_TILT_DIFFUSION_SEED:-42}
TIMESTEPS=${RFD3_LOW_TILT_TIMESTEPS:-"50 200"}
WALLTIME_50=${RFD3_LOW_TILT_WALLTIME_50:-12:00:00}
WALLTIME_200=${RFD3_LOW_TILT_WALLTIME_200:-24:00:00}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_FILE=${RFD3_LOW_TILT_JOB_FILE:-"$RUN_BASE/c5_low_tilt_p100_${TIMESTAMP}.tsv"}

if [[ ! "$POSES_PER_BIN" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RFD3_LOW_TILT_POSES_PER_BIN must be a positive integer"
    exit 2
fi
if [[ ! "$DIFFUSION_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_LOW_TILT_DIFFUSION_SEED must be a non-negative integer"
    exit 2
fi
case "$P100_PARTITIONS" in
    lrz-dgx-1-p100x8 | \
    lrz-hpe-p100x4 | \
    lrz-dgx-1-p100x8,lrz-hpe-p100x4 | \
    lrz-hpe-p100x4,lrz-dgx-1-p100x8)
        ;;
    *)
        echo "ERROR: RFD3_P100_PARTITIONS may contain only LRZ P100 partitions"
        exit 2
        ;;
esac
if [[ ! -f "$ENSEMBLE" ]]; then
    echo "ERROR: C5 pose ensemble does not exist: $ENSEMBLE"
    exit 2
fi
for STEP_COUNT in $TIMESTEPS; do
    if [[ ! "$STEP_COUNT" =~ ^(50|200)$ ]]; then
        echo "ERROR: RFD3_LOW_TILT_TIMESTEPS may contain only 50 and 200"
        exit 2
    fi
done

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "ERROR: neither python nor python3 is available"
    exit 2
fi

EXPECTED_HEADER=$'tilt_bin\tpose_rank\tpose_seed\ttilt_deg\tdiffusion_seed\tsteps\tjob_id\tdependency\tmanifest'
if [[ -f "$JOB_FILE" ]]; then
    OBSERVED_HEADER=$(head -n 1 "$JOB_FILE")
    if [[ "$OBSERVED_HEADER" != "$EXPECTED_HEADER" ]]; then
        echo "ERROR: existing job file has an incompatible header: $JOB_FILE"
        exit 2
    fi
    echo "Resuming low-tilt P100 batch from: $JOB_FILE"
else
    printf '%s\n' "$EXPECTED_HEADER" >"$JOB_FILE"
fi

SELECTION_FILE=$(mktemp)
trap 'rm -f "$SELECTION_FILE"' EXIT

"$PYTHON_BIN" - "$ENSEMBLE" "$POSES_PER_BIN" >"$SELECTION_FILE" <<'PY'
import json
import sys
from pathlib import Path

ensemble_path = Path(sys.argv[1])
poses_per_bin = int(sys.argv[2])
data = json.loads(ensemble_path.read_text())

bins = (
    ("0-10", 0.0, 10.0, False),
    ("10-20", 10.0, 20.0, False),
    ("20-30", 20.0, 30.0, True),
)
ranking = data["ranking"]

for label, lower, upper, include_upper in bins:
    selected = []
    for rank, candidate in enumerate(ranking, start=1):
        if not candidate.get("accepted", False):
            continue
        tilt = candidate.get("maximum_principal_axis_tilt_deg")
        if tilt is None:
            continue
        tilt = float(tilt)
        inside = lower <= tilt < upper
        if include_upper:
            inside = lower <= tilt <= upper
        if not inside:
            continue
        selected.append((rank, candidate, tilt))
        if len(selected) == poses_per_bin:
            break
    if len(selected) != poses_per_bin:
        raise SystemExit(
            f"ERROR: tilt bin {label} contains only {len(selected)} accepted "
            f"poses; requested {poses_per_bin}"
        )
    for rank, candidate, tilt in selected:
        manifest = Path(candidate["directory"]) / "manifest.json"
        print(
            label,
            rank,
            candidate["pose_seed"],
            f"{tilt:.6f}",
            manifest,
            sep="\t",
        )
PY

echo "Selected C5 poses:"
column -t -s $'\t' "$SELECTION_FILE" 2>/dev/null || cat "$SELECTION_FILE"

# Submit all 50-step jobs first so short diagnostics enter the queue before
# the corresponding 200-step jobs.
for STEP_COUNT in $TIMESTEPS; do
    case "$STEP_COUNT" in
        50)
            WALLTIME=$WALLTIME_50
            ;;
        200)
            WALLTIME=$WALLTIME_200
            ;;
    esac

    while IFS=$'\t' read -r TILT_BIN POSE_RANK POSE_SEED TILT MANIFEST; do
        if [[ ! -f "$MANIFEST" ]]; then
            echo "ERROR: candidate manifest does not exist: $MANIFEST"
            exit 2
        fi
        if awk -F '\t' \
            -v pose_seed="$POSE_SEED" \
            -v manifest="$MANIFEST" \
            -v diffusion_seed="$DIFFUSION_SEED" \
            -v steps="$STEP_COUNT" \
            'NR > 1 && $3 == pose_seed && $5 == diffusion_seed &&
             $6 == steps && $9 == manifest { found = 1 }
             END { exit(found ? 0 : 1) }' \
            "$JOB_FILE"; then
            echo "Already submitted: pose=$POSE_SEED seed=$DIFFUSION_SEED steps=$STEP_COUNT"
            continue
        fi

        DEPENDENCY=none
        SBATCH_DEPENDENCY=()
        if [[ "$STEP_COUNT" == "200" ]]; then
            SHORT_JOB_ID=$(awk -F '\t' \
                -v pose_seed="$POSE_SEED" \
                -v manifest="$MANIFEST" \
                -v diffusion_seed="$DIFFUSION_SEED" \
                'NR > 1 && $3 == pose_seed && $5 == diffusion_seed &&
                 $6 == 50 && $9 == manifest { print $7; exit }' \
                "$JOB_FILE")
            if [[ -z "$SHORT_JOB_ID" ]]; then
                echo "ERROR: no matching 50-step job is recorded for pose=$POSE_SEED"
                exit 2
            fi
            DEPENDENCY="afterok:$SHORT_JOB_ID"
            SBATCH_DEPENDENCY=(--dependency="$DEPENDENCY")
        fi

        JOB_NAME="c5m-${TILT_BIN}-s${DIFFUSION_SEED}-n${STEP_COUNT}"
        if ! JOB_ID=$(sbatch --parsable \
                --partition="$P100_PARTITIONS" \
                --job-name="$JOB_NAME" \
                --time="$WALLTIME" \
                "${SBATCH_DEPENDENCY[@]}" \
                --export="ALL,RFD3_NUM_TIMESTEPS=${STEP_COUNT},RFD3_SEED=${DIFFUSION_SEED},RFD3_MOBILITY_APPLY_UPDATES=false,RFD3_POSE_CANDIDATE_MANIFEST=${MANIFEST}" \
                scripts/rfd3_mosaic/archive/legacy_direct/lhd101_c5_mobile_pilot_p100.sbatch); then
            echo "Submission stopped, usually because of the Slurm/QOS job limit."
            echo "Resume with:"
            echo "RFD3_LOW_TILT_JOB_FILE=$JOB_FILE bash scripts/rfd3_mosaic/archive/legacy_direct/submit_lhd101_c5_low_tilt_p100.sh"
            exit 75
        fi
        JOB_ID=${JOB_ID%%;*}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$TILT_BIN" \
            "$POSE_RANK" \
            "$POSE_SEED" \
            "$TILT" \
            "$DIFFUSION_SEED" \
            "$STEP_COUNT" \
            "$JOB_ID" \
            "$DEPENDENCY" \
            "$MANIFEST" | tee -a "$JOB_FILE"
    done <"$SELECTION_FILE"
done

echo "Submitted low-tilt P100 jobs recorded in: $JOB_FILE"
