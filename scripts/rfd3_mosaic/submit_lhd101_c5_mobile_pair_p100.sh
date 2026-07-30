#!/bin/bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE=${RFD3_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}
DEFAULT_MANIFEST="$RUN_BASE/lhd101_c5_mobile_lhs_v1/candidate_0419_seed_3419/manifest.json"
POSE_CANDIDATE_MANIFEST=${RFD3_POSE_CANDIDATE_MANIFEST:-"$DEFAULT_MANIFEST"}
P100_PARTITIONS=${RFD3_P100_PARTITIONS:-lrz-dgx-1-p100x8,lrz-hpe-p100x4}
DIFFUSION_SEED=${RFD3_SEED:-42}
UPDATE_INTERVAL=${RFD3_MOBILITY_UPDATE_INTERVAL:-5}
TARGET_MAX_TILT=${RFD3_MOBILITY_TARGET_MAX_TILT_DEGREES:-20.0}
LINKER_LENGTH=${RFD3_LINKER_LENGTH:-}
TIMESTEPS=${RFD3_MOBILE_TIMESTEPS:-"50 200"}
MODES=${RFD3_MOBILE_MODES:-"proposal applied"}
WALLTIME_50=${RFD3_MOBILE_WALLTIME_50:-12:00:00}
WALLTIME_200=${RFD3_MOBILE_WALLTIME_200:-24:00:00}
JOB_FILE=${RFD3_MOBILE_JOB_FILE:-"$RUN_BASE/c5_mobile_seed3419_p100_v1.tsv"}
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/experimental/lhd101_c5_mobile.yaml"
PILOT_SCRIPT="$PROJECT_DIR/scripts/rfd3_mosaic/lhd101_c5_mobile_pilot_p100.sbatch"

if [[ ! "$DIFFUSION_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_SEED must be a non-negative integer"
    exit 2
fi
if [[ ! "$UPDATE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RFD3_MOBILITY_UPDATE_INTERVAL must be a positive integer"
    exit 2
fi
if [[ ! "$TARGET_MAX_TILT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "ERROR: RFD3_MOBILITY_TARGET_MAX_TILT_DEGREES must be non-negative"
    exit 2
fi
if [[ -n "$LINKER_LENGTH" && ! "$LINKER_LENGTH" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_LINKER_LENGTH must be a non-negative integer"
    exit 2
fi
if [[ ! -f "$POSE_CANDIDATE_MANIFEST" ]]; then
    echo "ERROR: pose candidate manifest does not exist:"
    echo "$POSE_CANDIDATE_MANIFEST"
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
if [[ ! -f "$CONFIG" || ! -f "$PILOT_SCRIPT" ]]; then
    echo "ERROR: tracked C5 mobility config or pilot script is missing"
    exit 2
fi
for STEP_COUNT in $TIMESTEPS; do
    if [[ ! "$STEP_COUNT" =~ ^(50|200)$ ]]; then
        echo "ERROR: RFD3_MOBILE_TIMESTEPS may contain only 50 and 200"
        exit 2
    fi
done
for MODE in $MODES; do
    if [[ "$MODE" != "proposal" && "$MODE" != "applied" ]]; then
        echo "ERROR: RFD3_MOBILE_MODES may contain only proposal and applied"
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

read -r POSE_SEED TILT_DEGREES < <(
    "$PYTHON_BIN" - "$POSE_CANDIDATE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
sample = manifest.get("initialization_samples", {}).get("primary_seed", {})
pose_seed = manifest.get("pose_seed", sample.get("random_seed"))
tilt = sample.get("principal_axis_tilt_deg")
if pose_seed is None:
    raise SystemExit("ERROR: candidate manifest has no pose seed")
if tilt is None:
    raise SystemExit("ERROR: candidate manifest has no principal-axis tilt")
print(pose_seed, f"{float(tilt):.6f}")
PY
)

MANIFEST_SHA256=$(sha256sum "$POSE_CANDIDATE_MANIFEST" | awk '{print $1}')
CONFIG_SHA256=$(sha256sum "$CONFIG" | awk '{print $1}')
PILOT_SCRIPT_SHA256=$(sha256sum "$PILOT_SCRIPT" | awk '{print $1}')
LINKER_LABEL=${LINKER_LENGTH:-config-midpoint}
EXPERIMENT_FINGERPRINT=$(
    printf '%s\0' \
        "$MANIFEST_SHA256" \
        "$CONFIG_SHA256" \
        "$PILOT_SCRIPT_SHA256" \
        "$POSE_SEED" \
        "$DIFFUSION_SEED" \
        "$UPDATE_INTERVAL" \
        "$TARGET_MAX_TILT" \
        "$LINKER_LABEL" |
        sha256sum |
        awk '{print $1}'
)

EXPECTED_HEADER=$'mode\tpose_seed\ttilt_deg\tdiffusion_seed\tsteps\tupdate_interval\ttarget_max_tilt_deg\tlinker_length\tmanifest_sha256\tconfig_sha256\tpilot_script_sha256\texperiment_fingerprint\tjob_id\tdependency\tmanifest'
if [[ -f "$JOB_FILE" ]]; then
    OBSERVED_HEADER=$(head -n 1 "$JOB_FILE")
    if [[ "$OBSERVED_HEADER" != "$EXPECTED_HEADER" ]]; then
        echo "ERROR: existing job file has an incompatible header: $JOB_FILE"
        exit 2
    fi
    if ! awk -F '\t' \
        -v fingerprint="$EXPERIMENT_FINGERPRINT" \
        'NR > 1 && $12 != fingerprint { mismatch = 1 }
         END { exit(mismatch ? 1 : 0) }' \
        "$JOB_FILE"; then
        echo "ERROR: existing job file belongs to a different experiment:"
        echo "$JOB_FILE"
        echo "Use the original parameters or choose a new RFD3_MOBILE_JOB_FILE."
        exit 2
    fi
    echo "Resuming C5 mobility pair from: $JOB_FILE"
else
    printf '%s\n' "$EXPECTED_HEADER" >"$JOB_FILE"
fi

echo "C5 P100 mobility comparison"
echo "manifest=$POSE_CANDIDATE_MANIFEST"
echo "pose_seed=$POSE_SEED"
echo "principal_axis_tilt_deg=$TILT_DEGREES"
echo "diffusion_seed=$DIFFUSION_SEED"
echo "update_interval=$UPDATE_INTERVAL"
echo "target_max_tilt_deg=$TARGET_MAX_TILT"
echo "linker_length=$LINKER_LABEL"
echo "manifest_sha256=$MANIFEST_SHA256"
echo "config_sha256=$CONFIG_SHA256"
echo "pilot_script_sha256=$PILOT_SCRIPT_SHA256"
echo "experiment_fingerprint=$EXPERIMENT_FINGERPRINT"
echo "modes=$MODES"
echo "timesteps=$TIMESTEPS"
echo "partitions=$P100_PARTITIONS"
echo "job_file=$JOB_FILE"

# Submit both 50-step controls first. Each 200-step job receives an afterok
# dependency on the matching 50-step mode, so a failed short validation cannot
# silently consume a full-length P100 allocation.
for STEP_COUNT in $TIMESTEPS; do
    case "$STEP_COUNT" in
        50)
            WALLTIME=$WALLTIME_50
            ;;
        200)
            WALLTIME=$WALLTIME_200
            ;;
    esac

    for MODE in $MODES; do
        if [[ "$MODE" == "applied" ]]; then
            APPLY_UPDATES=true
            MODE_CODE=a
        else
            APPLY_UPDATES=false
            MODE_CODE=p
        fi

        if awk -F '\t' \
            -v mode="$MODE" \
            -v steps="$STEP_COUNT" \
            -v fingerprint="$EXPERIMENT_FINGERPRINT" \
            'NR > 1 && $1 == mode && $5 == steps &&
             $12 == fingerprint { found = 1 }
             END { exit(found ? 0 : 1) }' \
            "$JOB_FILE"; then
            echo "Already submitted: mode=$MODE seed=$DIFFUSION_SEED steps=$STEP_COUNT"
            continue
        fi

        DEPENDENCY=none
        SBATCH_DEPENDENCY=()
        if [[ "$STEP_COUNT" == "200" ]]; then
            SHORT_JOB_ID=$(awk -F '\t' \
                -v mode="$MODE" \
                -v fingerprint="$EXPERIMENT_FINGERPRINT" \
                'NR > 1 && $1 == mode && $5 == 50 &&
                 $12 == fingerprint { print $13; exit }' \
                "$JOB_FILE")
            if [[ -z "$SHORT_JOB_ID" ]]; then
                echo "ERROR: no matching 50-step job is recorded for mode=$MODE"
                echo "Submit or resume the 50-step pair before the 200-step pair."
                exit 2
            fi
            DEPENDENCY="afterok:$SHORT_JOB_ID"
            SBATCH_DEPENDENCY=(--dependency="$DEPENDENCY")
        fi

        JOB_NAME="c5m${MODE_CODE}-p${POSE_SEED}-n${STEP_COUNT}"
        if ! JOB_ID=$(sbatch --parsable \
                --partition="$P100_PARTITIONS" \
                --job-name="$JOB_NAME" \
                --time="$WALLTIME" \
                "${SBATCH_DEPENDENCY[@]}" \
                --export="ALL,RFD3_NUM_TIMESTEPS=${STEP_COUNT},RFD3_SEED=${DIFFUSION_SEED},RFD3_MOBILITY_APPLY_UPDATES=${APPLY_UPDATES},RFD3_MOBILITY_UPDATE_INTERVAL=${UPDATE_INTERVAL},RFD3_MOBILITY_TARGET_MAX_TILT_DEGREES=${TARGET_MAX_TILT},RFD3_LINKER_LENGTH=${LINKER_LENGTH},RFD3_POSE_CANDIDATE_MANIFEST=${POSE_CANDIDATE_MANIFEST}" \
                scripts/rfd3_mosaic/lhd101_c5_mobile_pilot_p100.sbatch); then
            echo "Submission stopped, usually because of the Slurm/QOS job limit."
            echo "Resume with:"
            echo "RFD3_MOBILE_JOB_FILE=$JOB_FILE bash scripts/rfd3_mosaic/submit_lhd101_c5_mobile_pair_p100.sh"
            exit 75
        fi
        JOB_ID=${JOB_ID%%;*}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$MODE" \
            "$POSE_SEED" \
            "$TILT_DEGREES" \
            "$DIFFUSION_SEED" \
            "$STEP_COUNT" \
            "$UPDATE_INTERVAL" \
            "$TARGET_MAX_TILT" \
            "$LINKER_LABEL" \
            "$MANIFEST_SHA256" \
            "$CONFIG_SHA256" \
            "$PILOT_SCRIPT_SHA256" \
            "$EXPERIMENT_FINGERPRINT" \
            "$JOB_ID" \
            "$DEPENDENCY" \
            "$POSE_CANDIDATE_MANIFEST" | tee -a "$JOB_FILE"
    done
done

echo "Submitted C5 P100 mobility comparison recorded in: $JOB_FILE"
