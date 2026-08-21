#!/bin/bash

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic
SBATCH_SCRIPT="$PROJECT_DIR/scripts/rfd3_mosaic/archive/legacy_direct/lhd101_cn_full_p100.sbatch"
POSE_ROOT="$RUN_BASE/lhd101_c5_qd_v2"
DIFFUSION_SEED="${RFD3_MATRIX_DIFFUSION_SEED:-44}"
RUN_TAG="${RFD3_MATRIX_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
JOB_FILE="$PROJECT_DIR/c5_attention_validation_${RUN_TAG}.tsv"

P100_PARTITIONS=lrz-dgx-1-p100x8,lrz-hpe-p100x4
H100_PARTITION=lrz-hgx-h100-94x4

if [[ ! "$DIFFUSION_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_MATRIX_DIFFUSION_SEED must be a non-negative integer"
    exit 2
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script missing: $SBATCH_SCRIPT"
    exit 2
fi

declare -a POSE_RECORDS=(
    "3063:$POSE_ROOT/candidate_0063_seed_3063/manifest.json:5722375"
    "3458:$POSE_ROOT/candidate_0458_seed_3458/manifest.json:5722380"
    "3145:$POSE_ROOT/candidate_0145_seed_3145/manifest.json:5722385"
)
declare -a STEP_COUNTS=(50 200)
declare -a ACCELERATORS=(P100 H100)

for record in "${POSE_RECORDS[@]}"; do
    IFS=: read -r pose_seed manifest baseline_job <<<"$record"
    if [[ ! -f "$manifest" ]]; then
        echo "ERROR: pose manifest missing: $manifest"
        exit 2
    fi
done

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    logical_id job_id accelerator pose_seed diffusion_seed timesteps \
    pre_fix_50step_p100_job manifest >"$JOB_FILE"

logical_index=0
for record in "${POSE_RECORDS[@]}"; do
    IFS=: read -r pose_seed manifest baseline_job <<<"$record"
    for timesteps in "${STEP_COUNTS[@]}"; do
        for accelerator in "${ACCELERATORS[@]}"; do
            logical_index=$((logical_index + 1))
            logical_id=$(printf 'A%02d' "$logical_index")
            if [[ "$accelerator" == P100 ]]; then
                partition="$P100_PARTITIONS"
            else
                partition="$H100_PARTITION"
            fi
            job_name=$(printf 'a%02d-c5-p%s-t%s-%s' \
                "$logical_index" "$pose_seed" "$timesteps" \
                "${accelerator,,}")
            if submit_output=$(
                sbatch \
                    --job-name="$job_name" \
                    --partition="$partition" \
                    --export="ALL,RFD3_CYCLIC_ORDER=5,RFD3_NUM_TIMESTEPS=${timesteps},RFD3_SEED=${DIFFUSION_SEED},RFD3_POSE_CANDIDATE_MANIFEST=${manifest},RFD3_ACCELERATOR_LABEL=${accelerator}" \
                    "$SBATCH_SCRIPT"
            ); then
                job_id=${submit_output##* }
            else
                job_id=SUBMIT_FAILED
            fi
            comparison_job=-
            if [[ "$accelerator" == P100 && "$timesteps" == 50 ]]; then
                comparison_job="$baseline_job"
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$logical_id" "$job_id" "$accelerator" "$pose_seed" \
                "$DIFFUSION_SEED" "$timesteps" "$comparison_job" \
                "$manifest" | tee -a "$JOB_FILE"
        done
    done
done

echo "Job manifest: $JOB_FILE"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$JOB_FILE"
else
    cat "$JOB_FILE"
fi
