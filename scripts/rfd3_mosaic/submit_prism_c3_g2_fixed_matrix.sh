#!/bin/bash

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
SBATCH_SCRIPT="$PROJECT_DIR/scripts/rfd3_mosaic/prism_c3_g2_fixed_mosaic.sbatch"
PROFILES="${RFD3_PROFILES:-h100 a100 p100}"
SEEDS="${RFD3_SEEDS:-42}"
NUM_TIMESTEPS="${RFD3_NUM_TIMESTEPS:-200}"
CAMPAIGN="${RFD3_RUN_CAMPAIGN:-prism_c3_g2_fixed_matrix}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script does not exist: $SBATCH_SCRIPT"
    exit 2
fi

for profile in $PROFILES; do
    case "$profile" in
        h100)
            partition=lrz-hgx-h100-94x4
            cpus=12
            memory=440G
            walltime=08:00:00
            ;;
        a100)
            partition=lrz-dgx-a100-80x8
            cpus=12
            memory=440G
            walltime=08:00:00
            ;;
        p100)
            partition=lrz-dgx-1-p100x8,lrz-hpe-p100x4
            cpus=8
            memory=120G
            walltime=08:00:00
            ;;
        *)
            echo "ERROR: unsupported profile: $profile"
            exit 2
            ;;
    esac
    for seed in $SEEDS; do
        if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
            echo "ERROR: invalid seed: $seed"
            exit 2
        fi
        job_name="prism-c3-${profile}-s${seed}"
        sbatch \
            --job-name="$job_name" \
            --partition="$partition" \
            --cpus-per-task="$cpus" \
            --mem="$memory" \
            --time="$walltime" \
            --export="ALL,RFD3_ACCELERATOR_LABEL=$profile,RFD3_SEED=$seed,RFD3_NUM_TIMESTEPS=$NUM_TIMESTEPS,RFD3_RUN_CAMPAIGN=$CAMPAIGN" \
            "$SBATCH_SCRIPT"
    done
done

