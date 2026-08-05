#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${RFD3_A_PROJECT_DIR:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic}"
OUTPUT_ROOT="${RFD3_A_OUTPUT_ROOT:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}"
SBATCH_SCRIPT="$PROJECT_DIR/scripts/rfd3_mosaic/lhd101_c3_central_motif_probe_p100.sbatch"
CAMPAIGN="${RFD3_A_CAMPAIGN:-central-c3-official-A-8_4}"
PROFILE="${RFD3_A_PROFILE:-p100}"
SEEDS="${RFD3_A_SEEDS:-102 103 104}"
TEMPLATE_OVERRIDE="${RFD3_A_TEMPLATE_INPUT:-}"
B_CAMPAIGN_ROOT="${RFD3_A_B_CAMPAIGN_ROOT:-${OUTPUT_ROOT}/central-c3-batch5-registry-v4}"
DRY_RUN="${RFD3_A_DRY_RUN:-false}"

source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"

echo "A control: legacy RFD3 state with realignment explicitly enabled"
echo "seeds:     $SEEDS"
echo "steps:     200"
echo "profile:   $PROFILE"
echo "campaign:  $CAMPAIGN"
echo "dry run:   $DRY_RUN"

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "false" ]]; then
    echo "ERROR: RFD3_A_DRY_RUN must be true or false" >&2
    exit 2
fi

for seed in $SEEDS; do
    if [[ -n "$TEMPLATE_OVERRIDE" ]]; then
        template_input="$TEMPLATE_OVERRIDE"
    else
        case "$seed" in
            102)
                template_input="$B_CAMPAIGN_ROOT/central-n35-c35-s102/5729451/input/rfd3_input.json"
                ;;
            103)
                template_input="$B_CAMPAIGN_ROOT/central-n35-c35-s103/5729452/input/rfd3_input.json"
                ;;
            104)
                template_input="$B_CAMPAIGN_ROOT/central-n35-c35-s104/5729453/input/rfd3_input.json"
                ;;
            *)
                echo "ERROR: no frozen B input is registered for seed $seed; set RFD3_A_TEMPLATE_INPUT" >&2
                exit 2
                ;;
        esac
    fi
    if [[ ! -f "$template_input" ]]; then
        echo "ERROR: A-control template input does not exist: $template_input" >&2
        exit 2
    fi
    echo "seed $seed template: $template_input"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: arm=A realignment=True seed=$seed"
        continue
    fi
    sbatch \
        --job-name="c3-A-s${seed}" \
        --export="ALL,RFD3_CENTRAL_PROBE_ARM=A,RFD3_CENTRAL_TEMPLATE_INPUT=$template_input,RFD3_CENTRAL_FIXED_SELECTOR=B1-31,RFD3_CENTRAL_N_LENGTH=35,RFD3_CENTRAL_C_LENGTH=35,RFD3_NUM_TIMESTEPS=200,RFD3_SEED=$seed,RFD3_CENTRAL_CAMPAIGN=$CAMPAIGN" \
        "$SBATCH_SCRIPT"
done
