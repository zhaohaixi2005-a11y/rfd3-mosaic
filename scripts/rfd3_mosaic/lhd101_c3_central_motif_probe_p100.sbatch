#!/bin/bash -l
#SBATCH --job-name=c3-central
#SBATCH --partition=lrz-dgx-1-p100x8,lrz-hpe-p100x4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic
CHECKPOINT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt

ARM="${RFD3_CENTRAL_PROBE_ARM:?Set RFD3_CENTRAL_PROBE_ARM to A, B, C, or D}"
TEMPLATE_INPUT="${RFD3_CENTRAL_TEMPLATE_INPUT:?Set RFD3_CENTRAL_TEMPLATE_INPUT to an existing C3 adapter rfd3_input.json}"
FIXED_SELECTOR="${RFD3_CENTRAL_FIXED_SELECTOR:-B1-31}"
N_LENGTH="${RFD3_CENTRAL_N_LENGTH:-35}"
C_LENGTH="${RFD3_CENTRAL_C_LENGTH:-35}"
NUM_TIMESTEPS="${RFD3_NUM_TIMESTEPS:-10}"
SEED="${RFD3_SEED:-42}"
CAMPAIGN="${RFD3_CENTRAL_CAMPAIGN:-central_motif_c3_abcd}"
EXAMPLE_ID=central_motif_c3_probe
RUN_DIR="$RUN_BASE/$CAMPAIGN/arm_${ARM}/$SLURM_JOB_ID"
INPUT_DIR="$RUN_DIR/input"

if [[ ! "$ARM" =~ ^[ABCD]$ ]]; then
    echo "ERROR: RFD3_CENTRAL_PROBE_ARM must be A, B, C, or D"
    exit 2
fi
if [[ ! -f "$TEMPLATE_INPUT" ]]; then
    echo "ERROR: template input does not exist: $TEMPLATE_INPUT"
    exit 2
fi
for value in "$N_LENGTH" "$C_LENGTH" "$NUM_TIMESTEPS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: terminal lengths and timesteps must be positive integers"
        exit 2
    fi
done

case "$ARM" in
    A)
        ARM_LABEL=official_original_realign
        FINALIZATION_MODE=official_reinsert_then_project
        REALIGNMENT=True
        PRESERVE=False
        REQUIRE_GROUPS=False
        STATE_MODE=legacy_asu
        NOISE_MODE=independent
        ;;
    B)
        ARM_LABEL=exact_mosaic
        FINALIZATION_MODE=motif_precedence
        REALIGNMENT=False
        PRESERVE=True
        REQUIRE_GROUPS=True
        STATE_MODE=orbit_average
        NOISE_MODE=coupled
        ;;
    C)
        ARM_LABEL=official_original_no_realign
        # Strict one-variable control for arm A: retain the official
        # reinsert-then-project finalization and legacy symmetry/noise state,
        # changing only allow_realignment from True to False.
        FINALIZATION_MODE=official_reinsert_then_project
        REALIGNMENT=False
        PRESERVE=False
        REQUIRE_GROUPS=False
        STATE_MODE=legacy_asu
        NOISE_MODE=independent
        ;;
    D)
        ARM_LABEL=legacy_complete_orbit_restore
        FINALIZATION_MODE=motif_precedence
        REALIGNMENT=False
        PRESERVE=True
        REQUIRE_GROUPS=True
        STATE_MODE=legacy_asu
        NOISE_MODE=independent
        ;;
esac

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"
export FOUNDRY_CHECKPOINT_DIRS=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints

if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: refusing to reuse run directory: $RUN_DIR"
    exit 2
fi
mkdir -p "$RUN_DIR" "$INPUT_DIR"
exec >"$RUN_DIR/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
    2>"$RUN_DIR/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"

echo "=== C3 central-motif controlled probe ==="
echo "arm=$ARM ($ARM_LABEL)"
echo "template=$TEMPLATE_INPUT"
echo "fixed_selector=$FIXED_SELECTOR"
echo "terminal_lengths=$N_LENGTH,$C_LENGTH"
echo "timesteps=$NUM_TIMESTEPS seed=$SEED"
echo "allow_realignment=$REALIGNMENT"
echo "fixed_motif_finalization_mode=$FINALIZATION_MODE"
echo "preserve_fixed_motif_during_symmetry=$PRESERVE"
echo "require_motif_constraint_groups=$REQUIRE_GROUPS"
echo "symmetry_state_mode=$STATE_MODE"
echo "symmetry_noise_mode=$NOISE_MODE"
date
hostname
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

python -m rfd3_mosaic.rfd3_central_motif_probe \
    --template-input "$TEMPLATE_INPUT" \
    --output-dir "$INPUT_DIR" \
    --fixed-selector "$FIXED_SELECTOR" \
    --n-terminal-length "$N_LENGTH" \
    --c-terminal-length "$C_LENGTH" \
    --example-id "$EXAMPLE_ID"

python -m rfd3_mosaic.rfd3_prevalidate \
    --input "$INPUT_DIR/rfd3_input.json" \
    --report "$RUN_DIR/rfd3_prevalidation.json"

python -m rfd3.run_inference \
    inference_sampler.kind=symmetry \
    out_dir="$RUN_DIR" \
    inputs="$INPUT_DIR/rfd3_input.json" \
    ckpt_path="$CHECKPOINT" \
    "json_keys_subset=[$EXAMPLE_ID]" \
    seed="$SEED" \
    diffusion_batch_size=1 \
    n_batches=1 \
    inference_sampler.num_timesteps="$NUM_TIMESTEPS" \
    inference_sampler.allow_realignment="$REALIGNMENT" \
    ++inference_sampler.fixed_motif_finalization_mode="$FINALIZATION_MODE" \
    ++inference_sampler.preserve_fixed_motif_during_symmetry="$PRESERVE" \
    ++inference_sampler.require_motif_constraint_groups="$REQUIRE_GROUPS" \
    ++inference_sampler.symmetry_state_mode="$STATE_MODE" \
    ++inference_sampler.symmetry_noise_mode="$NOISE_MODE" \
    low_memory_mode=True \
    skip_existing=False \
    dump_trajectories=False \
    prevalidate_inputs=True

RESULT_JSON=$(find "$RUN_DIR" -maxdepth 1 -type f \
    -name '*model_0.json' -print -quit)
if [[ -z "$RESULT_JSON" ]]; then
    echo "ERROR: RFD3 produced no model_0 metadata JSON"
    exit 1
fi
python -m rfd3_mosaic.rfd3_central_motif_audit \
    --probe-input "$INPUT_DIR/rfd3_input.json" \
    --result-json "$RESULT_JSON" \
    --output "$RUN_DIR/central_motif_audit.json" \
    --report-only

echo "=== C3 central-motif arm $ARM completed ==="
date
