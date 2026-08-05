#!/bin/bash -l
#SBATCH --job-name=prism-c3-fixed
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_BASE=${RFD3_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}
CHECKPOINT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/cyclic/prism_c3_g2_fixed.yaml"
SOURCE_PDB="$PROJECT_DIR/examples/rfd3_mosaic/inputs/Prism_C3_G2_fixed_motif.pdb"

PROFILE="${RFD3_ACCELERATOR_LABEL:?Set RFD3_ACCELERATOR_LABEL to h100, a100, or p100}"
SEED="${RFD3_SEED:-42}"
NUM_TIMESTEPS="${RFD3_NUM_TIMESTEPS:-200}"
CAMPAIGN="${RFD3_RUN_CAMPAIGN:-prism_c3_g2_fixed_matrix}"
EXAMPLE_ID="prism_c3_g2_fixed"
RUN_DIR="$RUN_BASE/$CAMPAIGN/$PROFILE/seed_$SEED/$SLURM_JOB_ID"
ADAPTER_DIR="$RUN_DIR/adapter"

if [[ ! "$PROFILE" =~ ^(h100|a100|p100)$ ]]; then
    echo "ERROR: RFD3_ACCELERATOR_LABEL must be h100, a100, or p100"
    exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_SEED must be a non-negative integer"
    exit 2
fi
if [[ ! "$NUM_TIMESTEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( NUM_TIMESTEPS < 2 || NUM_TIMESTEPS > 200 )); then
    echo "ERROR: RFD3_NUM_TIMESTEPS must be an integer from 2 to 200"
    exit 2
fi
for required in "$CHECKPOINT" "$CONFIG" "$SOURCE_PDB"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required file does not exist: $required"
        exit 2
    fi
done
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: refusing to reuse run directory: $RUN_DIR"
    exit 2
fi

mkdir -p "$RUN_DIR" "$ADAPTER_DIR"
exec >"$RUN_DIR/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
    2>"$RUN_DIR/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"
export FOUNDRY_CHECKPOINT_DIRS=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints

echo "=== Prism C3 G2 exact fixed-orbit Mosaic run ==="
echo "profile=$PROFILE seed=$SEED timesteps=$NUM_TIMESTEPS"
echo "config=$CONFIG"
echo "source_pdb=$SOURCE_PDB"
date
hostname
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

python -m rfd3_mosaic.rfd3_adapter \
    --config "$CONFIG" \
    --output-dir "$ADAPTER_DIR" \
    --base-directory "$PROJECT_DIR" \
    --example-id "$EXAMPLE_ID" \
    --linker-length 5

python -m rfd3_mosaic.rfd3_prevalidate \
    --input "$ADAPTER_DIR/rfd3_input.json" \
    --report "$RUN_DIR/rfd3_prevalidation.json"

python -m rfd3.run_inference \
    inference_sampler.kind=symmetry \
    out_dir="$RUN_DIR" \
    inputs="$ADAPTER_DIR/rfd3_input.json" \
    ckpt_path="$CHECKPOINT" \
    "json_keys_subset=[$EXAMPLE_ID]" \
    seed="$SEED" \
    diffusion_batch_size=1 \
    n_batches=1 \
    inference_sampler.num_timesteps="$NUM_TIMESTEPS" \
    inference_sampler.allow_realignment=False \
    ++inference_sampler.fixed_motif_finalization_mode=motif_precedence \
    ++inference_sampler.preserve_fixed_motif_during_symmetry=True \
    ++inference_sampler.require_motif_constraint_groups=True \
    ++inference_sampler.symmetry_state_mode=orbit_average \
    ++inference_sampler.symmetry_noise_mode=coupled \
    ++inference_sampler.enable_orbit_rigid_motif_mobility=False \
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
    --probe-input "$ADAPTER_DIR/rfd3_input.json" \
    --result-json "$RESULT_JSON" \
    --max-joint-rmsd 0.001 \
    --max-joint-coordinate-error 0.001 \
    --output "$RUN_DIR/fixed_orbit_audit.json" \
    --report-only

python -m rfd3_mosaic.rfd3_scaffold_audit \
    --result-json "$RESULT_JSON" \
    --rfd3-input "$ADAPTER_DIR/rfd3_input.json" \
    --output "$RUN_DIR/scaffold_validity_audit.json" \
    --expected-symmetry-multiplicity 3 \
    --report-only

python -m rfd3_mosaic.rfd3_audit_gate \
    --report "$RUN_DIR/fixed_orbit_audit.json" \
    --report "$RUN_DIR/scaffold_validity_audit.json"

echo "=== Prism C3 G2 exact fixed-orbit run completed ==="
date

