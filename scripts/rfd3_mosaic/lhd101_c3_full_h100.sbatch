#!/bin/bash
#SBATCH --job-name=rfd3-c3-h100
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
# The script redirects stdout/stderr into $RUN_ROOT/$SLURM_JOB_ID after it
# creates that directory.  Slurm's early bootstrap stream can be discarded.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_ROOT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c3_full_h100
CHECKPOINT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
ADAPTER_DIR="$RUN_DIR/adapter"
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
SEED="${RFD3_SEED:-44}"
POSE_CANDIDATE_MANIFEST="${RFD3_POSE_CANDIDATE_MANIFEST:-}"
LINKER_LENGTH="${RFD3_LINKER_LENGTH:-}"

if [[ -n "$POSE_CANDIDATE_MANIFEST" ]]; then
    ADAPTER_POSE_ARGS=(
        --pose-candidate-manifest "$POSE_CANDIDATE_MANIFEST"
    )
else
    ADAPTER_POSE_ARGS=(--pose-seed "$SEED")
fi
if [[ -n "$LINKER_LENGTH" ]]; then
    if [[ ! "$LINKER_LENGTH" =~ ^[0-9]+$ ]]; then
        echo "ERROR: RFD3_LINKER_LENGTH must be a non-negative integer"
        exit 2
    fi
    ADAPTER_LINKER_ARGS=(--linker-length "$LINKER_LENGTH")
else
    ADAPTER_LINKER_ARGS=()
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"
export FOUNDRY_CHECKPOINT_DIRS=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints

mkdir -p "$RUN_DIR" "$ADAPTER_DIR"
exec >"$RUN_DIR/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
    2>"$RUN_DIR/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"

echo "=== RFD3 Mosaic native C3 H100 full baseline, seed $SEED ==="
echo "RFD3_POSE_CANDIDATE_MANIFEST=${POSE_CANDIDATE_MANIFEST:-<seed-sampled>}"
echo "RFD3_LINKER_LENGTH=${LINKER_LENGTH:-<config-midpoint>}"
date
hostname
if ! nvidia-smi; then
    echo "WARNING: nvidia-smi returned nonzero; continuing to CUDA runtime checks"
fi
python --version
python -c "import rfd3; print(rfd3.__file__)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); print('capability', torch.cuda.get_device_capability(0))"

python -m rfd3_mosaic.rfd3_adapter \
    --config "$CONFIG" \
    --output-dir "$ADAPTER_DIR" \
    --base-directory "$PROJECT_DIR" \
    "${ADAPTER_LINKER_ARGS[@]}" \
    "${ADAPTER_POSE_ARGS[@]}"

ls -lh "$CHECKPOINT" "$ADAPTER_DIR/rfd3_input.json" "$ADAPTER_DIR/presymmetrized_input.cif"

python -m rfd3_mosaic.rfd3_prevalidate \
    --input "$ADAPTER_DIR/rfd3_input.json" \
    --report "$RUN_DIR/rfd3_prevalidation.json"

echo "RFD3_MOSAIC_INFERENCE_SAMPLER_KIND=symmetry"
echo "RFD3_MOSAIC_SYMMETRY_STATE_MODE=orbit_average"
echo "RFD3_MOSAIC_SYMMETRY_NOISE_MODE=coupled"
echo "RFD3_MOSAIC_REALIGNMENT=disabled"
echo "RFD3_MOSAIC_STATIC_MASTER_MOTIF=enabled"
echo "RFD3_MOSAIC_INTERFACE_CONSTRAINT_GROUPS=required"
sha256sum "$CONFIG" "$ADAPTER_DIR/rfd3_input.json" "$ADAPTER_DIR/presymmetrized_input.cif"

# Exact orbit mode is intentionally frame-fixed; see the full-single script.
python -m rfd3.run_inference \
    inference_sampler.kind=symmetry \
    out_dir="$RUN_DIR" \
    inputs="$ADAPTER_DIR/rfd3_input.json" \
    ckpt_path="$CHECKPOINT" \
    'json_keys_subset=[lhd101_c3_interface_seed]' \
    seed="$SEED" \
    diffusion_batch_size=1 \
    n_batches=1 \
    inference_sampler.num_timesteps=200 \
    inference_sampler.allow_realignment=False \
    ++inference_sampler.preserve_fixed_motif_during_symmetry=True \
    ++inference_sampler.require_motif_constraint_groups=True \
    ++inference_sampler.symmetry_state_mode=orbit_average \
    ++inference_sampler.symmetry_noise_mode=coupled \
    low_memory_mode=False \
    skip_existing=False \
    dump_trajectories=False \
    prevalidate_inputs=True

RESULT_JSON=$(find "$RUN_DIR" -maxdepth 1 -type f \
    -name '*model_0.json' -print -quit)
if [[ -z "$RESULT_JSON" ]]; then
    echo "ERROR: RFD3 produced no model_0 metadata JSON"
    exit 1
fi
python -m rfd3_mosaic.rfd3_seed_audit \
    --result-json "$RESULT_JSON" \
    --adapter-mapping "$ADAPTER_DIR/mapping.json" \
    --config "$CONFIG" \
    --base-directory "$PROJECT_DIR" \
    --output "$RUN_DIR/seed_integrity_audit.json" \
    --report-only

python -m rfd3_mosaic.rfd3_scaffold_audit \
    --result-json "$RESULT_JSON" \
    --rfd3-input "$ADAPTER_DIR/rfd3_input.json" \
    --output "$RUN_DIR/scaffold_validity_audit.json" \
    --expected-symmetry-multiplicity 3 \
    --report-only

python -m rfd3_mosaic.rfd3_audit_gate \
    --report "$RUN_DIR/seed_integrity_audit.json" \
    --report "$RUN_DIR/scaffold_validity_audit.json"

echo "=== H100 native C3 full baseline completed ==="
date
find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
