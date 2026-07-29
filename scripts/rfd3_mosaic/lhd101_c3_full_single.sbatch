#!/bin/bash
#SBATCH --job-name=rfd3-c3-full
#SBATCH --partition=lrz-dgx-1-v100x8,lrz-dgx-1-p100x8,lrz-hpe-p100x4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# The script redirects stdout/stderr into $RUN_ROOT/$SLURM_JOB_ID after it
# creates that directory.  Slurm's early bootstrap stream can be discarded.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_ROOT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c3_full
CHECKPOINT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
ADAPTER_DIR="$RUN_DIR/adapter"
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
SEED="${RFD3_SEED:-42}"
POSE_CANDIDATE_MANIFEST="${RFD3_POSE_CANDIDATE_MANIFEST:-}"
COMPACTNESS_WEIGHT="${RFD3_COMPACTNESS_WEIGHT:-0.0}"
COMPACTNESS_END_FRAC="${RFD3_COMPACTNESS_END_FRAC:-0.75}"
COMPACTNESS_MAX_STEP="${RFD3_COMPACTNESS_MAX_STEP:-0.5}"

if [[ -n "$POSE_CANDIDATE_MANIFEST" ]]; then
    ADAPTER_POSE_ARGS=(
        --pose-candidate-manifest "$POSE_CANDIDATE_MANIFEST"
    )
else
    ADAPTER_POSE_ARGS=(--pose-seed "$SEED")
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

echo "=== RFD3 Mosaic native C3 full single-design baseline ==="
echo "RFD3_SEED=$SEED"
echo "RFD3_POSE_CANDIDATE_MANIFEST=${POSE_CANDIDATE_MANIFEST:-<seed-sampled>}"
echo "RFD3_COMPACTNESS_WEIGHT=$COMPACTNESS_WEIGHT"
echo "RFD3_COMPACTNESS_END_FRAC=$COMPACTNESS_END_FRAC"
echo "RFD3_COMPACTNESS_MAX_STEP=$COMPACTNESS_MAX_STEP"
date
hostname
if ! nvidia-smi; then
    echo "WARNING: nvidia-smi returned nonzero; continuing to CUDA runtime checks"
fi
python --version
python -c "import rfd3; print(rfd3.__file__)"

python -m rfd3_mosaic.rfd3_adapter \
    --config "$CONFIG" \
    --output-dir "$ADAPTER_DIR" \
    --base-directory "$PROJECT_DIR" \
    "${ADAPTER_POSE_ARGS[@]}"

ls -lh "$CHECKPOINT" "$ADAPTER_DIR/rfd3_input.json" "$ADAPTER_DIR/presymmetrized_input.cif"

python -m rfd3_mosaic.rfd3_prevalidate \
    --input "$ADAPTER_DIR/rfd3_input.json" \
    --report "$RUN_DIR/rfd3_prevalidation.json"

echo "RFD3_MOSAIC_INFERENCE_SAMPLER_KIND=symmetry"
echo "RFD3_MOSAIC_FIXED_MOTIF_REINSERTION=enabled"
echo "RFD3_MOSAIC_STEPWISE_FIXED_MOTIF_PRESERVATION=enabled"
sha256sum "$CONFIG" "$ADAPTER_DIR/rfd3_input.json" "$ADAPTER_DIR/presymmetrized_input.cif"

# RFD3 only reinserts indexed fixed motifs during sampling when
# allow_realignment is enabled; keep the original interface seed rigid.
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
    inference_sampler.allow_realignment=True \
    +inference_sampler.insert_motif_at_end=True \
    ++inference_sampler.preserve_fixed_motif_during_symmetry=True \
    ++inference_sampler.interface_seed_compactness_weight="$COMPACTNESS_WEIGHT" \
    ++inference_sampler.interface_seed_compactness_end_frac="$COMPACTNESS_END_FRAC" \
    ++inference_sampler.interface_seed_compactness_max_step="$COMPACTNESS_MAX_STEP" \
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
python -m rfd3_mosaic.rfd3_seed_audit \
    --result-json "$RESULT_JSON" \
    --adapter-mapping "$ADAPTER_DIR/mapping.json" \
    --config "$CONFIG" \
    --base-directory "$PROJECT_DIR" \
    --output "$RUN_DIR/seed_integrity_audit.json" \
    --report-only

python -m rfd3_mosaic.rfd3_scaffold_audit \
    --result-json "$RESULT_JSON" \
    --output "$RUN_DIR/scaffold_validity_audit.json" \
    --report-only

echo "=== Native C3 guided diagnostic completed ==="
date
find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
