#!/bin/bash
#SBATCH --job-name=rfd3-c3-full
#SBATCH --partition=lrz-dgx-1-v100x8,lrz-dgx-1-p100x8,lrz-hpe-p100x4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/slurm-%x-%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/slurm-%x-%j.err

set -euo pipefail

source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_ROOT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c3_full
CHECKPOINT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"
ADAPTER_DIR="$RUN_DIR/adapter"
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
SEED="${RFD3_SEED:-42}"
POSE_CANDIDATE_MANIFEST="${RFD3_POSE_CANDIDATE_MANIFEST:-}"

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

echo "=== RFD3 Mosaic native C3 full single-design baseline ==="
date
hostname
nvidia-smi
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
sha256sum "$CONFIG" "$ADAPTER_DIR/rfd3_input.json" "$ADAPTER_DIR/presymmetrized_input.cif"

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
    low_memory_mode=True \
    skip_existing=False \
    dump_trajectories=False \
    prevalidate_inputs=True

echo "=== Native C3 full baseline completed ==="
date
find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
