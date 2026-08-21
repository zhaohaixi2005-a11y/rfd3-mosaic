#!/bin/bash
#SBATCH --job-name=c5-mobility
#SBATCH --partition=lrz-dgx-1-p100x8,lrz-hpe-p100x4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic
CHECKPOINT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
CONFIG="$PROJECT_DIR/configs/rfd3_mosaic/experimental/lhd101_c5_mobile.yaml"
EXAMPLE_ID=lhd101_c5_mobile_interface_seed
RUN_DIR="$RUN_BASE/native_c5_mobility/$SLURM_JOB_ID"
ADAPTER_DIR="$RUN_DIR/adapter"

POSE_CANDIDATE_MANIFEST="${RFD3_POSE_CANDIDATE_MANIFEST:?Set RFD3_POSE_CANDIDATE_MANIFEST to one low-tilt candidate compiled from the experimental mobility config}"
SEED="${RFD3_SEED:-42}"
NUM_TIMESTEPS="${RFD3_NUM_TIMESTEPS:-50}"
APPLY_UPDATES="${RFD3_MOBILITY_APPLY_UPDATES:-false}"
UPDATE_INTERVAL="${RFD3_MOBILITY_UPDATE_INTERVAL:-5}"
TARGET_MAX_TILT="${RFD3_MOBILITY_TARGET_MAX_TILT_DEGREES:-20.0}"
LINKER_LENGTH="${RFD3_LINKER_LENGTH:-}"
MAX_TOTAL_ROTATION=5.0

if [[ ! "$NUM_TIMESTEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( NUM_TIMESTEPS < 2 || NUM_TIMESTEPS > 200 )); then
    echo "ERROR: RFD3_NUM_TIMESTEPS must be an integer from 2 to 200"
    exit 2
fi
if [[ "$APPLY_UPDATES" != "true" && "$APPLY_UPDATES" != "false" ]]; then
    echo "ERROR: RFD3_MOBILITY_APPLY_UPDATES must be true or false"
    exit 2
fi
if [[ ! "$UPDATE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RFD3_MOBILITY_UPDATE_INTERVAL must be a positive integer"
    exit 2
fi
if [[ ! -f "$POSE_CANDIDATE_MANIFEST" ]]; then
    echo "ERROR: pose candidate manifest does not exist: $POSE_CANDIDATE_MANIFEST"
    exit 2
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

python - \
    "$POSE_CANDIDATE_MANIFEST" \
    "$APPLY_UPDATES" \
    "$TARGET_MAX_TILT" \
    "$MAX_TOTAL_ROTATION" \
    "$CONFIG" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
apply_updates = sys.argv[2] == "true"
target_max_tilt = float(sys.argv[3])
maximum_rotation = float(sys.argv[4])
config_path = Path(sys.argv[5])

manifest = json.loads(manifest_path.read_text())
actual_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
expected_config_sha = manifest.get("config", {}).get("sha256")
if expected_config_sha != actual_config_sha:
    raise SystemExit(
        "ERROR: pose manifest was not compiled from the experimental "
        "mobility config: "
        f"{expected_config_sha!r} != {actual_config_sha!r}"
    )
sample = manifest.get("initialization_samples", {}).get("primary_seed", {})
tilt = sample.get("principal_axis_tilt_deg")
if tilt is None:
    if apply_updates:
        raise SystemExit(
            "ERROR: applied mobility requires principal_axis_tilt_deg in "
            f"{manifest_path}"
        )
    print("Pose preflight: principal-axis tilt unavailable (proposal-only)")
else:
    tilt = float(tilt)
    print(f"Pose preflight: principal-axis tilt={tilt:.3f} deg")
    reachable_tilt = target_max_tilt + maximum_rotation
    if apply_updates and tilt > reachable_tilt:
        raise SystemExit(
            "ERROR: this pose cannot enter the target tilt interval within "
            f"the {maximum_rotation:.1f}-degree cumulative mobility bound: "
            f"initial={tilt:.3f}, reachable threshold={reachable_tilt:.3f}. "
            "Choose a lower-tilt manifest or run proposal-only."
        )
PY

echo "=== C5 scaffold-aware interface-seed mobility pilot ==="
echo "RFD3_POSE_CANDIDATE_MANIFEST=$POSE_CANDIDATE_MANIFEST"
echo "RFD3_SEED=$SEED"
echo "RFD3_NUM_TIMESTEPS=$NUM_TIMESTEPS"
echo "RFD3_MOBILITY_APPLY_UPDATES=$APPLY_UPDATES"
echo "RFD3_MOBILITY_UPDATE_INTERVAL=$UPDATE_INTERVAL"
echo "RFD3_MOBILITY_TARGET_MAX_TILT_DEGREES=$TARGET_MAX_TILT"
echo "RFD3_LINKER_LENGTH=${LINKER_LENGTH:-<config-midpoint>}"
echo "NOTE: false is proposal-only; true applies bounded 1 A / $MAX_TOTAL_ROTATION deg motion."
date
hostname
nvidia-smi
python --version
python -c "import rfd3; print(rfd3.__file__)"

echo "=== Repository validation before mobility inference ==="
python -m unittest discover \
    -s tests/rfd3_mosaic/unit \
    -p 'test_scaffold_guidance.py' \
    -v
python -m unittest discover \
    -s tests/rfd3_mosaic/unit \
    -p 'test_motif_mobility.py' \
    -v
python -m unittest discover \
    -s tests/rfd3_mosaic/unit \
    -p 'test_symmetry_motif_finalization.py' \
    -v
python -m unittest discover \
    -s tests/rfd3_mosaic/unit \
    -p 'test_*.py' \
    -v

python -m rfd3_mosaic.rfd3_adapter \
    --config "$CONFIG" \
    --output-dir "$ADAPTER_DIR" \
    --base-directory "$PROJECT_DIR" \
    --example-id "$EXAMPLE_ID" \
    "${ADAPTER_LINKER_ARGS[@]}" \
    --pose-candidate-manifest "$POSE_CANDIDATE_MANIFEST"

python -m rfd3_mosaic.rfd3_prevalidate \
    --input "$ADAPTER_DIR/rfd3_input.json" \
    --report "$RUN_DIR/rfd3_prevalidation.json"

echo "RFD3_MOSAIC_MOBILITY_PROPOSAL_SOURCE=scaffold_boundary"
echo "RFD3_MOSAIC_DYNAMIC_CONDITIONING=enabled"
echo "RFD3_MOSAIC_EXACT_SYMMETRY=enabled"
sha256sum \
    "$CONFIG" \
    "$ADAPTER_DIR/rfd3_input.json" \
    "$ADAPTER_DIR/presymmetrized_input.cif"

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
    ++inference_sampler.preserve_fixed_motif_during_symmetry=True \
    ++inference_sampler.require_motif_constraint_groups=True \
    ++inference_sampler.symmetry_state_mode=orbit_average \
    ++inference_sampler.symmetry_noise_mode=coupled \
    ++inference_sampler.enable_orbit_rigid_motif_mobility=True \
    ++inference_sampler.motif_mobility_proposal_source=scaffold_boundary \
    ++inference_sampler.motif_mobility_apply_updates="$APPLY_UPDATES" \
    ++inference_sampler.motif_mobility_update_interval="$UPDATE_INTERVAL" \
    ++inference_sampler.motif_mobility_target_max_tilt_degrees="$TARGET_MAX_TILT" \
    ++inference_sampler.motif_mobility_start_fraction=0.35 \
    ++inference_sampler.motif_mobility_end_fraction=0.75 \
    ++inference_sampler.motif_mobility_response=0.25 \
    ++inference_sampler.motif_mobility_per_step_translation=0.10 \
    ++inference_sampler.motif_mobility_per_step_rotation_degrees=0.5 \
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

python - "$RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
diagnostics = data.get("motif_mobility_diagnostics")
if diagnostics is None:
    raise SystemExit("ERROR: result metadata has no motif mobility diagnostics")
print(json.dumps(diagnostics, indent=2))
PY

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
    --expected-symmetry-multiplicity 5 \
    --report-only

python -m rfd3_mosaic.rfd3_audit_gate \
    --report "$RUN_DIR/seed_integrity_audit.json" \
    --report "$RUN_DIR/scaffold_validity_audit.json"

echo "=== C5 scaffold-aware mobility pilot completed ==="
date
find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
