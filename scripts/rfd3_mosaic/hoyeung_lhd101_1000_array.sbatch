#!/usr/bin/env bash
#SBATCH --job-name=hoyeung-lhd101
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-dgx-1-v100x8,lrz-v100x2,lrz-dgx-1-p100x8,lrz-hpe-p100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

HOYEUNG_REPO=${HOYEUNG_REPO:?Set HOYEUNG_REPO to the original RFdiffusion_interfaceseed checkout}
RFDIFFUSION_PYTHON=${RFDIFFUSION_PYTHON:?Set RFDIFFUSION_PYTHON to the original RFdiffusion environment python}
RFDIFFUSION_MODEL_DIR=${RFDIFFUSION_MODEL_DIR:-$HOYEUNG_REPO/models}
RFD1_OUTPUT_ROOT=${RFD1_OUTPUT_ROOT:?Set RFD1_OUTPUT_ROOT to one new campaign directory}
RFD1_DESIGNS_PER_TASK=${RFD1_DESIGNS_PER_TASK:-10}
RFD1_EXPECTED_TOTAL=${RFD1_EXPECTED_TOTAL:-1000}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]] \
    || [[ ! "$RFD1_DESIGNS_PER_TASK" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "$RFD1_EXPECTED_TOTAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid array task or design-count setting"
    exit 2
fi
if [[ ! -x "$RFDIFFUSION_PYTHON" ]]; then
    echo "ERROR: RFdiffusion Python is not executable: $RFDIFFUSION_PYTHON"
    exit 2
fi
if [[ ! -f "$RFDIFFUSION_MODEL_DIR/Base_ckpt.pt" ]]; then
    echo "ERROR: original RFdiffusion Base_ckpt.pt is missing from $RFDIFFUSION_MODEL_DIR"
    exit 2
fi
INPUT_PDB="$HOYEUNG_REPO/examples/input_pdbs/7mwr_interface.pdb"
if [[ ! -f "$INPUT_PDB" ]]; then
    echo "ERROR: Ho-Yeung input is missing: $INPUT_PDB"
    exit 2
fi

START=$((TASK_ID * RFD1_DESIGNS_PER_TASK))
if (( START >= RFD1_EXPECTED_TOTAL )); then
    echo "ERROR: task starts outside requested total: start=$START total=$RFD1_EXPECTED_TOTAL"
    exit 2
fi
COUNT=$RFD1_DESIGNS_PER_TASK
if (( START + COUNT > RFD1_EXPECTED_TOTAL )); then
    COUNT=$((RFD1_EXPECTED_TOTAL - START))
fi

CHUNK_DIR="$RFD1_OUTPUT_ROOT/chunks/$(printf '%04d' "$TASK_ID")"
DESIGN_PREFIX="$RFD1_OUTPUT_ROOT/designs/lhd101_hoyeung"
mkdir -p "$CHUNK_DIR" "$RFD1_OUTPUT_ROOT/designs"
exec >"$CHUNK_DIR/slurm-${SLURM_JOB_ID}_${TASK_ID}.out" \
    2>"$CHUNK_DIR/slurm-${SLURM_JOB_ID}_${TASK_ID}.err"

cd "$HOYEUNG_REPO"
export PYTHONPATH="$HOYEUNG_REPO:${PYTHONPATH:-}"

echo "=== Original Ho-Yeung RFdiffusion Interface-Seed ==="
echo "repo=$HOYEUNG_REPO"
echo "revision=$(git rev-parse HEAD 2>/dev/null || true)"
echo "python=$RFDIFFUSION_PYTHON"
echo "model_dir=$RFDIFFUSION_MODEL_DIR"
echo "input=$INPUT_PDB"
echo "task_id=$TASK_ID start=$START count=$COUNT total=$RFD1_EXPECTED_TOTAL"
echo "output_prefix=$DESIGN_PREFIX"
date --iso-8601=seconds
hostname
nvidia-smi
"$RFDIFFUSION_PYTHON" - <<'PY'
import torch
import rfdiffusion
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("rfdiffusion", rfdiffusion.__file__)
PY
sha256sum \
    "$INPUT_PDB" \
    "$RFDIFFUSION_MODEL_DIR/Base_ckpt.pt" \
    "$HOYEUNG_REPO/scripts/run_inference.py" \
    "$HOYEUNG_REPO/config/inference/interface_seed.yaml"

# This is the repository's published command. Only output location and the
# disjoint scheduler index/count are changed to materialize 1000 designs.
"$RFDIFFUSION_PYTHON" "$HOYEUNG_REPO/scripts/run_inference.py" \
    --config-name=interface_seed \
    diffuser.T=50 \
    inference.symmetry=C3 \
    inference.asy_motif=True \
    inference.num_designs="$COUNT" \
    inference.design_startnum="$START" \
    inference.output_prefix="$DESIGN_PREFIX" \
    inference.input_pdb="$INPUT_PDB" \
    inference.motif_drag=True \
    inference.asy_motif_weight=1 \
    inference.asy_motif_rot_range='[180,180,180]' \
    inference.asy_motif_dist=25 \
    inference.asy_motif_dist_range=5 \
    inference.deterministic=False \
    inference.model_directory_path="$RFDIFFUSION_MODEL_DIR" \
    "contigmap.contigs=[Y211-241/70-100/X165-194]" \
    'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' \
    potentials.olig_intra_all=True \
    potentials.olig_inter_all=False \
    potentials.guide_scale=2.0 \
    potentials.guide_decay=quadratic \
    hydra.run.dir="$CHUNK_DIR/hydra"

observed=0
for ((index=START; index<START+COUNT; index++)); do
    if [[ -s "${DESIGN_PREFIX}_${index}.pdb" \
        && -s "${DESIGN_PREFIX}_${index}.trb" ]]; then
        observed=$((observed + 1))
    fi
done
if (( observed != COUNT )); then
    echo "ERROR: expected $COUNT complete PDB/TRB pairs, observed $observed"
    exit 1
fi
printf '%s\n' "$START $COUNT $observed" >"$CHUNK_DIR/complete.txt"
echo "=== chunk complete: start=$START count=$COUNT ==="
date --iso-8601=seconds
