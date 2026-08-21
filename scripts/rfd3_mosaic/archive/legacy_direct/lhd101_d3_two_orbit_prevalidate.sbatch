#!/bin/bash -l
#SBATCH --job-name=d3-2orb-pre
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source "$HOME/software_paths.sh"
source "${SHARED_MAMBAFORGE:?}/etc/profile.d/conda.sh"
conda activate "${RC_FOUNDRY_ENV:?}"

PROJECT_DIR=${PROJECT_DIR:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic}
RUN_ROOT=${RFD3_D3_VALIDATION_ROOT:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/d3_two_orbit_engineering/declared_frame_prevalidation}
RUN_DIR="$RUN_ROOT/${SLURM_JOB_ID:?}"

cd "$PROJECT_DIR"
exec bash scripts/rfd3_mosaic/archive/legacy_direct/validate_lhd101_d3_two_orbit.sh "$RUN_DIR"
