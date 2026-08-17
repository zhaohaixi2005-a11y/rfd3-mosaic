#!/usr/bin/env bash
set -euo pipefail

project_dir=${RFD3_MOSAIC_PROJECT_DIR:-$(pwd)}
cd "$project_dir"
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"

designs=(
  experiments/lrz_public_c3_locked_packing_patch_capture_v100_50step.yaml
  experiments/lrz_public_c3_joint_packing_patch_capture_v100_50step.yaml
)
if (( $# > 0 )); then
  profiles=("$@")
else
  profiles=(v100)
fi

for profile in "${profiles[@]}"; do
  for design in "${designs[@]}"; do
    echo "========== validate: $design | $profile =========="
    python -m rfd3_mosaic.cli validate "$design" --profile "$profile"
    echo "========== submit: $design | $profile =========="
    python -m rfd3_mosaic.cli submit "$design" --profile "$profile"
  done
done
