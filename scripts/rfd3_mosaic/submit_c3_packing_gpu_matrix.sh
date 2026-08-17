#!/usr/bin/env bash
set -euo pipefail

# Cross-GPU replay of the same two scientific contracts. Keeping each YAML's
# seed unchanged makes A100/H100/V100 differences attributable to runtime
# hardware rather than to a different diffusion trajectory.
project_dir=${RFD3_MOSAIC_PROJECT_DIR:-$(pwd)}
cd "$project_dir"

export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"

designs=(
  experiments/lrz_public_c3_locked_packing_v100_50step.yaml
  experiments/lrz_public_c3_joint_packing_mobility_v100_50step.yaml
)

if (( $# > 0 )); then
  profiles=("$@")
else
  # V100/P100 queues are the default fast evidence path.  Pass a100_80g or
  # h100 explicitly when those queues are available.
  profiles=(v100 p100)
fi

for profile in "${profiles[@]}"; do
  for design in "${designs[@]}"; do
    echo "========== validate: $design | $profile =========="
    python -m rfd3_mosaic.cli validate "$design" --profile "$profile"

    echo "========== submit: $design | $profile =========="
    python -m rfd3_mosaic.cli submit "$design" --profile "$profile"
  done
done
