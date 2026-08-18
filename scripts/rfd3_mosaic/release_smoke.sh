#!/bin/bash
set -euo pipefail

REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SMOKE_ROOT=$(mktemp -d)
trap 'rm -rf "$SMOKE_ROOT"' EXIT

WHEEL=$(find "$REPOSITORY/dist" -maxdepth 1 -name 'rfd3_mosaic-*.whl' \
    -print | sort | tail -n 1)
if [[ -z "$WHEEL" ]]; then
    echo "No RFD3-Mosaic wheel found below $REPOSITORY/dist" >&2
    exit 1
fi
uv pip install --no-deps --target "$SMOKE_ROOT/site-packages" "$WHEEL"

printf 'release-smoke-checkpoint\n' > "$SMOKE_ROOT/rfd3_latest.ckpt"
printf 'data_release_smoke\n' > "$SMOKE_ROOT/source.cif"
cat > "$SMOKE_ROOT/template.json" <<'EOF'
{"release-smoke": {"input": "source.cif", "symmetry": {"id": "C3"}}}
EOF
cat > "$SMOKE_ROOT/local-profile.yaml" <<EOF
schema_version: 1
name: local-release-smoke
executor: local
setup_commands: []
checkpoint: $SMOKE_ROOT/rfd3_latest.ckpt
foundry_checkpoint_dirs: $SMOKE_ROOT
EOF
cat > "$SMOKE_ROOT/experiment.yaml" <<EOF
schema_version: 1
name: installed-release-smoke
topology:
  kind: central_motif
  template_input: $SMOKE_ROOT/template.json
  fixed_selector: A1
resources:
  profile: $SMOKE_ROOT/local-profile.yaml
output:
  root: $SMOKE_ROOT/runs
  campaign: release-smoke
EOF
cd "$SMOKE_ROOT"
PYTHONPATH="$SMOKE_ROOT/site-packages" python -m rfd3_mosaic.cli \
    capabilities --format json >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" python -m rfd3_mosaic.cli doctor \
    --profile local \
    --checkpoint "$SMOKE_ROOT/rfd3_latest.ckpt" \
    --format json >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" python -m rfd3_mosaic.cli render \
    "$SMOKE_ROOT/experiment.yaml" \
    --output-dir "$SMOKE_ROOT/rendered" >/dev/null
test ! -e "$SMOKE_ROOT/rendered/source_snapshot.tar.gz"
grep -q 'python -m rfd3_mosaic.experiment_worker' \
    "$SMOKE_ROOT/rendered/generated_job.sbatch"
echo "RFD3-Mosaic wheel smoke: PASSED"
