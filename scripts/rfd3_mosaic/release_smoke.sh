#!/bin/bash
set -euo pipefail

REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SMOKE_ROOT=$(mktemp -d)
trap 'rm -rf "$SMOKE_ROOT"' EXIT
PYTHON_BIN=${PYTHON:-"$REPOSITORY/.venv-local/bin/python"}
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=$(command -v python3 || command -v python || true)
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "No Python interpreter is available for the wheel smoke" >&2
    exit 1
fi

WHEEL=$(find "$REPOSITORY/dist" -maxdepth 1 -name 'rfd3_mosaic-*.whl' \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d ' ' -f 2-)
if [[ -z "$WHEEL" ]]; then
    echo "No RFD3-Mosaic wheel found below $REPOSITORY/dist" >&2
    exit 1
fi
WHEEL_BASENAME=$(basename "$WHEEL")
DIST_VERSION=${WHEEL_BASENAME%-py3-none-any.whl}
SDIST="$REPOSITORY/dist/$DIST_VERSION.tar.gz"
if [[ ! -f "$SDIST" ]]; then
    echo "No RFD3-Mosaic source distribution found below $REPOSITORY/dist" >&2
    exit 1
fi
if tar -tzf "$SDIST" | grep -E \
    '/(configs/rfd3_mosaic/sites|docs/internal|experiments|reports|scripts)/' \
    >/dev/null; then
    echo "Source distribution contains internal deployment material" >&2
    exit 1
fi
# Use the selected interpreter's standard installer when it is available.
# GitHub-hosted runners provide ``pip`` with setup-python but do not promise
# the optional ``uv`` executable.  Local Mosaic development environments are
# intentionally allowed to be uv-created, pip-less environments, so retain a
# guarded uv fallback instead of requiring either installer everywhere.
if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --target "$SMOKE_ROOT/site-packages" \
        "$WHEEL"
elif command -v uv >/dev/null 2>&1; then
    uv pip install --no-deps --target "$SMOKE_ROOT/site-packages" "$WHEEL"
else
    echo "Wheel smoke requires pip for $PYTHON_BIN or the uv executable" >&2
    exit 1
fi

PROFILE_ROOT="$SMOKE_ROOT/site-packages/rfd3_mosaic/resources/configs/rfd3_mosaic/execution"
test -f "$PROFILE_ROOT/local.yaml"
test -f "$PROFILE_ROOT/slurm-example.yaml"
for private_profile in v100.yaml p100.yaml a100_80g.yaml h100.yaml; do
    test ! -e "$PROFILE_ROOT/$private_profile"
done
if grep -R -l -E '/dss/dssfs|pn57ki|re73rub2|/home/haixi|login\.ai\.lrz' \
    "$SMOKE_ROOT/site-packages/rfd3_mosaic" >/dev/null; then
    echo "Wheel contains a private deployment path" >&2
    exit 1
fi

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
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli \
    capabilities --format json >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli doctor \
    --profile local \
    --checkpoint "$SMOKE_ROOT/rfd3_latest.ckpt" \
    --format json >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli \
    examples --format json >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli \
    examples --copy central-motif \
    --output "$SMOKE_ROOT/example.yaml" >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli \
    examples --copy supplied-interface-oligomer \
    --output "$SMOKE_ROOT/interface-oligomer.yaml" >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli \
    profiles --format json >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli \
    profiles --copy-slurm "$SMOKE_ROOT/slurm.yaml" >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli init \
    "$SMOKE_ROOT/design.yaml" \
    --task central-motif \
    --input "$SMOKE_ROOT/source.cif" \
    --motif-selector A1 \
    --designs 3 \
    --pose-radius-minimum 12 \
    --pose-radius-maximum 18 \
    --pose-orientation uniform_so3 \
    --pose-seed 100 >/dev/null
PYTHONPATH="$SMOKE_ROOT/site-packages" "$PYTHON_BIN" -m rfd3_mosaic.cli render \
    "$SMOKE_ROOT/experiment.yaml" \
    --output-dir "$SMOKE_ROOT/rendered" >/dev/null
test ! -e "$SMOKE_ROOT/rendered/source_snapshot.tar.gz"
grep -q 'python -m rfd3_mosaic.experiment_worker' \
    "$SMOKE_ROOT/rendered/generated_job.sbatch"
echo "RFD3-Mosaic wheel smoke: PASSED"
