#!/usr/bin/env bash
# Source this file from the repository root:
#   source scripts/rfd3_mosaic/activate_local_dev.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Use: source scripts/rfd3_mosaic/activate_local_dev.sh" >&2
  exit 2
fi

MOSAIC_LOCAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOSAIC_LOCAL_VENV="$MOSAIC_LOCAL_ROOT/.venv-local"

if [[ ! -x "$MOSAIC_LOCAL_VENV/bin/python" ]]; then
  echo "Local environment is missing: $MOSAIC_LOCAL_VENV" >&2
  echo "Run scripts/rfd3_mosaic/setup_local_cpu_dev.sh first." >&2
  return 1
fi

export VIRTUAL_ENV="$MOSAIC_LOCAL_VENV"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$MOSAIC_LOCAL_ROOT/src:$MOSAIC_LOCAL_ROOT/models/rfd3/src${PYTHONPATH:+:$PYTHONPATH}"

# Foundry parses these values as booleans. Desktop shells sometimes define
# DEBUG=release, which is not a valid environs boolean and breaks imports.
export DEBUG=false
export TYPE_CHECK=false
export NAN_CHECK=true

# The local CPU environment intentionally uses Biotite's built-in chemical
# component data and does not ship full CCD/PDB mirrors.  Empty values make
# that policy explicit to AtomWorks and avoid misleading import-time warnings.
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-}"
export PDB_MIRROR_PATH="${PDB_MIRROR_PATH:-}"

export RFD3_MOSAIC_LOCAL_CPU=1
hash -r

echo "RFD3-Mosaic local CPU development environment active"
echo "python: $(python --version 2>&1)"
echo "root:   $MOSAIC_LOCAL_ROOT"
echo "mirrors: external CCD/PDB mirrors optional for local CPU prevalidation"

unset MOSAIC_LOCAL_ROOT MOSAIC_LOCAL_VENV
