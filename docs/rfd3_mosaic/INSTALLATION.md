# Installing RFD3-Mosaic

RFD3-Mosaic supports two execution layouts without changing the scientific
compiler or sampler:

- a source checkout with immutable source snapshots for Slurm;
- an installed wheel with packaged profiles and compatibility metadata.

The supported product scope for this release is the already validated Cn/Dn
constraint, supplied-interface, generated-interface packing and bounded
mobility path. O/I and advanced cage topology solving remain experimental and
are not required for installation readiness.

## Install from GitHub

Python 3.12 is required. Install PyTorch for the target CUDA runtime first,
then install the Mosaic branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "rfd3-mosaic[rfd3] @ git+https://github.com/zhaohaixi2005-a11y/rfd3-mosaic.git@refactor/product-core-v1"
```

For development, clone the repository and install it editable:

```bash
git clone --branch refactor/product-core-v1 \
  git@github.com:zhaohaixi2005-a11y/rfd3-mosaic.git
cd rfd3-mosaic
python -m pip install -e ".[rfd3,dev]"
```

## Checkpoint

Place `rfd3_latest.ckpt` in `~/.foundry/checkpoints/`, or copy the bundled
`local` profile and change its `checkpoint` and `foundry_checkpoint_dirs`.
Cluster users may continue to select the bundled `v100`, `p100`, `a100_80g`
or `h100` profiles, or pass an absolute profile path.

## Verify the installation

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

`doctor` checks Python, PyTorch/CUDA, RFD3 imports, packaged compatibility
metadata, the selected executor and checkpoint. It performs no inference and
does not modify files.

## Run locally

Use a public design YAML and override its output root if needed:

```bash
rfd3-mosaic validate design.yaml
rfd3-mosaic plan design.yaml --profile local
rfd3-mosaic run design.yaml \
  --profile local \
  --run-root "$PWD/runs"
```

The local executor is synchronous: the command returns after inference and
all required audits finish. It uses exactly the same frozen configuration,
worker, compiler, sampler and result audits as Slurm.

## Run on Slurm

```bash
rfd3-mosaic run design.yaml --profile v100
```

Source checkouts continue to archive and hash the exact Mosaic/RFD3 source
used by every queued job. Existing LRZ profiles and commands are unchanged.

## Build and test a release

```bash
make local-test
make mosaic-release-smoke
```

The release smoke builds a wheel, installs it in a temporary environment,
changes out of the checkout, and verifies `capabilities` plus `doctor`. This
prevents a wheel that silently depends on repository-only configuration files.
