# Installing RFD3-Mosaic

> **Release status:** research preview. The supported Cn/Dn paths are usable,
> but APIs and experimental assembly capabilities may still change before the
> first stable release.

RFD3-Mosaic supports two execution layouts without changing the scientific
compiler or sampler:

- a source checkout with immutable source snapshots for Slurm;
- an installed wheel with packaged profiles and compatibility metadata.

RFD3-Mosaic is not tied to any institution, hostname, partition or GPU model.
It can run wherever its Python, PyTorch, RFD3 and checkpoint requirements are
available. The bundled `local` and Slurm executors only describe how a process
is launched; they do not restrict the underlying machine. Additional
scheduler adapters can be added without changing the scientific design path.

The supported product scope is the validated Cn/Dn constraint,
supplied-interface, generated-interface packing and bounded-mobility path.
Polyhedral compiler paths and advanced automatic cage solving remain research
features and are not installation prerequisites.

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

For development, clone the repository and install it editable. HTTPS requires
no GitHub SSH-key configuration:

```bash
git clone --branch refactor/product-core-v1 \
  https://github.com/zhaohaixi2005-a11y/rfd3-mosaic.git
cd rfd3-mosaic
python -m pip install -e ".[rfd3,dev]"
```

## Checkpoint

RFD3-Mosaic does not redistribute model weights. Place an authorized
`rfd3_latest.ckpt` in `~/.foundry/checkpoints/`, or copy the bundled `local`
profile and change its `checkpoint` and `foundry_checkpoint_dirs`.

## Verify the installation

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

`doctor` checks Python, PyTorch/CUDA, RFD3 imports, packaged compatibility
metadata, the selected executor and checkpoint. It performs no inference and
does not modify files.

## Run directly

Use a public design YAML and override its output root if needed:

```bash
rfd3-mosaic validate design.yaml
rfd3-mosaic plan design.yaml --profile local
rfd3-mosaic run design.yaml \
  --profile local \
  --run-root "$PWD/runs"
```

The direct executor is synchronous: the command returns after inference and
all required audits finish. Despite the historical profile name `local`, it
may run on a laptop, workstation, shared GPU server or allocated compute node.
It uses exactly the same frozen configuration, worker, compiler, sampler and
result audits as a scheduler-backed run.

## Run on Slurm

If the target environment uses Slurm, its partitions, accounts, environment
activation and checkpoint locations are site-specific. Copy the generic
template and edit it for that cluster:

```bash
cp configs/rfd3_mosaic/execution/slurm-example.yaml my-cluster.yaml
# Edit partition, resources, setup_commands and checkpoint first.
rfd3-mosaic doctor --profile "$PWD/my-cluster.yaml"
rfd3-mosaic run design.yaml --profile "$PWD/my-cluster.yaml"
```

Source checkouts continue to archive and hash the exact Mosaic/RFD3 source
used by every queued job. Site-specific profiles are deployment configuration,
not a requirement of the public software.

## Build and test a release

```bash
make local-test
make mosaic-release-smoke
```

The release smoke builds a wheel, installs it in a temporary environment,
changes out of the checkout, and verifies `capabilities` plus `doctor`. This
prevents a wheel that silently depends on repository-only configuration files.
