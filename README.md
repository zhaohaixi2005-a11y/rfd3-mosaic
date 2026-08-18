# RFD3-Mosaic

Constraint-compiled design of symmetric protein assemblies with
RFdiffusion3.

> [!IMPORTANT]
> **Project status: research preview.** RFD3-Mosaic is under active
> development. Its supported Cn/Dn workflows are usable and covered by
> automated compiler, runtime and audit tests, but the project is not yet a
> stable production release. Experimental capabilities are identified
> explicitly and may change before the first stable version.

This repository is the canonical development repository for RFD3-Mosaic.

RFD3-Mosaic extends the open-source Foundry/RFdiffusion3 stack with an
assembly-aware compiler and constrained sampler. It is designed for workflows
that must preserve supplied motifs or interface seeds while generating the
remaining protein scaffold under exact symmetry.

## What it provides

- exact preservation of fixed motifs and supplied interface geometry;
- Cn and Dn symmetric assembly generation;
- fixed, bounded-mobile and jointly guided rigid components;
- generated-interface packing guidance with translation and rotation;
- multiple components, interface seeds and polymer connections;
- deterministic configuration lowering and replayable RFD3 inputs;
- post-generation audits for motif recovery, symmetry, interfaces, clashes
  and chain continuity;
- site-independent execution: the scientific compiler and sampler are not
  tied to a particular server, institution or GPU model.

Tetrahedral, octahedral and icosahedral compiler paths are available for
research evaluation, but are not part of the current supported release scope.
Fully automatic arbitrary cage solving and downstream sequence/refolding
workflows also remain under development.

## Installation

Python 3.12 is required. Install a PyTorch build appropriate for the local
CPU or CUDA runtime first, then install RFD3-Mosaic:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "rfd3-mosaic[rfd3] @ git+https://github.com/zhaohaixi2005-a11y/rfd3-mosaic.git@refactor/product-core-v1"
```

RFD3 model weights are not distributed by this repository. Place an
authorized `rfd3_latest.ckpt` at `~/.foundry/checkpoints/` or provide its path
through a custom execution profile.

Verify the installation without running inference:

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

See the [installation guide](docs/rfd3_mosaic/INSTALLATION.md) for editable
development installs and execution-environment configuration.

## Quick start

Create a design directly from a structure and selectors. For a central motif
whose surrounding symmetric interface should be generated:

```bash
rfd3-mosaic init my-design.yaml \
  --task central-motif \
  --input motif.pdb \
  --motif-selector A12-20 \
  --symmetry C3
```

For a complete two-sided interface seed that must remain unchanged:

```bash
rfd3-mosaic init my-design.yaml \
  --task supplied-interface \
  --input interface-seed.pdb \
  --side-a A165-194 \
  --side-b B211-241 \
  --symmetry C3
```

Then inspect, validate and run the generated YAML:

```bash
rfd3-mosaic plan my-design.yaml
rfd3-mosaic validate my-design.yaml
rfd3-mosaic run my-design.yaml
```

`rfd3-mosaic examples` lists maintained templates, and
`rfd3-mosaic profiles` lists execution profiles. Both commands can copy a
portable starting file without requiring a source checkout.

The bundled synchronous executor can run on any compatible machine. Slurm is
also supported through a site-defined profile. These are launch mechanisms,
not restrictions on the server or GPU that may be used.

## User workflows

RFD3-Mosaic currently exposes two primary workflows:

1. **Preserve supplied interfaces.** The user supplies one or more complete
   interface seeds. Mosaic preserves each seed's internal geometry and
   generates the requested polymer connections and scaffold.
2. **Create a symmetric interface around a motif.** The user supplies a
   central motif. Mosaic keeps the motif fixed internally and guides generated
   regions toward a symmetric, clash-aware interface. Component translation
   and rotation can be locked or bounded by the selected motion policy.

Ordinary-user configurations describe the intended assembly and seed usage.
Expert configurations may additionally specify components, ports, group
relations, mobility subspaces and packing controls. Both modes compile to the
same internal assembly representation and RFD3 runtime.

## Validation model

A generated coordinate file is not considered successful merely because
inference finished. Required result audits evaluate:

- fixed-motif completeness and joint rigid recovery;
- declared symmetry and component-orbit consistency;
- supplied or generated interface relations;
- backbone continuity, clashes and compactness;
- runtime packing-guidance execution when enabled.

The CLI reports a run as passed only when every required audit passes.

## Documentation

- [Installation and environment setup](docs/rfd3_mosaic/INSTALLATION.md)
- [Quick start and user workflows](docs/rfd3_mosaic/QUICKSTART.md)
- [Implemented capabilities and limitations](DEVELOPMENT_STATUS.md)
- [Current research status](docs/rfd3_mosaic/PROJECT_STATUS.md)

Detailed architecture plans, experiment records and site-specific deployment
notes are retained as development evidence, but are not prerequisites for
installing or using the public software.

## Development

For an editable installation:

```bash
git clone --branch refactor/product-core-v1 \
  https://github.com/zhaohaixi2005-a11y/rfd3-mosaic.git
cd rfd3-mosaic
python -m pip install -e ".[rfd3,dev]"
make local-test
```

Build and test an installable artifact with:

```bash
make mosaic-release-smoke
```

Contributions should preserve backward compatibility for validated workflows
and include focused regression tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Upstream and attribution

RFD3-Mosaic is derived from the open-source
[Foundry](https://github.com/RosettaCommons/foundry) and RFdiffusion3 codebase.
It is an independent research extension and is not an official Rosetta
Commons or Institute for Protein Design release. Upstream model documentation
is available in [models/rfd3/README.md](models/rfd3/README.md).

If you use this software, cite the relevant upstream RFdiffusion3, Foundry,
AtomWorks and model publications in addition to any future RFD3-Mosaic
release citation.

## License

This repository retains the upstream BSD 3-Clause license. See
[LICENSE.md](LICENSE.md).
