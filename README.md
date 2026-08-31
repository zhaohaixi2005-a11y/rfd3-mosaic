<h1 align="center">RFD3-Mosaic</h1>

<p align="center">
  <strong>Constraint-compiled symmetric protein assembly design with RFdiffusion3</strong>
</p>

<p align="center">
  <a href="docs/rfd3_mosaic/PROJECT_STATUS.md"><img alt="Project status: research preview" src="https://img.shields.io/badge/status-research_preview-334155?style=flat-square"></a>
  <a href="pyproject.toml"><img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-2563EB?style=flat-square&amp;logo=python&amp;logoColor=white"></a>
  <a href="LICENSE.md"><img alt="License: BSD 3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-0F766E?style=flat-square"></a>
  <a href="models/rfd3/README.md"><img alt="Backend: RFdiffusion3" src="https://img.shields.io/badge/backend-RFdiffusion3-7C3AED?style=flat-square"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting started</a> ·
  <a href="docs/rfd3_mosaic/WORKFLOW_GUIDE.md">User guide</a> ·
  <a href="#design-workflows">Design workflows</a> ·
  <a href="#how-mosaic-works">Architecture</a> ·
  <a href="#outputs-and-interpretation">Outputs</a> ·
  <a href="#repository-layout">Repository layout</a> ·
  <a href="docs/rfd3_mosaic/README.md">Documentation</a>
</p>

RFD3-Mosaic is a constraint-compilation and execution framework for symmetric
protein backbone design with [RFdiffusion3](models/rfd3/README.md). Starting
from a user-defined assembly specification, it constructs symmetry-aware RFD3
inputs, samples independent assembly poses, executes constrained backbone
generation and records structural audits with full run provenance.

The framework supports fixed-motif scaffolding, supplied-interface
preservation, multi-component assembly graphs, controlled rigid-body motion
and reproducible batch generation through one consistent workflow.

> [!IMPORTANT]
> RFD3-Mosaic is an actively developed research preview. Cn and Dn workflows
> form the current release target; finite polyhedral groups and advanced
> multi-interface designs are available as research workflows. Validation
> evidence is summarized in the
> [project status](docs/rfd3_mosaic/PROJECT_STATUS.md).

## Getting started

### Requirements

- Python 3.12;
- a PyTorch installation compatible with the target CPU or CUDA runtime;
- an authorized RFdiffusion3 checkpoint;
- a GPU suitable for RFdiffusion3 inference when generating structures.

Portable local and Slurm execution profiles separate scientific designs from
site configuration. Resource requirements scale with assembly size and
symmetry multiplicity.

### Install

Install PyTorch for the target environment first, then install the current
development branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "rfd3-mosaic[rfd3] @ git+https://github.com/Khmelinskaia-Lab/foundry.git@hx/rfd3-mosaic-product-core"
```

RFD3 model weights are not redistributed by this repository. Place an
authorized `rfd3_latest.ckpt` in `~/.foundry/checkpoints/`, or configure its
location in an execution profile.

Verify the installation without launching inference:

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

See the [installation guide](docs/rfd3_mosaic/INSTALLATION.md) for editable
installs, checkpoints and portable Slurm profiles.

### Run a first design

Create a design with the simple two-sided supplied-interface initializer:

```bash
rfd3-mosaic init design.yaml \
  --task supplied-interface \
  --input interface-seed.pdb \
  --side-a A20-35 \
  --side-b B40-55 \
  --symmetry C3 \
  --designs 100
```

Inspect the resolved plan and validate the compiled input before using GPU
time, then run it:

```bash
rfd3-mosaic plan design.yaml
rfd3-mosaic validate design.yaml
rfd3-mosaic run design.yaml
```

Inspect the completed run by job ID or directory:

```bash
rfd3-mosaic status RUN_ID_OR_DIRECTORY
rfd3-mosaic report RUN_ID_OR_DIRECTORY
```

For supplied non-covalent partners that should also form an additional
symmetry-related interface, start from the maintained compositional example:

```bash
rfd3-mosaic examples --copy supplied-interface-oligomer --output design.yaml
```

That example keeps the declared interface partners joint-rigid, grows their
scaffold without adding an artificial peptide bond, and explicitly opts into
an additional generated Cn interface. Sequence masking, all-glycine
conditioning, symmetric ligands, RASA, hotspots and H-bond conditioning are
optional fields in the same public YAML rather than separate task-specific
programs.

The [quick-start guide](docs/rfd3_mosaic/QUICKSTART.md) covers portable
examples, component motion, execution profiles and result inspection.
The [complete user workflow guide](docs/rfd3_mosaic/WORKFLOW_GUIDE.md)
separates required and optional input, provides copy-ready examples for every
supported ordinary-user task, and documents batch generation plus native RFD3
conditioning.
The simple initializer accepts two interface sides. The general public
assembly graph is not limited to two components: it supports multiple rigid
or joint-rigid components, multiple preserved/generated interface relations,
and explicit generated polymer connections.
The [native RFD3 capability matrix](docs/rfd3_mosaic/RFD3_NATIVE_CAPABILITIES.md)
maps each supported conditioning control to its Mosaic configuration field
and execution behavior.

## Design workflows

Mosaic provides two primary backbone-generation workflows.

### Preserve a supplied interface

Use this workflow when the input already contains the interface geometry that
must remain unchanged. Mosaic treats both sides as one joint-rigid object:
their internal geometry is exact, while the complete object may translate and
rotate relative to the symmetry frame when the design permits motion.

```bash
rfd3-mosaic init design.yaml \
  --task supplied-interface \
  --input interface-seed.pdb \
  --side-a A20-35 \
  --side-b B40-55 \
  --symmetry C3
```

### Generate an interface around a motif

Use this workflow when the input supplies a fixed structural motif and the
surrounding scaffold and symmetry-related interface must be generated.

```bash
rfd3-mosaic init design.yaml \
  --task central-motif \
  --input motif.pdb \
  --motif-selector A12-20 \
  --symmetry C3 \
  --component-motion guided
```

Interface relations are explicit in the assembly specification. A supplied
interface is preserved as declared, while generated interfaces can be added
independently with their own packing guidance.

### Fixed and movable geometry

Mosaic separates three levels that are often conflated in diffusion inputs:

- **fixed atoms** retain their declared coordinates;
- **joint-rigid components** preserve internal geometry while moving as a
  single SE(3) body;
- **generated polymer** is sampled by RFD3 under the compiled constraints.

For movable-assembly workflows, `designs: N` instantiates independent,
reproducible assembly poses and diffusion seeds. Fully locked arrangements
retain the declared pose. Multi-example execution lets RFD3 reuse one model
load without collapsing those per-design inputs into repeated diffusion from
one pose.

Bounded-mobile components use coarse-to-fine SE(3) optimization during
diffusion, allowing broad early pose adaptation followed by progressively
smaller refinements. Early capture evaluates a deterministic bounded
multi-start neighbourhood; generated-interface tasks also use intra-chain
core support by default, including a smooth worst-window term for long
unsupported arms. The schedule, objective terms and geometric invariants
are defined in the
[rigid-mobility mathematical contract](docs/rfd3_mosaic/RIGID_MOBILITY_MATHEMATICAL_CONTRACT.md).

## How Mosaic works

<p align="center">
  <img
    src="docs/rfd3_mosaic/assets/rfd3_mosaic_architecture.svg"
    alt="RFD3-Mosaic architecture: design declaration, assembly compiler, independent pose ensemble, constrained RFdiffusion3 sampling and auditable outputs"
    width="100%"
  >
</p>

The compiler resolves named components, motif and interface orbits, polymer
paths, symmetry transforms and motion policies before inference. Every
executable design is lowered to an explicit RFD3 specification with frozen
configuration, seeds and software provenance. Sampling then applies the
compiled fixed-geometry, symmetry, mobility and optional guidance contracts.

The execution stack separates five responsibilities:

| Layer | Responsibility |
| --- | --- |
| User specification | Declares the intended assembly and scientific constraints |
| Mosaic compiler | Resolves topology, symmetry, geometry, polymer paths and per-design poses |
| RFdiffusion3 | Generates conditioned protein backbones |
| Mosaic runtime | Enforces compiled constraints and records guidance behavior |
| Mosaic audits | Measures declared contracts and advisory structural properties |

## Outputs and interpretation

Each run preserves the resolved configuration, compiled input, generated
coordinates, runtime provenance and audit reports. Completed structures are
mirrored incrementally into `generated_structures_cif/` as plain CIF files;
the completed run also provides a structure-only ZIP for batch inspection.

Mosaic reports three different outcomes:

1. **Generated** — inference produced a finite coordinate structure.
2. **Contract checks** — declared invariants such as fixed geometry, symmetry,
   continuity and topology were measured.
3. **Advisory measurements** — task-dependent descriptors support ranking and
   scientific review.

All generated coordinates are retained. Contract results and advisory
measurements are reported separately, providing a traceable basis for
downstream ranking, sequence design, independent refolding and experimental
selection.

Useful post-run commands are:

```bash
rfd3-mosaic status RUN_ID_OR_DIRECTORY
rfd3-mosaic report RUN_ID_OR_DIRECTORY
rfd3-mosaic audit RUN_ID_OR_DIRECTORY
```

## Current capabilities

| Area | Current maturity |
| --- | --- |
| Cn/Dn symmetry, fixed motifs and supplied interfaces | Supported release target |
| Locked, bounded-mobile and joint-rigid components | Supported release target |
| Multiple components, motif orbits and polymer connections | Supported release target |
| Independent per-design assembly poses | Supported release target |
| Generated-interface guidance | Implemented; scientific calibration continues |
| T/O/I finite-group execution | Research capability with path-specific GPU evidence |
| Stabilizers, cosets, quotient orbits and advanced multi-interface cases | Controlled research capability |

The compiler validates symmetry, topology, connectivity and component-motion
semantics before execution, and reports actionable diagnostics when a design
specification is incomplete. Sequence design, independent refolding and
experimental validation are planned downstream workflow integrations.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/rfd3_mosaic/` | Compiler, runtime, CLI, reporting and audit implementation |
| `examples/rfd3_mosaic/` | Maintained, packaged user examples and small structure fixtures |
| `configs/rfd3_mosaic/` | Compatibility, symmetry and portable execution profiles |
| `docs/rfd3_mosaic/` | User guides, capability contracts and evidence summaries |
| `tests/rfd3_mosaic/` | CPU regression and public-contract tests |
| `models/rfd3/` | RFdiffusion3 backend and upstream model configuration |

Installed profiles are portable; the source checkout additionally contains
site profiles and frozen validation configurations for reproducible testing.

## Documentation

- [Documentation index](docs/rfd3_mosaic/README.md)
- [Installation and execution](docs/rfd3_mosaic/INSTALLATION.md)
- [Quick start](docs/rfd3_mosaic/QUICKSTART.md)
- [Complete user workflow guide](docs/rfd3_mosaic/WORKFLOW_GUIDE.md)
- [CLI reference](docs/rfd3_mosaic/USER_CLI.md)
- [Project status and GPU evidence](docs/rfd3_mosaic/PROJECT_STATUS.md)
- [Development status](DEVELOPMENT_STATUS.md)
- [Packing-guidance semantics](docs/rfd3_mosaic/PACKING_GUIDANCE.md)
- [Metric provenance](docs/rfd3_mosaic/STRUCTURE_METRIC_PROVENANCE.md)
- [Backbone-evaluation evidence](docs/rfd3_mosaic/BACKBONE_EVALUATION_EVIDENCE.md)
- [Security and sensitive-data policy](SECURITY.md)

## Development

```bash
git clone --branch hx/rfd3-mosaic-product-core \
  https://github.com/Khmelinskaia-Lab/foundry.git rfd3-mosaic
cd rfd3-mosaic
python -m pip install -e ".[rfd3,dev]"
make local-test
make mosaic-release-smoke
```

Contributions should preserve validated workflows, add focused regression
tests and distinguish CPU evidence from GPU evidence. See
[CONTRIBUTING.md](CONTRIBUTING.md).

Before sharing logs or structures, review the [security policy](SECURITY.md).

## Upstream software and citation

RFD3-Mosaic is a research extension of the open-source
[Foundry](https://github.com/RosettaCommons/foundry) and RFdiffusion3 stack. It
is not an official Rosetta Commons or Institute for Protein Design release.

Until a dedicated RFD3-Mosaic publication is available, cite the relevant
RFdiffusion3, Foundry, AtomWorks and model publications used in the workflow.
The upstream model documentation is retained in
[models/rfd3/README.md](models/rfd3/README.md).

## License

RFD3-Mosaic retains the upstream BSD 3-Clause license. See
[LICENSE.md](LICENSE.md).
