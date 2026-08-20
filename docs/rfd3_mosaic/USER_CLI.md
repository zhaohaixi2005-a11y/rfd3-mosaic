# RFD3-Mosaic command-line reference

RFD3-Mosaic provides one public command, `rfd3-mosaic`. The command-line
interface is independent of the machine or institution where it runs.

> **Status:** research preview. Commands documented here are the maintained
> public interface, but compatibility is not guaranteed until the first stable
> release.

## Installation check

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

- `doctor` checks imports, packaged resources, execution configuration and the
  RFD3 checkpoint without starting inference.
- `capabilities` reports implemented features and their validation maturity.

Use `--format json` with either command for machine-readable output.

## Core lifecycle

Every design follows the same lifecycle:

```text
user intent -> plan -> validate -> resolve if needed -> run -> audits -> report
```

### `init`

```bash
rfd3-mosaic init design.yaml \
  --task central-motif \
  --input motif.pdb \
  --motif-selector A12-20 \
  --symmetry C3 \
  --designs 100
```

Creates a schema-valid ordinary-user YAML and prints the exact `plan`,
`validate` and `run` commands to use next. Selectors remain explicit because
they define scientific intent; radius, neighbour transforms and raw packing
weights are not required.

The two tasks are:

- `central-motif`: preserve a motif and create surrounding interface packing;
- `supplied-interface`: preserve both sides of a complete supplied interface
  as one joint rigid geometric seed.

For `central-motif`, `--component-motion locked|guided|free` selects whether
the supplied arrangement stays fixed, moves in the calibrated constrained
subspace, or uses bounded SE(3). High-level `--packing`, `--interface-area`,
`--cavity` and `--diversity` preferences are also available. Hard symmetry,
motif, continuity and clash contracts are never disabled by these options.

`--designs N` controls how many independent RFD3 diffusion samples one run
produces **from one compiled initial pose**. It does not compile `N` different
radius/orientation poses. The equivalent YAML field is:

```yaml
sampling:
  timesteps: 200
  designs: 100
  seed: 42
```

Mosaic keeps `diffusion_batch_size=1` for predictable GPU memory use and asks
RFD3 for `N` sequential stochastic batches. Every result has its own structure,
metadata, semantic audits and scaffold audit below `audits/<design-id>/`.
The run report records produced, accepted and rejected counts; one rejected
design does not discard the remaining outputs of a multi-design screening run.
RFD3 seeds the engine once, so an output is reproducibly identified by the
frozen source/input, the common base `seed` and its batch index; `seed` is not
the pre-RFD3 pose seed. Use `resolve`/`search`, or a campaign with an explicit
pose-seed schedule, when each design should start from a different rigid pose.
For very large campaigns, choose a scheduler walltime that can accommodate the
requested count or split the total across several seeds/jobs.

### `examples` and `profiles`

```bash
rfd3-mosaic examples
rfd3-mosaic examples --copy supplied-interface --output design.yaml
rfd3-mosaic profiles
rfd3-mosaic profiles --copy-slurm my-cluster.yaml
```

These discovery commands work from both source checkouts and installed
packages. Copied examples resolve their bundled input and run paths; copied
Slurm profiles are generic site templates. Both support `--format json`, and
neither overwrites an existing file unless `--force` is supplied.

### `plan`

```bash
rfd3-mosaic plan design.yaml --profile local
```

Prints the resolved task, components, constraints, symmetry, interfaces,
generation regions and execution plan. It does not run inference.

### `validate`

```bash
rfd3-mosaic validate design.yaml
```

Performs schema validation, assembly lowering, geometry checks and RFD3
runtime-feature prevalidation. Invalid or unsupported designs fail before GPU
execution.

### `resolve`

```bash
rfd3-mosaic resolve intent.yaml \
  --output-dir resolved-designs \
  --top 4
```

Used only for ordinary intents that leave assembly variables unresolved. It
enumerates permitted candidate states, performs configured pose optimization,
ranks candidates and writes strictly replayed executable YAML files. It does
not invent supplied interface identities.

### `run`

```bash
rfd3-mosaic run design.yaml \
  --profile local \
  --run-root "$PWD/runs"
```

Renders a frozen execution envelope and launches it with the selected
executor. The `local` profile performs direct synchronous execution on any
compatible machine. A custom Slurm profile submits the same envelope through
the scheduler.

### `render`

```bash
rfd3-mosaic render design.yaml \
  --profile local \
  --output rendered-run
```

Writes the immutable runtime configuration without launching it. This is
useful for review, provenance inspection and deployment integration.

### `status`, `report` and `audit`

```bash
rfd3-mosaic status RUN_ID_OR_DIRECTORY
rfd3-mosaic report RUN_ID_OR_DIRECTORY
rfd3-mosaic audit RUN_ID_OR_DIRECTORY
```

- `status` summarizes execution state, outputs and required audits.
- `report` writes JSON and HTML reports from recorded run artifacts.
- `audit` evaluates a compatible existing run against the declared contracts.

Use `rfd3-mosaic runs --root /path/to/runs` to list an indexed run root.

For a large run root, create a non-destructive catalog instead of moving or
deleting run directories:

```bash
rfd3-mosaic runs \
  --root /path/to/runs \
  --rebuild \
  --catalog \
  --retain IMPORTANT_JOB_ID
```

Open the current catalog through the stable `/path/to/runs/RUN_CATALOG` link
(`/path/to/runs/_catalog/CURRENT` is the internal equivalent). Catalog snapshots
are themselves grouped under `snapshots/YYYYMMDD/`. The primary run view is
`by-date/YYYY-MM-DD/`: every UTC day has one parent directory, a
human-readable `RUNS.md`, and links whose names record the job ID, experiment,
state and source revision.
Additional views group the same immutable runs by source version and state,
provide direct structure links, and expose explicitly retained jobs. Catalog
entries are symbolic links; original outputs and per-run `software/` source
snapshots remain in place and are not copied or deleted.
Retained job IDs are carried forward automatically when the catalog is
refreshed.

New executions use a date-first physical layout directly:

```text
RUN_ROOT/
└── YYYY-MM-DD/
    ├── _requests/
    ├── _submissions/
    └── EXPERIMENT/
        └── JOB_ID/
            ├── input/
            ├── audits/
            ├── software/
            └── experiment_summary.json
```

Historical indexed runs can be physically migrated into the same layout. Run
the read-only plan first:

```bash
rfd3-mosaic runs --root /path/to/runs --rebuild \
  --reorganize-by-date plan --limit 20
```

After reviewing every `MOVE` and `SKIP`, apply exactly that policy with:

```bash
rfd3-mosaic runs --root /path/to/runs \
  --reorganize-by-date apply --limit 20
```

Only `completed` and `failed` runs are moved. Running and submitted jobs are
never touched. The operation refuses collisions, updates the persistent job
index and submission receipt, records every completed move under
`RUN_ROOT/_migrations/`, and removes only parent directories that become
completely empty.

## Design workflows

### Preserve supplied geometry

Create it with:

```bash
rfd3-mosaic init design.yaml \
  --task supplied-interface \
  --input interface-seed.pdb \
  --side-a A165-194 \
  --side-b B211-241 \
  --symmetry C3
```

Use this mode when the input already contains one or more complete interfaces.
Every supplied interface is preserved as a geometric entity. Generated
regions connect user-declared endpoints without changing the internal seed
geometry.

The complete seed can still move as one rigid body relative to the symmetry
axis. For example, add `--component-motion free` to `init`, or set:

```yaml
preferences:
  component_motion: free
```

This compiles both sides into one coupling group with bounded full SE(3)
motion; it never allows the two supplied interface sides to move independently.

`task: preserve_supplied_geometry` never invents a second generated interface.
The supplied interface remains the oligomeric contact; generated residues are
shaped as a monomer scaffold. `sampling.scaffold_packing: symmetric_generated`
is rejected for this task because it has the different meaning “create a new
generated--generated symmetry-neighbour interface”.

The same guidance path exposes an RFdiffusion-style intra/inter balance; this
is a pair of weights, not another workflow mode:

```yaml
guidance:
  intra_chain_weight: 1.0
  inter_chain_weight: 0.10
```

`intra_chain_weight` rewards long-range contacts inside each generated
monomer, a bounded length-normalized radius of gyration, and tertiary-contact
support across generated residues. `inter_chain_weight` follows contact-map
semantics: it scales only declared generated--generated interface edges. In a
supplied-interface design with no generated interface edge it is intentionally
inactive; a value below one is not silently converted into repulsion.

Experts who deliberately need to discourage a second broad generated contact
surface can independently set `guidance.inter_chain_excess_penalty`. It is
zero by default and is not exposed as an ordinary-user shortcut. Exact
supplied-seed geometry, symmetry, chain continuity and clash rejection remain
hard contracts at every weight.

Scientific compactness thresholds are also independent of runtime/safety
validation. Existing designs remain report-only. A calibrated campaign may
opt in explicitly:

```yaml
sampling:
  scaffold_core_quality:
    required: true
    maximum_mean_normalized_rg: 2.60
    minimum_mean_tertiary_support_fraction: 0.50
    maximum_long_range_contact_deficit: 0.25
```

The mobility audit reports the observed translation and rotation as fractions
of their declared full-SE(3) bounds. Nonzero movement is evidence, not a hard
acceptance condition: an already-good initial pose may correctly remain still.
It also reports the declared and effective proposal interval, the scheduled
active proposal count, the upper-bound translation/rotation search budget and
the bound-normalized soft-prior scales. `fixed` components remain absolutely
fixed. These scheduling fields apply only to `orbit_rigid` components: their
supplied internal interface geometry remains exact while the complete seed can
move as one symmetry-coupled rigid pose.

Existing YAMLs that omit these fields retain the original values
`intra_chain_weight=0` and `inter_chain_weight=1`.

### Create a symmetric interface

Create it with:

```bash
rfd3-mosaic init design.yaml \
  --task central-motif \
  --input motif.pdb \
  --motif-selector A12-20 \
  --symmetry C3
```

Use this mode when the input supplies a motif and Mosaic should generate new
packing around it. The motif remains internally rigid. Its assembly pose is
either locked or allowed to move inside a declared bounded subspace.

### Multiple supplied interface seeds

An ordinary intent may contain several distinct complete interface seeds and
a physical usage requirement for each seed. The user supplies the seed
identities and polymer connections. Mosaic may solve declared symmetry
relations and bounded component poses, but does not create additional seed
types on the user's behalf.

If the intent cannot be lowered unambiguously, `resolve` fails closed and asks
for an expert component/path declaration.

## Execution profiles

An execution profile describes process launch and resources; it does not
change the scientific design.

### Direct execution

The bundled `local` profile uses synchronous direct execution and the default
checkpoint location:

```text
~/.foundry/checkpoints/rfd3_latest.ckpt
```

The profile can be copied and edited for any workstation or shared server.

### Scheduler execution

Copy the generic Slurm template:

```bash
rfd3-mosaic profiles --copy-slurm my-cluster.yaml
```

Set the site's partition, resources, environment activation and checkpoint,
then pass the absolute profile path:

```bash
rfd3-mosaic doctor --profile "$PWD/my-cluster.yaml"
rfd3-mosaic run design.yaml --profile "$PWD/my-cluster.yaml"
```

No public workflow requires access to a particular institutional cluster.

## Exit and acceptance behavior

RFD3-Mosaic distinguishes three outcomes:

- **configuration failure:** the design could not be validated or lowered;
- **runtime failure:** inference or artifact generation did not finish;
- **audit failure:** inference produced a structure, but one or more required
  scientific contracts failed.

Only a completed run with every required audit passing receives a `PASSED`
verdict.

## Advanced commands

- `inspect`: detect components and candidate interfaces in an input structure;
- `search`: enumerate and rank assembly-graph candidates;
- `central` and `interface`: compatibility shortcuts for the two primary
  workflows.

Run `rfd3-mosaic COMMAND --help` for the authoritative arguments supported by
the installed version. Expert schema and implementation details remain under
active development and should be pinned to a commit for reproducibility.
