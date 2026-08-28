# RFD3-Mosaic command-line reference

For a task-oriented guide that separates required scientific input from
optional controls and includes complete fixed-motif, supplied-interface and
multi-component examples, start with the
[complete user workflow guide](WORKFLOW_GUIDE.md).

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

For supplied interfaces, `--interface-scaffold` distinguishes covalent
topology from non-covalent scaffolding. `adjacent-linker` creates one declared
symmetry-neighbour linker. `terminal-extensions` grows N/C scaffold from both
partners without connecting the partners to one another. Add
`--new-oligomer-interface` only when the preserved interface seed should form
an additional Cn oligomerization interface. The default remains off.

Native RFD3 sequence and ligand conditioning can be initialized with
`--sequence-conditioning fixed|masked|glycine` and repeatable
`--ligand-selector`. `masked` keeps selected backbone coordinates but unfixes
sequence through RFD3's `select_unfixed_sequence`; `glycine` writes a
backbone-only all-glycine conditioning fragment. Selected ligands are attached
to the same joint-rigid motion group and are expanded with the declared Cn
symmetry.

Add `--redesign-motif-sidechains` when the supplied protein backbone must stay
fixed but RFD3 should redesign its sequence and side-chain coordinates. This
can be combined with `--sequence-conditioning masked`.

For `central-motif`, `--component-motion locked|guided|free` selects whether
the supplied arrangement stays fixed, moves in the calibrated constrained
subspace, or uses bounded SE(3). High-level `--packing`, `--interface-area`,
`--cavity` and `--diversity` preferences are also available. Hard symmetry,
motif, continuity and clash contracts are never disabled by these options.

`--designs N` controls how many independently instantiated designs one run
produces.  When the design declares a variable initial pose (for example a
radius interval or `uniform_so3` orientation), every design receives its own
feasible pre-diffusion pose and its own diffusion seed.  A fully fixed
arrangement retains one exact pose and varies only diffusion.  The equivalent
YAML fields are:

```yaml
sampling:
  timesteps: 200
  designs: 100
  replicates_per_pose: 1  # default: one trajectory per variable pose
  seed: 42
  dump_trajectories: false  # optional; trajectory files can be large
  initial_pose:
    radius: {minimum: 20.0, maximum: 32.0}
    axial_offset: {minimum: -4.0, maximum: 4.0}
    orientation: {method: uniform_so3}
    seed: 1000
  screening:
    mode: advisory
    protocol: auto
    retain_all_outputs: true
```

The corresponding `init` arguments are `--pose-radius-minimum`,
`--pose-radius-maximum`, `--pose-axial-minimum`, `--pose-axial-maximum`,
`--pose-orientation`, `--pose-maximum-tilt-deg`, `--pose-seed` and
`--replicates-per-pose`. Mosaic does not silently invent a task-specific
radius or orientation prior.

Set `replicates_per_pose` above one only when intentionally comparing several
diffusion trajectories from the same assembly hypothesis. Mosaic samples pose
coordinates from the declared radius/axial distribution and Haar-uniform
SO(3), rejects geometrically invalid proposals, freezes the accepted inputs,
and sends them to RFD3 as one multi-example input. The model/checkpoint is
loaded once. Every result has its own structure, metadata, semantic audits and
scaffold audit below `audits/<design-id>/`.
Pose sampling does not prefer a central pore or one cage silhouette unless the
user explicitly declares a corresponding assembly-shape target.
The run report records generated outputs, geometry-contract flags and
advisory recommendations. A flagged design is never deleted. Set
`screening.mode: off` to suppress recommendations; destructive screening is
not part of the public schema. `protocol: hoyeung_lhd101` records campaign
intent, but the published cohort-median loop/Rg selection is only computed
after the cohort exists.
`pose_manifest.json` records the exact pose seed, diffusion seed, compiled
input and SHA256 for every output. The user supplies one YAML; Mosaic does not
create one authoring YAML per design.
For very large campaigns, choose a scheduler walltime that can accommodate the
requested count or split the total across several seeds/jobs.

Every run creates `generated_structures_cif/` when inference starts. As each
compressed RFD3 result finishes, Mosaic validates and atomically mirrors it
there as a plain `.cif`; `manifest.json` records the files already available.
Users can therefore inspect completed designs while a long multi-design job is
still running without opening a partially written gzip stream.

After the full run completes, Mosaic also writes
`generated_structures_cif.zip`. The ZIP contains only plain `.cif` members—no
configuration, logs, audits or source snapshot—so a run with
`sampling.designs: N` and N produced outputs has exactly N CIF members. The
adjacent
`generated_structures_cif_manifest.json` records the requested/produced counts
and archive SHA256 without being placed inside the structure-only ZIP.

### `examples` and `profiles`

```bash
rfd3-mosaic examples
rfd3-mosaic examples --copy supplied-interface --output design.yaml
rfd3-mosaic examples --copy supplied-interface-oligomer --output oligomer.yaml
rfd3-mosaic profiles
rfd3-mosaic profiles --copy-slurm my-cluster.yaml
```

These discovery commands work from both source checkouts and installed
packages. Copied examples resolve their bundled input and run paths; copied
Slurm profiles are generic site templates. Both support `--format json`, and
neither overwrites an existing file unless `--force` is supplied.

## Public YAML reference

A normal user configuration has seven conceptual sections. Only `name`,
`input`, `symmetry`, one or more generated regions and their fixed selectors
are essential; all conditioning and execution controls are optional.

| Section | Purpose |
| --- | --- |
| `task` | preserve a supplied interface or create a new symmetric interface |
| `generation` | terminal extensions or explicit between-linkers and lengths |
| `constraints` | fixed selectors and joint-rigid `coupling_group` identity |
| `conditioning` | optional native RFD3 sequence, ligand, RASA, hotspot and H-bond inputs |
| `sampling` | timesteps, design count, independent pose distribution and RFD3 global conditioning |
| `resources` | local or Slurm execution profile |
| `output` | run root and campaign name |

The maintained complete examples are the normative templates. In particular,
`supplied-interface-oligomer` demonstrates preserved non-covalent partners,
independent terminal scaffold, an optional additional generated interface and
per-design SE(3) pose sampling. The identifier is the name of one maintained
example, not a restriction to two-component assemblies; general assembly YAML
may declare multiple components and multiple interface relations.

### Native RFdiffusion3 conditioning

The public `conditioning` block supports:

- `sequence`: `masked` or `glycine` treatment of a complete materialized
  protein fragment;
- `ligands`: one-residue non-polymers coupled to a named rigid group;
- `buried`, `partially_buried`, `exposed`;
- `hotspots`;
- `hbond_acceptors`, `hbond_donors`;
- `redesign_motif_sidechains`;
- `origin_strategy: com|hotspots`.

The public `sampling` block additionally exposes `is_non_loopy` and
`plddt_enhanced`. Mosaic maps source selectors to the compiled RFD3 input and
then invokes RFD3's own parser during prevalidation. Unsupported combinations
(for example a user-selected origin on a quotient input whose group origin is
compiler-owned) fail before inference rather than being silently ignored.

RFdiffusion3 partial diffusion is not represented as symmetric motif
scaffolding: it changes the coordinates supplied as the starting structure and
conflicts with Mosaic's exact fixed-geometry contract. Users needing an
unmodified native partial-diffusion experiment should run that native RFD3
workflow rather than assuming Mosaic applied it.

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

Very high-order explicit-all-copy designs can require substantially more
memory to construct and prevalidate than a shared login node provides. For
those designs, submit with:

```bash
rfd3-mosaic run design.yaml \
  --profile large-gpu.yaml \
  --defer-runtime-preflight
```

This option is not a validation bypass. Mosaic still performs schema,
selector and constraint-binding checks before submission. Complete expanded
assembly construction and RFD3 runtime-feature prevalidation then run inside
the allocated worker before inference; a failed preflight prevents model
execution. Ordinary designs should keep the default eager preflight.

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
rfd3-mosaic audit RUN_ID_OR_DIRECTORY --reuse-reports
```

- `status` summarizes execution state, generated outputs, contracts and
  advisory diagnostics.
- `report` writes JSON and HTML reports from recorded run artifacts.
- `audit` evaluates a compatible existing run without rerunning diffusion or
  converting a quality proxy into an execution failure.
- `audit --reuse-reports` refreshes status and advisory screening from an
  already complete audit set without recomputing geometry or importing RFD3.
  This is useful after reporting-policy updates or on memory-limited login
  nodes. A failed audit attempt is recorded separately and never rewrites a
  previously completed inference run as failed.

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
  --side-a A20-35 \
  --side-b B40-55 \
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

#### Sequence conditioning on a fixed interface seed

Coordinate geometry and amino-acid identity are separate contracts.  By
default, a fixed motif keeps both.  When an existing polar surface should be
repacked, native RFD3 sequence conditioning can be requested explicitly:

```yaml
conditioning:
  sequence:
    - selector: A20-35
      mode: masked
    - selector: B40-55
      mode: masked
```

`masked` emits RFD3's `select_unfixed_sequence` and fixes only backbone atoms;
the joint-rigid backbone geometry is preserved while residue identities may
change.  `glycine` instead writes an explicit glycine-backbone conditioning
fragment and keeps that glycine identity fixed, reproducing the traditional
all-Gly surface-conditioning control.  Conditioning selectors must be
complete materialized `fixed_xyz` selectors.  Split a `fixed_xyz` declaration
when only a defined residue range should be masked or converted to glycine.

Omitting `conditioning` is byte-compatible with earlier designs: all motif
atoms and sequence identities remain fixed.  Mosaic fails validation if a
masked/Gly selector would retain fixed side-chain atoms, so an input cannot
claim that sequence is free while leaking the original side chains.

`task: preserve_supplied_geometry` never invents a second generated interface.
The supplied interface remains the oligomeric contact; generated residues are
shaped as a monomer scaffold. `sampling.scaffold_packing: symmetric_generated`
is rejected for this task because it has the different meaning “create a new
generated--generated symmetry-neighbour interface”.

For a cyclic interface seed, the two physical sides of one supplied interface
remain a **non-covalent same-copy pair**. A generated protomer may instead join
one side to the opposite side of an adjacent symmetry copy. When the sampled
SO(3) pose can reverse which adjacent copy is nearer, use:

```yaml
generation:
  - kind: between
    from_selector: B40-55
    to_selector: A20-35
    orbit_offset: nearest_adjacent
    length: {minimum: 70, maximum: 100}
```

The compiler compares only the two non-zero cyclic neighbours (`+1` and
`-1`), selects the smaller fixed-fragment CA-COM separation for that design's
sampled pose, and freezes the resulting integer offset in the compiled
Assembly IR. It never considers offset `0`, so it cannot turn the supplied
non-covalent interface itself into a peptide connection. The evaluated
distances and selected offset are recorded as `automatic_copy_relations` in
the RFD3 provenance. Non-cyclic groups continue to require an explicit named
group relation.

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

Movable rigid bodies use a three-stage schedule defined by percentages of each
component's declared active window, so the behavior is independent of whether
the run uses 50, 100 or 200 diffusion steps:

| Active-window interval | Phase | Relative proposal range |
| --- | --- | --- |
| first 40% | capture | up to 100% of the declared per-step SE(3) trust region |
| next 40% | settle | up to 50% of the declared per-step trust region |
| final 20% | polish | up to 20% of the declared per-step trust region |

The declared `max_translation` and `max_rotation_deg` remain cumulative hard
bounds in every phase. Energy-improving line search may shorten or reject an
unsafe proposal; the schedule permits a substantial early correction but does
not require movement when the current pose is already locally consistent.
After `end_fraction` the pose is frozen. This schedule never applies to
`pose.mode: fixed`, and it never changes the internal coordinates of a
joint-rigid seed.

The exact SE(3) state, objective terms, gradients, subspace projections,
phase-response formula, line search, cumulative bounds and atomic rollback
conditions are specified in the
[rigid-mobility mathematical contract](RIGID_MOBILITY_MATHEMATICAL_CONTRACT.md).
Its controller energy is an inference-time local geometry objective, not a
physical free energy or a claim that the final backbone will fold.

Existing frozen `packing_preferences_v1` runs retain their original behavior.
Newly compiled `packing_preferences_v2` designs keep
`intra_chain_weight=0` unless the user requests monomer-core guidance, while
the broad generated-interface contact prior is calibrated by the existing
`packing: loose|balanced|tight` preset (`0.06`, `0.10`, or `0.15`). An explicit
`inter_chain_weight` overrides that prior directly.

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

## Exit and result behavior

RFD3-Mosaic distinguishes execution from measured checks:

- **configuration failure:** the design could not be validated or lowered;
- **runtime failure:** inference or artifact generation did not finish;
- **generated:** every expected raw structure was generated and retained;
- **checks:** geometry, safety and task-objective measurements are reported
  independently beside the generated structures.

Mosaic does not infer whether a user will like or adopt a generated structure.
The terminal, HTML, JSON and text reports expose measured checks and potential
risks; the user makes the final selection.  Fixed-geometry, symmetry,
continuity and clash checks remain explicit facts and are never hidden merely
because a raw CIF exists.

Every run keeps three artifact roles separate:

- `input/presymmetrized_input.cif` is the sole compiled input for one-pose
  runs; a multi-pose run instead stores `input/pose_XXXXX/` inputs plus one
  combined `input/rfd3_input.json`; none is a generated design;
- each root-level `*_model_0.cif[.gz]` is one raw generated design, including
  outputs retained after a contract or advisory check flags them;
- a PyMOL `mosaic_aligned*` object is an in-memory visualization copy and is
  not an additional generated structure.

Consequently, `sampling.designs: 2` with a variable initial pose produces two
raw result CIFs from two independently compiled poses by default. With fixed
geometry it produces two trajectories from the same exact input. Always use
the per-design contract flags, advisory metrics and downstream refolding to
decide whether either result is useful for the user's objective.

## Advanced commands

- `inspect`: detect components and candidate interfaces in an input structure;
- `search`: enumerate and rank assembly-graph candidates;
- `central` and `interface`: compatibility shortcuts for the two primary
  workflows.

Run `rfd3-mosaic COMMAND --help` for the authoritative arguments supported by
the installed version. Expert schema and implementation details remain under
active development and should be pinned to a commit for reproducibility.
