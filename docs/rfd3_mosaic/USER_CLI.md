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

## Design workflows

### Preserve supplied geometry

Start from:

```bash
cp examples/rfd3_mosaic/simple_interface_seed.yaml design.yaml
```

Use this mode when the input already contains one or more complete interfaces.
Every supplied interface is preserved as a geometric entity. Generated
regions connect user-declared endpoints without changing the internal seed
geometry.

### Create a symmetric interface

Start from:

```bash
cp examples/rfd3_mosaic/simple_central_motif.yaml design.yaml
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
cp configs/rfd3_mosaic/execution/slurm-example.yaml my-cluster.yaml
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
