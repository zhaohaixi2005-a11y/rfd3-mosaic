# RFD3-Mosaic quick start

The ordinary workflow is **`init → run → report`**. One task is one editable
YAML and one starting pose; `sampling.designs` controls how many diffusion
trajectories start from that pose. RFD3 generates the new backbone and its
secondary structure. A secondary-structure blueprint is not required.

This guide assumes the [installation and checkpoint setup](INSTALLATION.md)
are complete. It does not depend on a particular institution or GPU model.

## Check the installation once

```bash
rfd3-mosaic doctor --profile local
```

`doctor` checks the Python environment, RFD3 imports, execution profile and
checkpoint without running inference. `local` means direct execution on the
current compatible machine or allocated compute node.

## 1. Create a task

For a supplied interface whose internal geometry must remain fixed:

```bash
rfd3-mosaic init my-design.yaml \
  --input interface-seed.pdb \
  --side-a A20-35 \
  --side-b B40-55 \
  --symmetry C3 \
  --designs 50
```

Replace the input path and residue selectors with your own seed. Review the
generation lengths in `my-design.yaml`: they specify how many residues RFD3
will generate between the declared endpoints. Both interface sides form one
joint-rigid seed. The default adjacent-copy connection joins opposite sides
of neighbouring seeds, preserving the supplied interface itself.

`init` infers the supplied-interface task from both side selectors. For a
single motif that needs a new interface, use `--motif-selector A12-20` instead
of the two side selectors. An explicit `--task` remains optional; conflicting
selector forms are rejected.

The generated YAML keeps the scientific declaration, the explicit motion
policy and changed high-level preferences. Omitted preferences use their
normal defaults. It is the file to edit for subsequent runs.

The short supplied-interface initializer supports Cn. Dn and multiple seed
connection patterns require an explicit assembly graph; see the
[user workflow guide](WORKFLOW_GUIDE.md). Mosaic does not infer an assembly
architecture from the structure alone.

## 2. Run

```bash
rfd3-mosaic run my-design.yaml
```

`run` validates the declaration and freezes its compiled input before
inference. Standalone `plan my-design.yaml` and `validate my-design.yaml` are
available for inspection and diagnosis; they are optional steps.
`plan` gives a short task summary; `plan my-design.yaml --details` adds
compiler and guidance details. The CPU `validate` checks include the native
sampler guard and feature pipeline, without loading model weights or running
the model.

For a Slurm site, create and edit a profile once, then pass it to `run`:

```bash
rfd3-mosaic profiles --copy-slurm my-cluster.yaml
rfd3-mosaic run my-design.yaml --profile my-cluster.yaml
```

Follow the [profile setup instructions](INSTALLATION.md) to set the account,
partition, environment and checkpoint before submitting.

## 3. Read the report and structures

```bash
rfd3-mosaic report RUN_ID_OR_DIRECTORY
```

Use the run ID or directory printed by `run`. `status RUN_ID_OR_DIRECTORY`
shows progress. Completed structures appear incrementally as plain CIFs in
`generated_structures_cif/`; the finished run also provides a structure-only
ZIP. The report separates generated outputs, geometry-contract checks and
advisory measurements. A generated structure is not yet proof of folding or
experimental success; flagged outputs remain available for inspection.

## Choose motion and diversity when needed

- `--component-motion locked` is the default: the chosen seed pose stays fixed.
- `--component-motion guided` allows bounded, guided motion of the whole seed.
- `--component-motion free` allows bounded SE(3) motion of the whole seed.

These choices preserve the complete seed's internal geometry. All 50 designs
in the example share their initial pose; mobile designs may end in different
poses. To explore different starting poses, declare a pose distribution and
use `prepare-poses` to create separate tasks. Renaming a task does not change
its pose. `replicates_per_pose` is deprecated; sharing is automatic. See
[task pose rules](TASK_POSES.zh-CN.md).

## Further options

Use `rfd3-mosaic --help` and `rfd3-mosaic init --help` for common commands and
options; `--help-all` shows the complete reference. Advanced commands remain
available.

The [user workflow guide](WORKFLOW_GUIDE.md) covers generation lengths,
terminal extensions, multiple components, sequence/ligand conditioning and
motion controls. The [CLI reference](USER_CLI.md) retains the full advanced
workflow. For an explicitly supplied complete backbone, experimental
[`prepare-scaffold`](COMPLETE_SCAFFOLD.zh-CN.md) provides reference-conditioned
partial diffusion with locked or coupled mobile seeds; this is an optional
workflow with its own supported scope.
