# RFD3-Mosaic user CLI

Before rendering or submitting a job, inspect the fully resolved plan:

```bash
rfd3-mosaic plan design.yaml
```

For machine-readable output:

```bash
rfd3-mosaic plan design.yaml --format json
```

`plan` does not create a run directory or submit a job. It reports the
effective motif constraints, sampler preset, timesteps, execution backend,
Slurm profile, output root, Mosaic commit and Foundry base commit.

The routine public interface is one command. Users should not copy or edit
long implementation-oriented Slurm scripts, and they do not need to write
`topology.kind` or exact-symmetry sampler settings themselves.

## Simplest central-motif run

```bash
python -m rfd3_mosaic.cli central \
  --input /path/to/adapter/rfd3_input.json \
  --motif B1-31 \
  --n-length 35 \
  --c-length 35 \
  --profile p100 \
  --output /path/to/runs
```

`--steps` defaults to 200 and `--seed` defaults to 42. Add `--dry-run` to
validate and render without submitting. The command generates and preserves
the complete internal experiment YAML automatically.

## Simplest interface-seed run

```bash
python -m rfd3_mosaic.cli interface \
  --config /path/to/interface_seed_config.yaml \
  --manifest /path/to/pose_candidate/manifest.json \
  --length 85 \
  --profile h100 \
  --output /path/to/runs
```

The YAML workflow below remains available for advanced, reviewed and batch
experiments.

## Supported design topologies

`interface_seed` fixes a complete cross-subunit interface orbit and generates
the scaffold segment between the fixed fragments. `central_motif` fixes one
motif orbit in the middle of each protomer and generates defined N- and
C-terminal regions around it.

Both use the `exact_mosaic` preset. That preset is intentionally not a loose
collection of user switches: it resolves to no realignment, exact symmetry
projection, coupled noise, complete fixed-motif constraint groups and
motif-precedence finalization.

## Quick start

Copy the closest example and edit only its paths and design parameters:

```bash
cp examples/rfd3_mosaic/interface_seed_cn.yaml my_experiment.yaml
# or: cp examples/rfd3_mosaic/central_motif_c3.yaml my_experiment.yaml
```

Then validate, inspect the generated job, and submit:

```bash
python -m rfd3_mosaic.cli validate my_experiment.yaml
python -m rfd3_mosaic.cli render my_experiment.yaml
python -m rfd3_mosaic.cli submit my_experiment.yaml
```

After an editable/package installation, the equivalent short command is:

```bash
rfd3-mosaic validate my_experiment.yaml
rfd3-mosaic submit my_experiment.yaml
```

Use another cluster profile without editing the experiment:

```bash
rfd3-mosaic submit my_experiment.yaml --profile h100
```

Built-in profiles are `p100`, `h100`, and `a100_80g`. An external YAML profile
path is also accepted, so machine-specific setup remains outside the scientific
experiment file.

## What the command does

```text
validate experiment and reject unknown fields
-> resolve all paths and the exact sampler preset
-> freeze resolved_config.yaml and provenance hashes
-> render a short generated_job.sbatch
-> compile the topology-specific RFD3 input in the allocated job
-> prevalidate the real RFD3 input
-> run exact symmetric diffusion
-> run motif/seed and scaffold audits
-> pass only if every required audit passes
```

Run outputs use:

```text
<output.root>/<output.campaign>/<experiment.name>/<SLURM_JOB_ID>/
```

Each run contains the frozen resolved config, input artifacts, Slurm logs,
RFD3 result, topology-specific motif audit, scaffold audit, provenance, and
`experiment_summary.json`. A failed worker records `status: failed` and the
exception before returning a non-zero job exit.

## Configuration boundary

Ordinary users choose topology inputs, generated lengths, diffusion steps,
random seed, output location and execution profile. They do not directly set
the internal exact-symmetry correctness switches. Advanced research controls
and diagnostic A/B arms remain available in the existing specialized scripts,
but are not the default product interface.

The example files contain placeholder storage paths and must be copied and
edited before validation. The central-motif `template_input` must be an
existing symmetric RFD3 adapter input containing a validated runtime transform
registry; it is not a raw PDB/CIF path.
