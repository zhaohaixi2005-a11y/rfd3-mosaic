# RFD3-Mosaic user CLI

## Inspect real capability maturity

Before writing a design, inspect which features are stable, engineering,
CPU-only or planned:

```bash
python -m rfd3_mosaic.cli capabilities
python -m rfd3_mosaic.cli capabilities --format json
```

The JSON form is the canonical machine-readable capability ledger. A feature
being accepted by the schema does not imply that it has a GPU backend.

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

## New topology-neutral design declaration (development interface)

The first strict public schema is now available for validation and planning:

```yaml
schema_version: 1
name: prism-c3
input: Prism_C3_G2.pdb
symmetry: C3

generation:
  - kind: between
    from_selector: A12-20
    to_selector: A26-37
    length: 90

constraints:
  - kind: cylindrical
    selector: A12-20,A26-37
    atoms: ca
    axis: symmetry
    keep: [radius, azimuth]

sampling:
  timesteps: 200
  seed: 42

resources:
  profile: h100

output:
  root: /path/to/runs
  campaign: prism-c3
```

Inspect it with the same executable:

```bash
rfd3-mosaic validate design.yaml
rfd3-mosaic plan design.yaml
```

The operators are optional. With no `constraints` field, the constraint plan
is empty and undeclared degrees of freedom remain normal diffusion degrees of
freedom. The canonical operators currently represented are:

- `fixed_xyz`: preserve the selected atoms as a rigid geometry component;
- `cylindrical`: preserve selected radius, azimuth and/or axial coordinates
  about the declared symmetry axis;
- `bounded_mobile`: allow selected rigid-pose degrees of freedom only inside
  explicit bounds.

The old descriptive spellings `full_xyz_fixed`, `ca_cylindrical_fixed` and
`bounded_mobile_interface` are accepted and canonicalized; they do not create
different pipelines.

`fixed_xyz` does **not** lock a design to the input file's laboratory frame.
One common translation or rotation of a complete fixed component is physically
irrelevant and is removed before its RMSD is evaluated. What is protected is
the full internal and relative geometry of every atom in that component,
including the relative placement of all symmetry copies.

One declaration is one component. A comma-separated selector within that
declaration is therefore fitted and audited jointly:

```yaml
constraints:
  - kind: fixed_xyz
    selector: A12-20,A26-37
    atoms: all
```

Several declarations are independent unless they name the same
`coupling_group`. Use a shared group when spatially separate selections must
retain their mutual pose and be superposed with one common rigid transform:

```yaml
constraints:
  - kind: fixed_xyz
    selector: A12-20
    coupling_group: catalytic_site
  - kind: fixed_xyz
    selector: B26-37
    coupling_group: catalytic_site
```

Omit `coupling_group` when each selection may have its own rigid-body gauge.
The audit then aligns each component independently. Absolute-coordinate error
is not a public constraint and never determines pass/fail.

The component pose is fixed by default. To let independently declared
components adapt to the generated scaffold while preserving every internal
distance, set bounded rigid mobility explicitly:

```yaml
constraints:
  - kind: fixed_xyz
    selector: A12-20
    coupling_group: mobile_left
    pose:
      mode: bounded_mobile
      max_translation: 3.0
      max_rotation_deg: 10.0
      start_fraction: 0.05
      end_fraction: 0.75
      response: 0.2
      max_step_translation: 0.25
      max_step_rotation_deg: 1.0

  - kind: fixed_xyz
    selector: A26-37
    coupling_group: fixed_right
    # pose.mode defaults to fixed
```

Here `mobile_left` and `fixed_right` are distinct components. The first may
translate and rotate inside its declared cumulative and per-step bounds; the
second remains at its compiled pose. Both retain their complete internal and
symmetry-copy geometry. Declarations sharing one `coupling_group` must use
identical `pose` settings because they represent one rigid component.

Designs in which every generated-region endpoint is explicitly covered by
`fixed_xyz` with `atoms: all` can be rendered or submitted. This includes
both the default fixed-pose components and components that opt into nested
`pose.mode: bounded_mobile`:

```bash
rfd3-mosaic render design.yaml
rfd3-mosaic submit design.yaml
```

The separate legacy-style top-level `bounded_mobile` operator is not the
component-pose control above and remains unavailable to the executable public
backend. `cylindrical`, partial-atom fixed XYZ, unconstrained endpoints, or
fixed regions detached from every generated region also fail with a direct
backend/lowering error. These cases are not silently converted to historical
adapter behavior. The stable central/interface compatibility commands below
remain available during migration.

Every submitted design with a bounded-mobile component automatically requires
three complementary checks. The constraint audit jointly superposes all fixed
fragments within each symmetry copy and verifies their internal rigid
geometry; the assembly symmetry audit verifies the relationship among copies;
and the component-mobility audit verifies that runtime updates were active and
never exceeded the declared cumulative translation or rotation bounds. A
fixed-pose component instead retains the stronger complete-orbit joint-fit
contract against its initial pose.

The routine public interface is one command. Users should not copy or edit
long implementation-oriented Slurm scripts, and they do not need to write
`topology.kind` or exact-symmetry sampler settings themselves.

## Static pose sampling versus diffusion sampling

These controls are intentionally separate. `sampling.initial_pose` applies
one rigid-body transform to the complete motif motion group before RFD3
starts. The outer `sampling.seed` controls diffusion randomness and does not
choose that pose.

```yaml
sampling:
  initial_pose:
    radius: {minimum: 20.0, maximum: 30.0}
    axial_offset: {minimum: -3.0, maximum: 3.0}
    radial_direction: [1.0, 0.0, 0.0]
    orientation: {method: uniform_so3}
    seed: 3000
  timesteps: 200
  seed: 42
```

The current public executable subset realizes one deterministic initial pose
per run. Radius and axial offset are sampled uniformly from the declared
intervals; `uniform_so3` is Haar-uniform. A fixed orientation may instead be
declared with `method: fixed` and `rotation_deg: [x, y, z]`.

Omit `initial_pose` entirely to keep the input pose unchanged. This setting
does not move a motif during diffusion; timestep mobility is a different,
currently experimental capability.

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

## Immutable submission identity

`render` and `submit` must run on a host that can read the selected RFD3
checkpoint. At render time Mosaic records SHA256 identities for:

- the complete Git source state, including tracked changes and untracked-file
  contents;
- the experiment, execution profile and Foundry compatibility contract;
- every assembly specification, fragment PDB/mmCIF, central-motif template,
  central-motif source structure and pose-candidate manifest used by the run;
- the exact RFD3 checkpoint.

The allocated worker verifies the frozen `resolved_config.yaml`, repository,
runtime dependencies and checkpoint before compiling an input or importing
RFD3. A missing or changed dependency fails the job with a direct identity
error instead of silently running different software or geometry.

Every rendered submission also contains `source_snapshot.tar.gz`. The job
verifies its archive digest, extracts it into `$RUN_DIR/software`, verifies
the per-file source manifest, and imports Mosaic, RFD3 and Foundry from that
private snapshot. Editing or synchronizing the shared checkout while a job is
queued therefore cannot change the code executed by that job. Runtime design
inputs remain external files and retain their separate fail-closed hashes.

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
