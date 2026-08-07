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

Before using a scheduler, run the strict public-design preflight:

```bash
rfd3-mosaic validate design.yaml
```

Validation does more than parse YAML. It binds selectors, lowers the design to
the common Assembly IR, expands every declared symmetry copy and applies the
standalone clash/interface geometry gates. Temporary compiler artifacts are
discarded, so the command writes no persistent run files. `render`, `run` and
`submit` execute the same preflight automatically and cannot bypass a geometry
failure.

## Run, inspect and report without hand-written shell

The canonical execution command is now:

```bash
rfd3-mosaic run design.yaml
```

`submit` remains a compatibility alias. Both commands validate the public
design, freeze its software/input/checkpoint provenance, render the selected
execution profile and submit it. Use `--dry-run` to stop after rendering.

Once submitted, inspect a known run directory directly:

```bash
rfd3-mosaic status /path/to/runs/campaign/design/5733788
```

Or find it from a Slurm JobID without writing a `find` command:

```bash
rfd3-mosaic status 5733788 --root /path/to/runs
```

`RFD3_MOSAIC_RUN_ROOT=/path/to/runs` may replace `--root`. Add
`--format json` for the canonical machine-readable status. The status combines
Slurm state, `experiment_summary.json`, every declared audit and discovered
structure/log artifact. A Slurm `COMPLETED` state is **not** reported as a
scientific pass by itself: `passed=true` requires a completed worker summary
and all required audits to pass.

Generate a portable report next to a completed run with:

```bash
rfd3-mosaic report /path/to/run
```

This writes `mosaic_report.html` and `mosaic_report.json`. The HTML is
self-contained and can be copied off the cluster; the JSON contains the same
status contract for automation. Use `--output /path/report.html` to choose a
different destination. `--no-scheduler` makes either command inspect only the
immutable run artifacts.

New submissions are registered under
`OUTPUT_ROOT/.rfd3-mosaic/jobs/<JOB_ID>.json`. This makes numeric JobID lookup
constant-time and gives the worker one durable place to record
`submitted -> running -> completed/failed`. List indexed work with:

```bash
rfd3-mosaic runs --root /path/to/runs
rfd3-mosaic runs --root /path/to/runs --format json
```

Runs created before this index remain readable through the existing discovery
fallback; they do not need to be moved or renamed. To import all historical
worker summaries into the index once, run:

```bash
rfd3-mosaic runs --root /path/to/runs --rebuild
```

The rebuild is idempotent, preserves newer submission metadata when present,
and reports malformed historical directories instead of silently omitting
them.

The CLI no longer invokes `sbatch` inline. Execution goes through a versioned
executor boundary, and current LRZ profiles resolve to `executor: slurm`.
This intentionally preserves identical Slurm behavior while creating the
correct extension point for local execution and other schedulers.

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
      proposal: scaffold_objectives
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

`proposal` selects the runtime pose signal. `denoiser_fit` is the compatibility
default and follows the denoiser's motif coordinates. `scaffold_objectives`
uses the generated scaffold boundaries and geometry objectives to propose a
bounded rigid translation and rotation. Several independently coupled
components may use `scaffold_objectives` in one run. Their proposals are
computed from the same immutable timestep snapshot and are accepted or
rejected atomically against one joint assembly objective. This multi-orbit
path is CPU-validated and remains an engineering interface until its 50-step
GPU gate passes. Keeping at least one component fixed is still useful when a
specific relative-pose change must be measured against an explicit gauge
anchor, but it is no longer a runtime restriction.

Advanced users may restrict the translation to symmetry-axis-aware
coordinates instead of enabling arbitrary SE(3):

```yaml
pose:
  mode: bounded_mobile
  subspace: radial          # or radial_axial
  proposal: scaffold_objectives
  max_translation: 3.0
```

`radial` changes only the distance from the declared symmetry axis;
`radial_axial` additionally permits translation along that axis.  Neither
mode rotates the component.  Omitting `subspace` retains full
`bounded_se3`, which also requires `max_rotation_deg`.

These coordinates are execution primitives, not a requirement that routine
users know the correct radius in advance.  The intended default product
workflow is an adaptive pose planner that derives feasible bounds from
linker reachability, group closure and clash-free geometry, records the
resolved values in provenance, and then uses the same projector.  Until that
planner is validated, automatic numeric bounds are not invented silently;
the explicit form above remains the reproducible expert interface.

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

Mobile runs also write `mobility_trajectory.json`. It records stable orbit and
component identifiers, per-timestep junction/clash/tilt/prior terms, proposed
translation and rotation increments, the joint energy change and whether each
proposal was committed or rolled back. For several mobile orbits, the audit
requires complete finite objective records and explicit atomic-joint decision
evidence; a final pose inside its numeric bounds is not sufficient by itself.

## Assembly graph for more than two seeds

Complex rings and cages are declared as a graph, not as `left`, `right`,
`third`, and progressively more topology-specific fields:

- `components` are rigid seed or motif nodes;
- `ports` are named interface faces owned by those rigid nodes;
- `interfaces` are geometric relationship edges between ports (or, for
  backward compatibility, directly between components when no ports exist);
- `connections` are directed peptide regions that RFD3 must generate;
- the declared `symmetry` expands every node and edge through one exact group
  action.

There is no two-component schema limit:

```yaml
schema_version: 1
name: three-seed-c3
input: motif.pdb
symmetry: C3

components:
  seed_alpha:
    selectors: [A12-20]
  seed_beta:
    selectors: [A26-37]
  catalytic_site:
    selectors: [B10-18, C30-42]
    geometry: joint_rigid

interfaces:
  - id: alpha_beta
    between: [seed_alpha, seed_beta]
    relation: {mode: preserve_input}
  - id: beta_site
    between: [seed_beta, catalytic_site]
    copy_relation: {orbit_offset: 1}
    relation:
      mode: contact
      distance: {minimum: 3.0, maximum: 8.0}

connections:
  - id: alpha_to_beta
    from: seed_alpha.C
    to: seed_beta.N
    length: {minimum: 25, maximum: 40}
```

For cage building blocks with several differently oriented faces, keep the
faces in one `joint_rigid` component and expose each face as a separate port:

```yaml
components:
  protomer_seed:
    selectors: [A12-20, A26-37, A45-53]
    geometry: joint_rigid

ports:
  face_alpha:
    component: protomer_seed
    selectors: [A12-20]
  face_beta:
    component: protomer_seed
    selectors: [A26-37]
  face_gamma:
    component: protomer_seed
    selectors: [A45-53]

interfaces:
  - id: alpha_beta_neighbour
    between: [face_alpha, face_beta]
    copy_relation: {transform: T:g01}
    relation: {mode: preserve_input}
  - id: gamma_homotypic_neighbour
    between: [face_gamma, face_gamma]
    copy_relation: {transform: T:g04}
    relation: {mode: preserve_input}
```

The three faces retain one common building-block pose, but each edge targets
a distinct symmetry neighbour. A port may connect to itself only through a
non-identity copy relation; an identity self-edge is rejected. In the current
backend every port selector must exactly match one complete selector owned by
its component. This explicit restriction prevents the compiler from silently
splitting a rigid fragment or guessing atom correspondence.

`geometry: rigid` is intentionally limited to one contiguous selector.
Several spatially separate fragments that must retain one common relative
pose use `geometry: joint_rigid`; they compile to one motion group and one
coupled fixed-geometry audit component. Components remain mutually independent
unless an interface relation connects them.

The compact connection spelling `component.C` to `component.N` is sufficient
for single-selector components. If a component has several selectors, use an
explicit endpoint so the compiler cannot guess which physical terminus is
intended:

```yaml
to:
  component: catalytic_site
  selector: C30-42
  terminus: n
```

For the two common motif-scaffolding tasks, users do **not** need to declare
ports, interface edges, contact counts or packing locations. The compiler
infers the task from the generation contig:

- `between` joins supplied fixed fragments. A cross-chain coupling group is
  treated as an existing interface seed and its complete orbit is restored.
- `terminal` grows away from a fixed central motif. Mosaic creates an internal
  output-stage interface objective between generated symmetry-neighbour
  chains and derives its coverage and contiguous-patch targets automatically.

Mosaic calls this the **simple user mode**. It is inferred automatically; no
extra `mode:` switch is required. A simple design normally declares only the
input motif, symmetry, generated contig, fixed selection, lengths, compute
profile and output. If the motif sits on a symmetry stabilizer (for example at
the origin), the compiler deterministically chooses a non-degenerate group
orbit pose and its nearest valid symmetry neighbour. If the input pose is
already usable, its frame is retained.

The explicit `components / ports / interfaces / connections` graph is the
**expert mode** for multi-face cages where several different neighbours
must be named. It is not required for an ordinary central motif or supplied
interface seed. Both modes compile into the same `AssemblySpecification` and
use the same RFD3 sampler, projector, provenance and audits; expert mode does
not select a separate legacy execution path.

Copy-ready examples are
`examples/rfd3_mosaic/simple_central_motif.yaml` and
`examples/rfd3_mosaic/simple_interface_seed.yaml`. The former asks Mosaic to
design a new oligomer interface around a central motif; the latter preserves a
supplied rigid interface geometry and generates its connecting scaffold.

`preserve_input` freezes the reference relative transform of its two ports and,
by default, also requires at least one inter-port heavy-atom contact below
4.5 A. `contact` can be written simply as `{mode: contact}`. Mosaic derives a
scale-aware residue-coverage target and a contiguous contact-patch target from
the generated regions; users are not expected to invent contact counts.
Legacy numeric contact and distance fields remain accepted only for expert
replay or controlled ablation experiments.
`copy_relation` defaults to the corresponding symmetry copy
(`orbit_offset: 0`) and may be set explicitly when an edge targets a
neighbouring cyclic/dihedral copy or a named finite-group transform.

Interface edges do not redefine component rigidity. The RFD3 fixed-atom
groups are compiled from component nodes, while relationship edges determine
how the same unified sampler treats the surrounding generated scaffold:

- `preserve_input` is an **input-stage exact relation**. The complete supplied
  interface orbit is restored by the hard projector at every timestep.
- `contact` is an **output-stage design target**. It automatically activates
  graph-scoped interface guidance inside the same sampler. The fixed motif
  does not move; generated residues on the two concrete symmetry-neighbour
  chains receive a bounded attractive/contact field followed by the same
  exact symmetry and fixed-orbit projection.

There is no separate “interface-seed sampler” and “interface-design sampler”.
Both declarations lower through the same assembly graph, Assembly IR, RFD3
adapter and timestep loop. Only the edge contract changes.

The result gate writes `assembly_interface_relation_audit.json`. Every
required symmetry-expanded `preserve_input` edge must retain its declared
relative transform and reference contact count. For an output-stage `contact`
edge, the audit excludes all input-mapped fixed motif residues and measures
the newly generated heavy atoms on the two concrete output chains. Thus a
perfectly restored motif cannot hide a missing designed interface. A required
edge with too few generated contacts, an out-of-range distance or a sub-2 A
overlap fails the run. `rfd3-mosaic status` and `report` expose that verdict.

The current design-interface controller is the first bounded CA-level packing
field, not a learned interface-quality oracle. It is symmetry-neighbour aware
and avoids the legacy all-to-all radial collapse. Its joint energy combines
nearest-pair attraction, balanced residue coverage on both interface sides,
contiguous-patch formation, residue-normalized clash repulsion and an optional
COM-distance target. Symmetry copies are averaged within each declared edge
before multiple source interfaces are combined, preventing orbit multiplicity
from acting as an accidental weight.
Same-chain adjacent token updates are smoothed before the bounded step to
reduce local backbone crumpling. Runtime reports expose these terms
separately. Sequence-aware side-chain packing and downstream fold/design
validation remain later maturity gates.
An output-stage interface must resolve to two distinct output chains; target a
non-identity symmetry neighbour rather than relying on same-chain self
distances.

In this first backend slice every component selector must also appear as a
connection endpoint, so all component atoms have an unambiguous place in the
generated chain topology. Unattached functional components will be enabled
only with explicit multichain/ligand ownership semantics.

Complete examples are
`examples/rfd3_mosaic/public_three_component_graph.yaml` and
`examples/rfd3_mosaic/public_multi_face_component.yaml`. Inspect one with
`rfd3-mosaic plan` first and use `rfd3-mosaic validate` for full symmetry
expansion and static clash/interface preflight.

### Search symmetry neighbours and component poses

When the component and its interface faces are known but the appropriate
finite-group neighbour is not, use the graph search command instead of
manually trying `T:g01`, `T:g02`, and so on:

```bash
python -m rfd3_mosaic.cli search design.yaml \
  --output-dir /path/to/search-output \
  --symmetry C3 \
  --symmetry C4 \
  --symmetry D3 \
  --interface alpha_beta_neighbour \
  --pose-samples 16 \
  --seed-start 1000 \
  --top 20
```

Omit `--symmetry` to search only the symmetry declared in the input design.
Repeat it to compare complete candidates across several Cn, Dn, T, O or I
groups. The chosen symmetry is frozen into each selected YAML; candidates from
different groups are ranked by the same full-assembly feasibility metrics.
This is a discrete architecture search over explicitly requested groups, not
an inference that an observed local transform uniquely determines the native
global symmetry.

Omit `--interface` to search every declared interface jointly. By default the
identity operation is excluded because the command is looking for neighbouring
copies; `--include-identity` enables it for non-self edges. The Cartesian
product is bounded by `--max-candidates` and fails before compilation when the
request is too large.

Each candidate uses only the canonical transform IDs from the same registry
consumed by Assembly IR and RFD3. Every candidate is expanded as a complete
assembly and scored using the production static reports: hard clashes,
required interface relations, generated-link contour reachability, configured
objectives, interface contacts and cavity/link geometry. The search writes:

```text
search-output/
  graph_search.json
  candidates/candidate_XXXXXX/
  selected/rank_XXXX_candidate_XXXXXX.yaml
```

Only statically feasible candidates are eligible for `selected/`. Before a
candidate is exposed there, Mosaic serializes it, reloads it through the
ordinary public loader and strictly recompiles it. The replayed initialized
structure must be identical to the assembly that was ranked. A candidate that
fails this gate remains visible in `graph_search.json` with
`replay_validated: false`, but cannot be submitted as a selected result.

Files below `selected/` are therefore replay-validated ordinary public designs
with concrete neighbour relations and deterministic pose seeds. They can be
inspected, validated and submitted through the normal path:

```bash
python -m rfd3_mosaic.cli validate \
  search-output/selected/rank_0001_candidate_000000.yaml
python -m rfd3_mosaic.cli submit \
  search-output/selected/rank_0001_candidate_000000.yaml
```

This first inverse-search slice compares an explicit user-requested set of
finite symmetries, but every candidate still uses one global full group action.
Discovering component stabilizers, coset assignments and generated topology
remains a later layer; those variables are not approximated by renaming
neighbour transforms.

### Visual fixed-orbit alignment in PyMOL

Do not use whole-structure `align` or `super` to inspect a generated assembly.
Generated residues have no reference counterpart, and the compiled input may
store several motif-fragment chains that RFD3 later merges into one output
chain. PyMOL cannot infer that correspondence reliably.

Load the run's `input/presymmetrized_input.cif` as `ref`, load the result as
`design`, and use the provenance-aware helper:

```pymol
run /path/to/rfd3-mosaic/scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py
mosaic_align_fixed ref, design, /absolute/path/to/run
```

The command reads the compiled fixed selectors, `diffused_index_map` and exact
symmetry registry, reproduces the constraint-orbit audit correspondence, and
creates `design_fixed_aligned`. It does not move or overwrite the original
`design` object. The printed atom count should equal the audit's
`matched_heavy_atoms`; its RMSD should be close to the audit's joint-orbit
RMSD, apart from coordinate-file rounding.

The shortest routine command loads both structures, preserves the raw output,
performs the same alignment and applies a reference/output color scheme:

```pymol
run /path/to/rfd3-mosaic/scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py
mosaic_load_run /absolute/path/to/run
```

This creates `mosaic_ref`, `mosaic_design_raw` and
`mosaic_design_aligned`. Re-run with another object prefix when comparing
several runs in one session:

```pymol
mosaic_load_run /absolute/path/to/second/run, candidate_2
```

To make the commands permanent, add the following line to `~/.pymolrc` on the
machine where PyMOL runs:

```pymol
run /absolute/path/to/rfd3-mosaic/scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py
```

For normal local inspection, drag the run's `presymmetrized_input.cif` and
result CIF into PyMOL. If they are the only two raw objects loaded, run:

```pymol
mosaic_align
```

The helper identifies the reference and result objects, finds the matching run
metadata below `/home/haixi/Documents/template`, and performs the same exact
fixed-orbit alignment. If object or run identification is ambiguous it fails
closed. The explicit fallback is:

```pymol
mosaic_align ref, design, /absolute/path/to/run
```

This is the first graph execution slice. It deliberately uses one global
finite symmetry action and lowers all components into the existing common
`AssemblySpecification`, constraint runtime and audits. Multiple independent
stabilizers and simultaneous vertex-, edge- and face-orbit semantics are not
silently approximated; those remain a later IR extension. Until LRZ unit and
GPU gates pass, `public_assembly_graph` remains `schema_only` in the capability
ledger.

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

For several independent rigid components, key `initial_poses` by the same
explicit `coupling_group` identifiers used by `fixed_xyz`:

```yaml
sampling:
  initial_poses:
    site_alpha:
      radius: {minimum: 70.0, maximum: 70.0}
      radial_direction: [0.94, 0.342, 0.0]
      orientation: {method: uniform_so3}
      seed: 101
    site_beta:
      radius: {minimum: 80.0, maximum: 80.0}
      radial_direction: [0.866, 0.5, 0.0]
      orientation: {method: uniform_so3}
      seed: 202
```

Each component is sampled from its own seed and is therefore independent of
declaration or compiler iteration order. Unknown coupling groups fail during
lowering. `initial_pose` and `initial_poses` are mutually exclusive; the
singular spelling remains the compact backwards-compatible form when the
design contains exactly one rigid component. Radius and axial offset are
sampled uniformly from the declared intervals; `uniform_so3` is Haar-uniform.
A fixed orientation may instead be declared with `method: fixed` and
`rotation_deg: [x, y, z]`.

Omit both pose fields to keep the input pose unchanged. Static initialization
does not move a motif during diffusion; timestep mobility is a separate
control.

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
`experiment_summary.json`. Mobile runs additionally contain
`mobility_trajectory.json`. A failed worker records `status: failed` and the
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
