# Archived RFD3-Mosaic CLI development history

> This document contains historical development commands, site-specific
> validation records and superseded user-interface notes. It is retained for
> provenance only and is not a current user guide. See
> `docs/rfd3_mosaic/USER_CLI.md` for the maintained public reference.

## Install and verify

The supported core can be installed directly from the maintained GitHub
branch. Full commands, checkpoint placement and release verification are in
[`INSTALLATION.md`](INSTALLATION.md). After installation, run:

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

`--profile local` executes synchronously on the current workstation. Existing
`v100`, `p100`, `a100_80g` and `h100` profiles continue to render immutable
Slurm jobs. Both modes use the same compiler, sampler and required result
audits; local execution is not a reduced scientific path.

## Local CPU development while the AI cluster is unavailable

The repository includes an isolated Python 3.12 CPU environment for compiler,
resolver, prevalidation, audit and unit-test work. It does not modify the LRZ
profiles or pretend to replace CUDA scientific validation.

Create or rebuild it once:

```bash
cd /home/haixi/Documents/mosaic
scripts/rfd3_mosaic/setup_local_cpu_dev.sh
```

Activate it in each new terminal:

```bash
cd /home/haixi/Documents/mosaic
source scripts/rfd3_mosaic/activate_local_dev.sh
```

The activation script uses `.venv-local`, adds the editable Mosaic/RFD3 source
trees to `PYTHONPATH`, and normalizes Foundry's boolean environment variables.
This matters on desktop sessions that export values such as `DEBUG=release`,
which `environs` correctly rejects as a boolean.

Run the complete local gate with:

```bash
make local-test
```

The local CPU environment supports:

- all schema/compiler/resolver/strict-replay unit tests;
- `inspect`, `plan`, `resolve`, `validate`, `render` and report/audit logic;
- RFD3 input construction and runtime feature prevalidation;
- deterministic CPU pose search and frozen-YAML reproduction.

It cannot establish GPU memory behavior, CUDA-kernel compatibility, diffusion
speed or scientific interface-packing quality. Those gates remain frozen
A100/H100/V100 canaries when the cluster returns. Checkpoints are intentionally
not downloaded for ordinary CPU development.

The local gate is also large enough for polyhedral ordinary-design replay.
The reference three-supplied-seed T intent now reaches one strictly replayed
candidate and independently validates a 24-chain runtime assembly. Extended
mmCIF chain identifiers are supported during empty-selection construction;
selected candidates undergo real RFD3 feature prevalidation, not adapter
serialization alone.

## Three-day demo quick start

Do not start by choosing a symmetry transform, contact count or radius. First
choose which of the two supported scientific tasks the input represents.

### Task A: preserve an interface that already exists in the input

Use this when the selected fragments already have the relative geometry that
must survive diffusion. Start from:

```bash
cp examples/rfd3_mosaic/simple_interface_seed.yaml my-interface.yaml
```

Keep all fragments that must move as one object in the same
`coupling_group`. The acceptance statement is: *all supplied fixed atoms pass
one joint rigid superposition, their complete symmetry orbit is recovered,
and the generated linker/scaffold passes continuity and clash audits*.

### Task B: keep a motif and create a new interface

Use this when the input motif is fixed internally but the neighbouring
generated regions must create packing that is not already supplied:

```bash
cp examples/rfd3_mosaic/simple_central_motif.yaml my-new-interface.yaml
```

The acceptance statement is stronger than a motif RMSD: *the motif orbit is
recovered, the generated interface passes its all-atom relation audit, the
packing proxy reaches its target, and the requested global pore/shape is
acceptable*. A zero motif RMSD alone is not a passed Task B design.

For either task, use one lifecycle:

```bash
python -m rfd3_mosaic.cli plan my-design.yaml
python -m rfd3_mosaic.cli validate my-design.yaml
python -m rfd3_mosaic.cli submit my-design.yaml

export RFD3_MOSAIC_RUN_ROOT=/absolute/path/to/runs
python -m rfd3_mosaic.cli status JOB_ID
python -m rfd3_mosaic.cli report JOB_ID
```

Use 50 timesteps only as an engineering canary. A presentation or scientific
candidate should be regenerated with 200 timesteps from a newly frozen
submission and must pass all task-specific gates.

### Reproduce the two current engineering records

The 5741271 baseline came from the central-motif/new-interface experiment,
not from 7mwr:

```bash
python -m rfd3_mosaic.cli plan \
  experiments/lrz_public_c3_contig_inferred_interface_v100_50step.yaml
python -m rfd3_mosaic.cli validate \
  experiments/lrz_public_c3_contig_inferred_interface_v100_50step.yaml
```

Its input is
`examples/rfd3_mosaic/inputs/Prism_C3_G2_fixed_motif.pdb`, fixed selector
`A12-20`, and two generated 35-residue termini. Run 5741271 recovered the
motif and C3 symmetry and passed the final heavy-atom interface audit, but its
runtime final packing proxy remained unsatisfied and its central radial
clearance was about 17.9 A. Treat it as a partial Task B baseline.

The 5741324 record is the experimental two-seed resolver path:

```bash
RESOLUTION_ROOT=/absolute/path/to/resolved-two-seed

python -m rfd3_mosaic.cli plan \
  experiments/lrz_simple_two_seed_c3_v100_50step_intent.yaml
python -m rfd3_mosaic.cli resolve \
  experiments/lrz_simple_two_seed_c3_v100_50step_intent.yaml \
  --output-dir "$RESOLUTION_ROOT" --timesteps 50 --top 4

SELECTED=$(find "$RESOLUTION_ROOT/selected" \
  -maxdepth 1 -name 'rank_0001_candidate_*.yaml' -print -quit)
python -m rfd3_mosaic.cli validate "$SELECTED"
python -m rfd3_mosaic.cli submit "$SELECTED"
```

Its source is
`examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb`; the two selected
patch pairs are documented in `CURRENT_PRODUCT_STATUS.md`. Generalized
post-hoc auditing now recovers all 273 fixed heavy atoms and all 6 supplied
interface instances across `C3 x 2 ASU chains`. The same result still has six
real CA clashes, however, so it remains a compiler/runtime canary rather than
a fully accepted cage.

To inspect a frozen run without guessing its source:

```bash
RUN_DIR=/absolute/path/to/JOB_ID
sed -n '1,180p' "$RUN_DIR/resolved_config.yaml"
python -m rfd3_mosaic.cli status "$RUN_DIR"
python -m rfd3_mosaic.cli report "$RUN_DIR"
```

## Start here: the normal user path

You do not need to understand Assembly IR, ports, transform IDs, contact
counts or sampler internals for the two normal motif-scaffolding tasks.

Choose one template:

- `examples/rfd3_mosaic/simple_interface_seed.yaml`: the supplied fragments
  already form the interface; Mosaic preserves their complete joint geometry
  and generates the connecting protein;
- `examples/rfd3_mosaic/simple_central_motif.yaml`: one fixed motif is in the
  protomer; Mosaic grows both termini and guides the generated neighbouring
  regions toward a new symmetric interface.

Copy the chosen YAML and normally edit only:

```yaml
name: my-design
input: /absolute/path/to/input.pdb
symmetry: C3

# Edit selectors and generated lengths for the input.
generation: [...]
constraints: [...]

sampling:
  timesteps: 200
  seed: 42

resources:
  profile: v100

output:
  root: /absolute/path/to/runs
  campaign: my-campaign
```

Then run exactly this lifecycle:

```bash
python -m rfd3_mosaic.cli plan my-design.yaml
python -m rfd3_mosaic.cli validate my-design.yaml
python -m rfd3_mosaic.cli run my-design.yaml

export RFD3_MOSAIC_RUN_ROOT=/absolute/path/to/runs
python -m rfd3_mosaic.cli status JOB_ID
python -m rfd3_mosaic.cli report JOB_ID
```

Use 50 timesteps for a fast pipeline canary and 200 for a real generation
campaign. A produced CIF is not automatically a successful design: keep only
runs reported as `PASSED`, then inspect/rank their interfaces. The expert
assembly graph is optional and should be used only when one component has
several named faces or several different symmetry-neighbour interfaces.

## Input-driven ordinary cage mode

For a new multi-interface cage, begin with either one PDB/mmCIF containing all
supplied interface seeds or a seed library whose entries point to separate
PDB/mmCIF files. Mosaic detects or validates the supplied interface patches and
writes an editable short intent instead of requiring ports, transform IDs,
radius or contact-count tuning:

```bash
rfd3-mosaic inspect input_interfaces.pdb \
  --output-dir cage_intent \
  --architecture cage \
  --subunits-min 12 --subunits-max 60 \
  --diameter-min 80 --diameter-max 160

rfd3-mosaic plan cage_intent/simple_design.yaml
rfd3-mosaic validate cage_intent/simple_design.yaml
```

That command records a broad cage request. Diameter and optional cavity ranges
are retained in every resolved expert YAML, participate in complete-assembly
CPU pose ranking, and are checked again against the final RFD3 output
morphology. General automatic choice of every cage architecture is still a
bounded candidate search, not an unconstrained promise to find a solution.

The first executable ordinary resolver is now available for one supplied
binary, `preserve_exact` interface in a Cn ring:

```bash
rfd3-mosaic inspect input_interfaces.pdb \
  --output-dir ring_intent \
  --architecture ring --composition auto --symmetry C3

rfd3-mosaic resolve ring_intent/simple_design.yaml \
  --output-dir resolved_ring --timesteps 50 \
  --symmetry C3

rfd3-mosaic validate \
  resolved_ring/selected/rank_0001_candidate_000000.yaml
rfd3-mosaic run \
  resolved_ring/selected/rank_0001_candidate_000000.yaml
```

`preserve_exact` is not a tuning preset. It means the complete supplied
interface (two or more participants, and any number of helices/fragments on
each participant) is one natural rigid hyperedge. Mosaic may solve the
translation and rotation of that complete seed relative to other seeds, but
never pulls its participants apart or repacks the supplied interface itself.
Ordinary mode intentionally has no deformable supplied-interface option.

For ordinary use the same strict workflow is also available as one command:

```bash
rfd3-mosaic run ring_intent/simple_design.yaml \
  --resolve-timesteps 50 --profile v100 --dry-run
```

Mosaic writes the complete resolver manifest below
`output.root/.rfd3-mosaic/resolutions/`, requires one strictly replayed
candidate, and only then re-enters the normal public-design `run` path.
Remove `--dry-run` to submit. Use `--resolution-dir PATH` when the resolver
artifacts should live elsewhere. If no candidate survives strict replay, the
command stops before rendering or submission and points to the manifest;
expert users can then run `resolve` with explicit search controls.

The YAML can remain portable across machines. Override only the site-specific
run location when needed:

```bash
rfd3-mosaic run simple_design.yaml \
  --run-root /scratch/$USER/rfd3-mosaic-runs \
  --campaign cage-screen --profile v100 --dry-run
```

`--run-root` changes the internal experiment envelope, provenance and run
index destination; `--output-dir` is different and controls only where the
rendered submission files are written. The resolved public YAML remains an
unaltered scientific record of the selected candidate.

`resolve` never silently executes rank 1. It retains both polymer-chain
directions and both adjacent-copy directions (deduplicated for C2), compiles
and ranks every candidate, reloads the written public YAML, and requires the
strict replay structure hash to equal the structure that was ranked. The
multi-seed bridge additionally requires native RFD3-adapter replay and the
expanded topology contract before a YAML appears under `selected/`. Use
`--timesteps 50` for a canary; omit it for the 200-step production default.
The selected YAML then follows the same expert `UserDesignSpec ->
AssemblySpecification -> Mosaic-RFD3 -> audit` path.

### Supplied multi-seed layout and pose resolution

The user supplies every natural interface seed and its physical usage. Mosaic
may decide polymer connectivity, finite-group relations and relative rigid
poses; it never creates a new seed identity or changes the internal geometry
of a supplied interface. Choose the file-frame policy explicitly:

An interface seed is one complete physical face, not a collection of
independent sides. It may contain two, three or more participants. Mosaic
stores all participants in one `joint_rigid` component and preserves their
mutual pose and packing exactly. A generated scaffold link may reference a
participant terminus, but that participant is never detached, independently
moved, re-paired or promoted to a new interface seed.

Resolver-emitted expert YAML retains that identity directly. A cooperative
site is one declaration, not several user-visible pairwise interfaces:

```yaml
interfaces:
  - id: cooperative_site
    between: [port_A, port_C, port_D]
    contact_pairs: [[port_A, port_C], [port_C, port_D]]
    relation: {mode: preserve_input}
```

`contact_pairs` is a connected contact-supported execution tree inferred by
inspection/resolution. It exists only to feed the current binary RFD3 feature
tensors; usage, provenance, symmetry multiplicity and final audit verdict are
computed once for `cooperative_site` as a whole. Ordinary users provide the
complete interface seed and `use`, not this tree.

```yaml
# Default: solve multiple independent source files; preserve one shared file.
seed_layout: auto

# Ignore relative placement between seeds and solve it, even in one file.
seed_layout: solve

# Treat one shared input pose as authoritative.
seed_layout: preserve_input
```

### One supplied interface between different-valency oligomers

`preserve_input` also supports the first mixed-stabilizer ordinary workflow.
The input contains the complete two sides of one natural interface; each side
may contain several protomer chains. The chain count is the component
valency, while `use.exact` is the number of physical interface instances in
the final assembly:

```yaml
schema_version: 1
mode: simple
name: tetrahedral-c2-c3-interface
input: my_prepositioned_c2_c3_interface.pdb

goal:
  architecture: cage
  composition: auto
  symmetry: [T]

seed_layout: preserve_input

interface_seeds:
  natural_interface:
    participants: [c2, c3]
    selectors:
      c2: A/12-20/*,A/28-36/*,B/12-20/*,B/28-36/*
      c3: C/12-20/*,C/28-36/*,D/12-20/*,D/28-36/*,E/12-20/*,E/28-36/*
    use: {exact: 12}
    geometry: preserve_exact

generation:
  length: {minimum: 8, maximum: 30}
```

For this contract Mosaic derives `6 x C2`, `4 x C3` and `12` physical
interfaces from the tetrahedral stabilizer/coset incidence. It preserves the
user's interface geometry and never invents another interface type. A selected
candidate must pass standalone hash replay, native RFD3 adapter construction
and runtime feature prevalidation. The `resolution_manifest.json` records the
component multiplicities and exact coset mapping.

The identical ordinary contract supports a pre-positioned O/C2--C4 interface
with `use: {exact: 24}`. Mosaic derives `12 x C2` and `6 x C4`, compacts fixed
fragments belonging to the same generated polymer path into one input chain,
and strictly prevalidates the resulting 48-chain RFD3 design.

A pre-positioned I/C2--C5 request with `use: {exact: 60}` is also executable
through strict CPU replay. Mosaic derives `30 x C2` and `12 x C5`, writes seven
representative parser chains with explicit per-entity cosets, and RFD3 expands
them to all 120 physical chains. Reports distinguish
`compact_parser_chain_count: 7` from `compact_physical_chain_count: 120`.
Compact input avoids the parser identifier limit; it does not reduce runtime
memory and is not yet a claim of GPU or scientific I-cage validation.

This input must already place the two oligomers in their meaningful relative
interface pose. Independently supplied C2 and C3 files whose relative pose is
unknown remain fail-closed because their SE(3) optimization must preserve two
different stabilizer frames simultaneously.

Use `solve` when the seed structures are individually correct but their
mutual placement is arbitrary. Use `preserve_input` when the whole input is an
already meaningful assembly. `preserve_input` rejects seeds from unrelated
source files. `solve` requires at least two supplied interfaces; it moves each
complete interface as a rigid hyperedge, never its participants independently.

During `solve`, Mosaic polls complete-seed poses both individually and in
connection-graph blocks. Two user-connected seeds may therefore translate or
rotate together (same or opposite direction) without temporarily breaking a
linker or interface hard contract; an unrelated seed is not dragged along.
Every accepted block move is recompiled on the complete expanded assembly and
is written back as an exact replayable pose.

The current executable full-orbit contract requires:

- at least two user-supplied `geometry: preserve_exact` seeds;
- one or more ordered, non-overlapping selectors per participant; several
  helices from one source chain remain one rigid interface participant and
  only the intervening same-chain gaps are generated;
- complete `N/CA/C` backbone anchors at both selector boundaries;
- `goal.composition: auto` or `homomer` unless heteromeric component ownership
  is made unambiguous by at least two supplied seeds and explicit
  `polymer_connections`; Mosaic does not invent a heteromer assignment;
- no stabilizer/coset action on an unknown-pose component; those components
  require stabilizer-aware local frames and therefore fail closed;
- a positive diameter and/or cavity range when size is constrained; both are
  hard final-output contracts, while `preferences.cavity` remains a softer
  compact/auto/open initialization bias.

For example, two seeds in one file whose mutual pose must be solved are:

```yaml
goal:
  architecture: ring
  composition: auto
  symmetry: [C3]

seed_layout: solve

interface_seeds:
  interface_alpha:
    participants: [A, B]
    selectors: {A: A/12-20/*, B: B/26-37/*}
    geometry: preserve_exact
    use: auto
  interface_beta:
    participants: [C, D]
    selectors: {C: C/8-17/*, D: D/30-41/*}
    geometry: preserve_exact
    use: auto

# Optional but authoritative. When present, Mosaic does not enumerate a
# different covalent pairing. Compact endpoints are interface.participant;
# `from` is the C-terminal fragment and `to` is the N-terminal fragment.
polymer_connections:
  - id: unit_one
    from: interface_alpha.A
    to: interface_beta.D
    # Optional authoritative group action. If omitted, Mosaic searches only
    # the missing relations required to close the requested cage.
    copy_relation: {transform: "T:g01"}
  - id: unit_two
    from: interface_beta.C
    to: interface_alpha.B
```

`interface_seeds` define non-covalent interface types; they are not consumed
by one connection. Their `use` value counts physical interface instances in
the complete symmetry-expanded assembly. `polymer_connections` instead bind
the N/C connectivity of the ASU fragments. A specific fragment can have at
most one incoming N-side and one outgoing C-side connection. Omitting
`polymer_connections` retains the legacy bounded topology enumeration;
providing it makes the declared participant pairing and direction
authoritative. An optional connection `copy_relation` also freezes a known
neighbour/group action. Mosaic never changes such a declaration; it searches
only unassigned connections and deterministically adds the smallest set of
relations required to generate the requested finite group. The declared and
completed relation counts are recorded in candidate metadata.

The same syntax also accepts a separate `source:` on each seed. In `auto`
mode, distinct source files automatically enter the same solve path.

Resolve, inspect the ranking and choose one emitted standard YAML explicitly:

```bash
rfd3-mosaic plan two_seed_ring.yaml
rfd3-mosaic resolve two_seed_ring.yaml \
  --output-dir resolved_two_seed \
  --pose-samples 8 --pose-optimize-top 4 --timesteps 50
rfd3-mosaic validate \
  resolved_two_seed/selected/rank_0001_candidate_000000.yaml
```

The resolver executes this auditable sequence:

```text
validate each supplied interface contact graph and requested multiplicity
-> canonicalize complete seeds when seed_layout resolves relative pose
-> enumerate scaffold paths by referencing participant termini while keeping
   every supplied interface hyperedge indivisible
-> assign finite-group relations and validate the expanded interface/unit graph
-> generate deterministic global Cn/Dn/T/O/I full-orbit pose starts
-> jointly optimize radius, azimuth, axial offset and rotation
-> reject required-interface, linker-contour, clash, closure or required-
   objective failures; prefer clear straight linker routes without treating
   the endpoint chord as the generated backbone path
-> write a normal public UserDesignSpec YAML
-> reload it and require exact structure-hash and RFD3-adapter replay
```

A supplied interface face is not limited to one contiguous helix. Use one
comma-separated selector per participant when several fixed fragments on the
same source chain jointly form that face:

```yaml
interface_seeds:
  interface_alpha:
    participants: [A, B]
    selectors:
      A: A/12-24/*,A/48-61/*
      B: B/105-118/*,B/137-151/*
    geometry: preserve_exact
    use: {exact: 12}
```

All four fragments and their relative interface geometry remain fixed as one
rigid seed. Mosaic preserves their sequence order, generates the two internal
gaps, and connects only the outer participant termini to other supplied
seeds. A participant spanning several unrelated source chains requires
expert component paths because ordinary mode will not invent covalent bonds.

`--pose-samples` controls deterministic global starts, not diffusion output
count. `--pose-optimize-top` limits how many initially ranked topology/pose
candidates receive coarse-to-fine continuous refinement. Candidates outside
that shortlist are explained, not silently promoted. Diversity preferences
may increase the effective number of global starts.

Path-cover output alone remains topology evidence with `executable: false`.
Only a candidate that binds real termini and finite-group relations, passes
the complete CPU geometry evaluation, freezes explicit scalar poses and
strictly replays can appear under `selected/`. The resolver never silently
runs rank 1.

### AI-cluster verification and promotion process

Development snapshots must be verified from the same frozen source tree that
will be rendered into the GPU job. On the AI login node, first run the focused
resolver regressions and then the complete unit suite:

```bash
cd /path/to/rfd3-mosaic
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"

for test_file in \
  test_seed_library.py \
  test_pose_optimizer.py \
  test_multi_seed_resolver.py \
  test_simple_resolver.py; do
  python -m unittest discover \
    -s tests/rfd3_mosaic/unit -p "$test_file" -v || exit 1
done

python -m unittest discover \
  -s tests/rfd3_mosaic/unit -p 'test_*.py' -v
```

After CPU tests pass, resolve and validate one real supplied-seed intent:

```bash
python -m rfd3_mosaic.cli resolve ordinary_intent.yaml \
  --output-dir resolved_candidates \
  --pose-samples 8 --pose-optimize-top 4 --timesteps 50

SELECTED=$(find resolved_candidates/selected \
  -maxdepth 1 -name 'rank_0001_*.yaml' -print -quit)
python -m rfd3_mosaic.cli validate "$SELECTED"
python -m rfd3_mosaic.cli submit "$SELECTED" --profile p100
```

A candidate is promoted only after all of these are true:

1. focused and full CPU suites pass from the frozen snapshot;
2. resolver metadata lists exactly the user-supplied interface IDs and
   `invented_interface_count: 0`;
3. the selected YAML passes strict structure-hash and RFD3-adapter replay;
4. a 50-step GPU canary passes fixed-seed, symmetry, interface-relation,
   continuity, clash and scaffold audits;
5. the result is visually inspected and its interface evidence is recorded;
6. a 200-step run and an independent seed/input reproduce the behavior before
   the capability is called scientifically validated.

`resolve` success alone means only that a replayable candidate was compiled.
A produced CIF alone also does not mean success. The required audits and the
scientific interface evidence are the acceptance boundary.

The generated YAML has this ordinary-user shape:

```yaml
kind: simple_cage_intent
name: my-cage
input: /path/to/input_interfaces.pdb
seed_layout: solve

goal:
  architecture: cage
  composition: auto
  symmetry: auto

interface_seeds:
  interface_A_B:
    participants: [A, B]
    selectors:
      A: A/12-20/*
      B: B/26-37/*
    use: auto
    geometry: preserve_exact

  interface_C_D:
    participants: [C, D]
    selectors:
      C: C/8-17/*
      D: D/30-41/*
    use: {minimum: 4, maximum: 12}
    geometry: preserve_exact

generation:
  length: {minimum: 40, maximum: 100}

resources:
  profile: v100
```

`use` is the requested number of physical occurrences of one interface
identity in the final assembly. Accepted forms are `auto`, an integer such as
`12`, `{exact: 12}`, or a minimum/maximum range. It does not mean “copy this
chain 12 times”: after symmetry expansion the compiler counts unique physical
interface instances and rejects incompatible expert architectures.

`participants` is not limited to two chains. A cooperative interface may use
`participants: [A, C, D]` (or more) with one selector for every participant.
Validation requires the participant contact graph to be connected; it does
not require every participant pair to contact every other pair. The complete
connected contact graph is one indivisible interface identity. Automatic
inspection initially emits pairwise candidates because those can be detected
without guessing chemical intent; users may merge them into one cooperative
multi-participant interface.

The inspection thresholds are frozen in the generated YAML so validation
replays the same input analysis. Users may delete false-positive interfaces,
change `use`, and restrict the scientific goal. They do not need to specify
radius, angle or symmetry-neighbour transform. For supplied-interface
workflows Mosaic does not design a new contact at all; it only preserves the
supplied interface while generating declared scaffold. The separate
`create_symmetric_interface` task is used when the input is a central motif
with no desired interface and generated regions must form one.

If the same two input chains touch at two residue-disconnected surfaces,
`inspect` emits separate IDs such as `interface_A_B_patch_001` and
`interface_A_B_patch_002`; it no longer merges them into one fictitious large
interface. The report also lists, for every observed chain, the detected port
count and interface IDs. At this stage a chain is only an observed input
component: Mosaic still must decide whether several seed fragments are joined
into one final polymer unit.

`plan` also performs a conservative generic full-orbit compatibility pass.
For example, `architecture: cage` plus `use: 12` retains `D6` and `T` as
generic candidates and rejects incompatible group orders. This is only the
discrete first filter: the report keeps polymer-unit ownership, connection
order, neighbour relations and continuous pose explicitly unresolved.

This ordinary intent path supports `inspect`, `plan`, `validate`, `resolve`
and one-command `render`/`run`. One-command execution does not bypass
selection: it persists the resolution, requires exactly the resolver's top
strictly replayed public YAML, and dispatches that YAML through the shared
execution path. Full-orbit unknown-pose initialization exists for Cn/Dn/T/O/I.
The reference three-seed T intent now passes strict replay and native RFD3
prevalidation; Dn/O/I remain family-specific CPU/GPU gates. Stabilizer-aware
unknown-pose placement, automatic composition equivalence and native variadic
sampler tensors remain blocked. Public
multi-participant supplied hyperedges already lower through a connected
binary compatibility tree while retaining one interface identity, usage and
audit group.
This fail-closed boundary prevents Mosaic from assuming that contact partners
are sequence-adjacent or symmetry-equivalent.

For several supplied seeds, bare deterministic path-cover records remain
topology-only evidence and `executable: false`. The full resolver can convert
them into runnable standard YAML only after backbone anchors, supplied contact
geometry, finite-group relations, expanded topology, complete-assembly pose,
compiler and strict replay checks succeed. Unknown-relative Cn examples have
already passed this CPU boundary. Dn/T/O/I starts are implemented but remain
family-specific validation gates; a failed family is never silently replaced
by C3 or advertised as executable.

The two-seed C3 and three-seed T `resolve -> validate` CPU replay gates have
passed. Promotion
beyond research-alpha scope still requires newly rendered 50-step GPU results
passing every required supplied-seed, symmetry, continuity, clash and scaffold
audit, plus one polyhedral multi-seed strict replay. Capability reporting keeps
those GPU/scientific gates separate from the already working CPU frontend.

`plan --format json` records this boundary explicitly as
`resolution_stage: intent` and `executable: false`, together with the
variables the resolver must still freeze. A successful intent validation
therefore means that the input, seed contacts and scientific request are
self-consistent; it is not a claim that a runnable cage architecture has
already been found.

Experts may bypass discovery by writing the explicit
`components / ports / interfaces / connections` graph. Both authoring modes
converge on the same compiler, sampler and audits; the ordinary mode is not a
second RFD3 implementation.

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

If inference produced a structure but an older audit implementation crashed,
rerun the complete applicable audit set without spending another GPU run:

```bash
rfd3-mosaic audit /path/to/run
rfd3-mosaic audit 5741324 --root /path/to/runs
```

`audit` reads the frozen resolved configuration and compiled RFD3 input,
overwrites the corresponding post-inference reports, reapplies the same
fail-closed gate used by the live worker, refreshes `experiment_summary.json`
and the HTML/JSON report, and records `inference_rerun: false`. It refuses to
guess when required frozen artifacts are absent.

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

## Expert topology-neutral design reference

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
  about the declared Cn/Dn principal symmetry axis. The unselected
  coordinates remain diffusion degrees of freedom; this is not an alias for
  Cartesian fixing or rigid-component mobility;
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

Interface generation and fixed-component mobility are separate decisions.
For the ordinary interface-design task, the default is:

```yaml
task: create_symmetric_interface
fixed_arrangement: locked
```

`locked` preserves the complete supplied arrangement: every fixed fragment,
all symmetry copies, and all inter-fragment distances and angles retain one
joint pose. Packing guidance still acts on the diffusion-generated residues,
so new scaffold/interface material can pack around an immovable C3, C4 or
polyhedral seed. If the input already defines a non-overlapping assembly pose,
Mosaic retains it exactly. If a single-ASU motif lies on the symmetry
stabilizer and therefore has no usable assembly radius yet, the compiler
chooses one deterministic clash-free initial pose and then locks that pose for
the whole diffusion trajectory.

If each supplied interface must remain internally exact but different rigid
components may translate or rotate relative to one another, request that
different physical problem explicitly:

```yaml
task: create_symmetric_interface
fixed_arrangement: optimize_components
```

This enables joint packing-aware component-pose optimization. It changes only
whole-component SE(3) poses; the selected atoms are never deformed. An
explicit `sampling.initial_pose` is also honored in either mode and defines
the frozen starting arrangement deliberately.

For expert declarations without a task preset, the component pose is fixed by
default. To let independently declared components adapt to the generated
scaffold while preserving every internal distance, set bounded rigid mobility
explicitly:

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
path has passed C3 and D3 multi-orbit GPU canaries and remains an engineering
interface while broader symmetry/seed campaigns accumulate. Keeping at least
one component fixed is still useful when a
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

`tilt_only` is the complementary rotation-only expert subspace. It permits
rotation about the two axes perpendicular to the declared symmetry axis,
with exactly zero translation and zero axial twist. `radial_rotation` and
`radial_axial_rotation` combine axis-aware translation with bounded rotation.
All of these use the same joint proposal, exact-symmetry projection,
accept-or-rollback transaction and mobility audit.

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
backend. `cylindrical` is executable for complete Cn/Dn orbits; compact
stabilizer/quotient ASUs and T/O/I cylindrical axes remain fail-closed.
Constrained motif fragments that are not linker endpoints are retained as
independent ASU polymer paths, with no invented covalent connection.
Partial-atom fixed XYZ and unconstrained generated endpoints still fail with
a direct backend/lowering error. These cases are not silently converted to
historical adapter behavior. Every cylindrical run requires
`cylindrical_coordinate_audit.json` in addition to the universal scaffold
audit.

A repository-owned, directly validatable example is available at
`examples/rfd3_mosaic/public_c3_cylindrical_design.yaml`. It deliberately
omits `output` so validation and planning do not inherit a machine-specific
run root:

```bash
python -m rfd3_mosaic.cli validate \
  examples/rfd3_mosaic/public_c3_cylindrical_design.yaml
python -m rfd3_mosaic.cli plan \
  examples/rfd3_mosaic/public_c3_cylindrical_design.yaml
```

To submit it, copy the file and add the normal site-specific `output.root`
and select the desired `resources.profile`. Cn/Dn uses the declared principal
symmetry axis; the compiler rejects polyhedral and compact quotient cases
instead of silently applying the wrong coordinate system.

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

## Pairwise interleaved seed topology: a supported special case

Here “more than two seeds” means **several non-covalent interface pairs already
represented in the single file named by `input:`**. It does not mean one PDB
per seed and it does not mean that every fixed helix belongs consecutively to
one protein chain.

Keep the two edge types explicit:

```text
interface pairs:  A1 <-> B1,  A2 <-> B2,  A3 <-> B3
protein units:    A1  -- B2,  A2  -- B3,  A3  -- B1
```

The original Ho-Yeung implementation uses this same interleaving: after
cyclic expansion, `(A,B)`, `(C,D)`, `(E,F)` remain interface pairs, while the
generated contigs connect halves of neighbouring pairs. Mosaic must preserve
both graphs independently.

The general Mosaic model is broader. One interface relation may contain three
or more participants, and one component may carry ports belonging to several
different relations:

```text
interface I1 participants = [P11, P12]
interface I2 participants = [P21, P22, P23]
component U1 ports = [P11, P21, P23]
```

The ordinary cage-intent schema and input-contact validator already accept
such variadic participant lists and require their contact graph to be
connected. The executable expert schema now preserves one variadic public
relation; the compiler lowers a contact-supported tree into current binary
Assembly IR/RFD3 members, counts physical usage once, and groups audit results
back into the original hyperedge. Native variadic sampler tensors remain a
later optimization rather than a semantic blocker.
Do not model a cooperative three-participant site as three independent passed
pairwise interfaces.

A single cooperative interface can be resolved without inventing topology
when each participant selector itself contains an explicit ordered path on one
source chain:

```yaml
interface_seeds:
  cooperative_site:
    participants: [A, B, C]
    selectors:
      A: A/24-28/*,A/144-150/*
      B: B/175-179/*,B/295-301/*
      C: C/326-330/*,C/446-452/*
    use: {exact: 1}
    geometry: preserve_exact
```

Mosaic connects only the first A fragment to the second A fragment, and the
equivalent B and C paths. It does not create A--B, B--C or C--A peptide
links. If a participant has only one isolated fragment, the ordinary
resolver refuses to guess the polymer graph and requests another supplied
interface seed or expert `connections`.

When that supplied oligomer is stabilized by the complete requested group
(for example the PI25 trimer under C3), the quotient contains one physical
component but still has three symmetry-related polymer paths. Mosaic freezes
the participant-to-transform assignment in the resolved YAML and emits a
preexpanded RFD3 ASU: A/B/C are annotated as `C3:e`, `C3:r1` and `C3:r2`
within one symmetry entity. They are not expanded into nine chains, and the
one-coset representation is not treated as an unconstrained asymmetric run.
This path currently requires one continuous chain per stabilizer transform;
more general internally branched stabilized entities fail closed.

Complex rings and cages are declared as a graph, not as `left`, `right`,
`third`, and progressively more topology-specific fields:

- `components` are rigid seed or motif nodes;
- `ports` are named interface faces owned by those rigid nodes;
- `interfaces` are geometric relationship edges between ports (or, for
  backward compatibility, directly between components when no ports exist);
- `connections` are directed peptide regions that RFD3 must generate;
- the declared `symmetry` expands every node and edge through one exact group
  action.

There is no two-pair schema limit. The current expert spelling for three pairs
is deliberately explicit:

```yaml
schema_version: 1
name: three-interface-pair-c3
input: three_interface_pairs.pdb
symmetry: C3

components:
  seed_1:
    selectors: [A12-20, B26-37]
    geometry: joint_rigid
  seed_2:
    selectors: [C12-20, D26-37]
    geometry: joint_rigid
  seed_3:
    selectors: [E12-20, F26-37]
    geometry: joint_rigid

ports:
  seed_1_a: {component: seed_1, selectors: [A12-20]}
  seed_1_b: {component: seed_1, selectors: [B26-37]}
  seed_2_a: {component: seed_2, selectors: [C12-20]}
  seed_2_b: {component: seed_2, selectors: [D26-37]}
  seed_3_a: {component: seed_3, selectors: [E12-20]}
  seed_3_b: {component: seed_3, selectors: [F26-37]}

interfaces:
  - id: supplied_pair_1
    between: [seed_1_a, seed_1_b]
    relation: {mode: preserve_input}
  - id: supplied_pair_2
    between: [seed_2_a, seed_2_b]
    relation: {mode: preserve_input}
  - id: supplied_pair_3
    between: [seed_3_a, seed_3_b]
    relation: {mode: preserve_input}

connections:
  - id: unit_1
    from: {component: seed_1, selector: A12-20, terminus: c}
    to: {component: seed_2, selector: D26-37, terminus: n}
    length: {minimum: 25, maximum: 40}
  - id: unit_2
    from: {component: seed_2, selector: C12-20, terminus: c}
    to: {component: seed_3, selector: F26-37, terminus: n}
    length: {minimum: 25, maximum: 40}
  - id: unit_3
    from: {component: seed_3, selector: E12-20, terminus: c}
    to: {component: seed_1, selector: B26-37, terminus: n}
    length: {minimum: 25, maximum: 40}
```

This shows topology semantics, not yet a claim that a completely pre-expanded
three-pair input has passed the GPU gate. The compiler still needs an explicit
copy/orbit ownership pass before it can safely decide whether these three
pairs are independent interface classes or already materialized symmetry
copies.

Separately, if one protein unit genuinely contains three or more sequential
fixed fragments, continuous connections may form an ordered path such as
`fragment_a -> fragment_b -> fragment_c`. The adapter emits the internal
fragment once. That optional feature must not be confused with the interleaved
interface-pair topology above.

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

The current design-interface controller is a bounded CA-level packing field,
not a learned interface-quality oracle. It is symmetry-neighbour aware and
avoids the legacy all-to-all radial collapse. Its joint energy combines
nearest-pair attraction, balanced residue coverage on both interface sides,
contiguous-patch formation, residue-normalized clash repulsion, an optional
COM-distance target, contact-patch orientation, contact-depth uniformity,
local CA-spacing protection and smooth worst-interface pressure. Symmetry
copies are averaged within each declared edge before source interfaces are
combined, preventing orbit multiplicity from acting as an accidental weight.
Same-chain adjacent token updates are smoothed before the bounded step.
Runtime diagnostics expose every term separately. These are differentiable
backbone proxies; true all-atom shape complementarity, solvent burial,
side-chain packing and downstream fold/design validation remain later
maturity gates.
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

A linker range authorizes Mosaic to choose an executable exact length. Before
strict replay, Mosaic measures every symmetry-expanded instance and freezes
the smallest deterministic choice that is at least the common-range midpoint
and long enough for the worst 3.8 A contour requirement. Connections sharing
one `tie_group` are solved against the intersection of their ranges and receive
one common exact length. Mosaic changes neither connection endpoints nor
interface identities; it rejects only when no user-authorized common length
exists. The frozen YAML, static ranking and native RFD3 contig therefore use
the same lengths.

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

For routine inspection, only two files are required:

- `input/presymmetrized_input.cif`
- the generated `*model_0.cif`

Drag both CIFs into PyMOL. If they are the only two raw structure objects,
type one command:

```pymol
mosaic_align
```

No JSON file, run directory or fixed object name is required. The helper
identifies which CIF is the motif reference, locates each reference fragment
inside the generated chains by residue sequence, and resolves symmetry-copy
ambiguity by selecting the largest set with one consistent three-dimensional
transform. It fails with an explicit mismatch error instead of silently
aligning unrelated residues.

The command creates `mosaic_aligned`, leaves both input objects unchanged,
colors the reference cyan and aligned design magenta, and disables the raw
design. Re-running the command replaces only `mosaic_aligned`. The printed
residue coverage should include the complete reference for an ordinary fixed
or jointly rigid motif orbit.

To make the commands permanent, add the following line to `~/.pymolrc` on the
machine where PyMOL runs:

```pymol
run /absolute/path/to/rfd3-mosaic/scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py
```

When several unrelated structures are already loaded, the optional explicit
form is:

```pymol
mosaic_align reference_object, design_object
```

For audit reproduction, the metadata-backed interface remains available. It
reads the compiled selectors, `diffused_index_map` and exact symmetry registry
and should match the audit's joint-orbit RMSD apart from coordinate rounding:

```pymol
mosaic_align ref, design, /absolute/path/to/run
```

The routine two-CIF workflow is recommended for visual inspection. Use the
metadata-backed form only when the exact compiled residue provenance is itself
part of the analysis.

The original graph execution slice used one global finite symmetry action.
The CPU compiler now additionally supports the first geometry-bound mixed
component action: one pre-positioned C2--C3 supplied interface under T lowers
to separate six-copy and four-copy component coset orbits and one twelve-edge
physical interface orbit. Multiple independently positioned stabilizer
components, simultaneous arbitrary vertex/edge/face orbit combinations and
dynamic mixed-orbit mobility are still not silently approximated. The static
public assembly graph remains `gpu_canary` following the audited T run
5735772; the mixed C2--C3 slice is CPU strict-replay validated and has not yet
earned a separate GPU/scientific claim.

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
queued therefore cannot change the code executed by that job. Public runs also
store a normalized `public_user_design.yaml` beside `resolved_config.yaml`, so
later edits to the authoring YAML or execution profile do not invalidate or
change an already rendered job. Structure inputs remain external files with
separate fail-closed hashes; changing an input PDB/mmCIF after render is still
rejected rather than silently changing the queued design.

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
