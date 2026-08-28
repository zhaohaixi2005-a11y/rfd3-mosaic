# RFD3-Mosaic user workflow guide

This guide is the complete ordinary-user path from an input structure to a
reviewable set of RFdiffusion3 backbones. It distinguishes required scientific
input from optional controls and provides copy-ready examples for the two
supported design tasks.

RFD3-Mosaic is a command-line research application. Users declare the
assembly they intend to build; Mosaic compiles that declaration into explicit,
replayable RFdiffusion3 inputs. It does not require users to author RFD3 JSON,
symmetry matrices, Slurm scripts or one YAML file per generated structure.

## The five-command workflow

After installation and one-time checkpoint/profile setup, an ordinary design
uses this lifecycle:

```bash
rfd3-mosaic init design.yaml [TASK OPTIONS]
rfd3-mosaic plan design.yaml
rfd3-mosaic validate design.yaml
rfd3-mosaic run design.yaml --profile PROFILE.yaml
rfd3-mosaic report RUN_ID_OR_DIRECTORY
```

`init` writes a complete YAML. `plan` is read-only. `validate` compiles and
preflights the RFD3 input without launching diffusion. `run` performs
generation and audits. `report` creates HTML and JSON summaries.

Run these once after installing or moving to another machine:

```bash
rfd3-mosaic doctor --profile PROFILE.yaml
rfd3-mosaic capabilities
```

## Choose the scientific task first

| Question | Task | What remains exact | What RFD3 generates |
| --- | --- | --- | --- |
| I have a functional motif but need a new symmetric interface | `central-motif` | selected motif geometry | scaffold plus a new oligomerization interface |
| I already have an interface and need to scaffold it | `supplied-interface` | both interface partners as one joint-rigid object | declared linker or terminal scaffold |
| I have a supplied multi-fragment complex and want to extend or reassemble it | simple `supplied-interface`, or the general assembly graph | every declared rigid or joint-rigid component | user-declared polymer paths and optional additional interfaces |

Do not use `central-motif` when the input already contains the biological
interface that must be preserved. Do not enable `--new-oligomer-interface`
unless a second, generated interface is scientifically intended.

## What the user must provide

The initializer requires only the scientific information Mosaic cannot infer
without changing the task.

| Input | Required | Example | Meaning |
| --- | --- | --- | --- |
| output YAML | yes | `design.yaml` | one experiment declaration, not one structure |
| task | yes | `central-motif` | which of the two physical problems to compile |
| input structure | yes | `seed.cif` | PDB or mmCIF containing the supplied geometry |
| symmetry | yes | `C3` | user-declared target: Cn, Dn, T, O or I where supported |
| motif selector | central-motif only | `A10-25` | fixed functional geometry |
| interface side A | supplied-interface only | `A20-35` | first partner of the supplied interface |
| interface side B | supplied-interface only | `B40-55` | second partner of the supplied interface |
| generation length | defaulted; scientific review required | `30` or `70-100` | number of residues RFD3 may generate |
| output root/profile | defaulted; deployment review required | `runs/`, `cluster.yaml` | where and how the job executes |

Selectors use chain followed by residue range, for example `A12-20`. Complex
multi-fragment selectors are supported in YAML, but ordinary users should
start with `inspect` when chain or residue identity is uncertain:

```bash
rfd3-mosaic inspect seed.cif --output-dir inspection
```

## What is optional

All options below have safe defaults. Omitting them does not disable fixed
geometry, provenance or result auditing.

| Optional control | Default | Use it when |
| --- | --- | --- |
| `--designs` | `1` | more independent generated structures are required |
| `--timesteps` | `200` | a short engineering canary or a full campaign is desired |
| `--component-motion` | `locked` | the complete rigid seed may move relative to the symmetry frame |
| pose radius/axial/orientation | no pose resampling | different assembly-level initial poses should be explored |
| `--replicates-per-pose` | `1` | several diffusion trajectories should share one pose intentionally |
| packing/cavity/diversity/interface-area preferences | balanced/auto/medium/auto | the user wants a high-level preference rather than expert loss weights |
| sequence masking or glycine conditioning | fixed sequence | the supplied surface identity should not condition RFD3 directly |
| motif side-chain redesign | off | backbone stays fixed but RFD3 may redesign motif sequence/side chains |
| ligand/RASA/hotspot/H-bond conditioning | off | the scientific input contains those features |
| non-loopy and pLDDT enhancement | on | normally leave the maintained defaults unchanged |
| trajectory output | off | denoising debugging is needed and additional storage is acceptable |
| advisory screening | on, retain all | normally leave enabled; it never deletes structures |

## Workflow A: fixed motif, generate a new interface

Use this for a motif that is not itself the oligomerization interface.

```bash
rfd3-mosaic init fixed-motif-c3.yaml \
  --task central-motif \
  --input functional-motif.cif \
  --motif-selector A10-25 \
  --symmetry C3 \
  --n-length 35 \
  --c-length 35 \
  --component-motion locked \
  --timesteps 200 \
  --designs 50
```

`locked` means the supplied motif arrangement does not move. Generated atoms
can still form scaffold and a symmetry-related interface. Use `guided` only
when the complete rigid motif may undergo packing-aware bounded movement, or
`free` when broader bounded SE(3) motion is explicitly intended.

The equivalent maintained example is:

```bash
rfd3-mosaic examples --copy central-motif --output fixed-motif-c3.yaml
```

Generated-interface measurements are advisory scientific evidence. Mosaic
does not invent a universal interface-success threshold or delete backbones
that do not meet a task-specific preference.

## Workflow B: preserve a supplied interface and connect adjacent copies

Use this when the two supplied fragments form a non-covalent interface, while
one generated protein chain connects opposite halves of adjacent cyclic
copies.

```bash
rfd3-mosaic init supplied-interface-c3.yaml \
  --task supplied-interface \
  --input interface-seed.cif \
  --side-a A20-35 \
  --side-b B40-55 \
  --symmetry C3 \
  --interface-scaffold adjacent-linker \
  --linker-minimum 70 \
  --linker-maximum 100 \
  --timesteps 200 \
  --designs 50
```

The two interface fragments share one joint-rigid coupling group. Mosaic does
not deform their relative geometry and does not join the same-copy interface
partners with a peptide bond. For each independently sampled cyclic pose, the
compiler compares the `+1` and `-1` neighbours and freezes the nearer valid
polymer direction. Offset zero is never considered.

Copy the maintained example with:

```bash
rfd3-mosaic examples --copy supplied-interface \
  --output supplied-interface-c3.yaml
```

## Workflow C: extend a supplied non-covalent complex

Use this when supplied partners must remain non-covalently associated while
RFD3 grows independent terminal scaffold. The example uses one concrete
symmetry only to make the command executable; the workflow is not restricted
to that symmetry or to a particular oligomer size.

```bash
rfd3-mosaic init supplied-complex.yaml \
  --task supplied-interface \
  --input supplied-complex.cif \
  --side-a A1-80 \
  --side-b B1-60 \
  --symmetry C4 \
  --interface-scaffold terminal-extensions \
  --new-oligomer-interface \
  --sequence-conditioning masked \
  --redesign-motif-sidechains \
  --component-motion guided \
  --pose-radius-minimum 20 \
  --pose-radius-maximum 32 \
  --pose-axial-minimum -4 \
  --pose-axial-maximum 4 \
  --pose-orientation uniform_so3 \
  --pose-seed 4200 \
  --n-length 30 \
  --c-length 30 \
  --timesteps 200 \
  --designs 50
```

`terminal-extensions` is essential here: it grows the two non-covalent
partners independently. It does not create an artificial peptide bond between
them. `--new-oligomer-interface` is a separate explicit choice that activates
generated-generated interface guidance.

Copy the complete maintained YAML with:

```bash
rfd3-mosaic examples --copy supplied-interface-oligomer \
  --output supplied-complex.yaml
```

The short initializer intentionally has two interface-side arguments. It is
the convenient front end for the common two-sided case, not a two-component
limit in the compiler. A three-part seed, several preserved interfaces, or a
mixture of preserved and generated relations uses the general assembly graph:

```yaml
name: multi-component-seed
input: supplied-complex.cif
symmetry: C4

components:
  alpha:
    selectors: [A1-40]
  beta:
    selectors: [B1-35]
  gamma:
    selectors: [C1-30, C45-60]
    geometry: joint_rigid

interfaces:
  - id: preserve_alpha_beta
    between: [alpha, beta]
    relation: {mode: preserve_input}
  - id: preserve_beta_gamma
    between: [beta, gamma]
    relation: {mode: preserve_input}

connections:
  - id: alpha_to_gamma
    from: alpha.C
    to: {component: gamma, selector: C1-30, terminus: n}
    length: {minimum: 25, maximum: 45}
```

Components are rigid graph nodes. A `joint_rigid` component may contain
several fragments whose complete relative geometry must remain exact.
Interfaces are geometric relations and do not create peptide bonds;
connections are the only declarations that create generated polymer paths.
Therefore an interface seed may contain two, three or more supplied fragments.
The repository also includes a complete, validation-tested
[three-component assembly-graph example](../../examples/rfd3_mosaic/public_three_component_graph.yaml).

For a larger campaign, set the desired integer output count:

```yaml
sampling:
  timesteps: 200
  designs: 250  # choose the count required by the experiment
  replicates_per_pose: 1
```

One YAML produces the requested number of independently named outputs. With a
stochastic `initial_pose`, the default gives each design its own pose and
diffusion seed.
Without `initial_pose`, a locked design retains one declared pose and varies
only the diffusion trajectory.

## Native RFdiffusion3 conditioning

Conditioning is optional. Use only channels supported by the scientific input.
The complete mapping and deliberate safety boundaries are documented in
[Native RFdiffusion3 capabilities](RFD3_NATIVE_CAPABILITIES.md).

### Sequence treatment

Choose exactly one treatment for each fixed protein selector:

| Treatment | CLI | Effect |
| --- | --- | --- |
| fixed | `--sequence-conditioning fixed` | preserve input identity and fixed atoms; default |
| masked | `--sequence-conditioning masked` | hide sequence identity; fixed coordinates remain governed by the motif contract |
| glycine | `--sequence-conditioning glycine` | condition on backbone-only all-glycine identity |

Add `--redesign-motif-sidechains` with `masked` when RFD3 should redesign
motif sequence and side-chain coordinates while preserving its backbone.
Glycine conditioning and motif side-chain redesign are intentionally
incompatible because glycine itself is the conditioning identity.

### Ligands, surface state, hotspots and hydrogen bonds

The short initializer accepts repeatable ligand selectors:

```bash
  --ligand-selector B1 \
  --ligand-selector D1
```

More detailed atom-level conditioning is written in the generated YAML:

```yaml
conditioning:
  sequence:
    - {selector: A20-35, mode: masked}
    - {selector: B40-55, mode: masked}
  ligands:
    - {selector: L1, coupling_group: supplied_interface}
  buried:
    - {selector: L1, atoms: ALL}
  partially_buried:
    - {selector: A42-45, atoms: TIP}
  exposed:
    - {selector: A80-85, atoms: ALL}
  hotspots:
    - {selector: A42-45, atoms: TIP}
  hbond_acceptors:
    - {selector: L1, atoms: O1,O2}
  hbond_donors:
    - {selector: A67, atoms: N}
  redesign_motif_sidechains: true
  origin_strategy: hotspots
```

`origin_strategy: hotspots` requires at least one hotspot selection. Atom names
are checked during `validate`; invalid or ambiguous selectors stop before GPU
inference.

### Sampling controls passed to RFD3

```yaml
sampling:
  timesteps: 200
  designs: 50
  replicates_per_pose: 1
  seed: 42000
  low_memory_mode: true
  is_non_loopy: true
  plddt_enhanced: true
  dump_trajectories: false
```

The maintained defaults favor normal production use. Enable
`dump_trajectories` only for debugging because trajectory files can be large.

## Pose diversity and diffusion diversity are different

For a stochastic pose declaration:

```yaml
sampling:
  designs: 100
  replicates_per_pose: 1
  seed: 20000
  initial_pose:
    radius: {minimum: 20.0, maximum: 32.0}
    axial_offset: {minimum: -4.0, maximum: 4.0}
    orientation: {method: uniform_so3}
    seed: 10000
```

Mosaic produces 100 independent assembly poses and 100 diffusion trajectories.
Set `replicates_per_pose: 5` only for an intentional experiment with five
diffusion trajectories per pose; 100 requested designs then use 20 poses.
Every assignment is recorded in `pose_manifest.json`.

Available orientation policies are:

- `fixed`: retain the declared orientation;
- `uniform_so3`: unbiased Haar-uniform rigid orientation;
- `principal_axis_cone`: explicit opt-in prior restricting the long-axis tilt.

Mosaic does not silently impose a ring shape, pore size or preferred helix
orientation. Declare assembly-size intent explicitly under `assembly_shape`
when the scientific task requires it.

## Local and Slurm execution

`local` means direct synchronous execution, not a particular computer:

```bash
rfd3-mosaic run design.yaml --profile local --run-root "$PWD/runs"
```

For Slurm, create one site profile once:

```bash
rfd3-mosaic profiles --copy-slurm my-cluster.yaml
# Edit account, partition, setup commands and checkpoint paths.
rfd3-mosaic doctor --profile my-cluster.yaml
rfd3-mosaic run design.yaml --profile my-cluster.yaml
```

The design YAML remains portable. Scheduler account, environment activation,
checkpoint and GPU request belong in the profile, not in the scientific task.

## Outputs

Every run retains:

- resolved configuration and exact input structure;
- compiler-generated `rfd3_input.json` and pre-diffusion CIFs;
- one root-level generated `*_model_0.cif.gz` per produced design;
- an incremental `generated_structures_cif/` directory of plain CIF files;
- `generated_structures_cif.zip`, containing only generated CIF structures;
- pose/diffusion seed provenance and source snapshot;
- per-design constraint, scaffold and advisory audit JSON;
- HTML and JSON reports.

Use:

```bash
rfd3-mosaic status RUN_ID_OR_DIRECTORY
rfd3-mosaic report RUN_ID_OR_DIRECTORY
rfd3-mosaic audit RUN_ID_OR_DIRECTORY
```

`GENERATED` means RFD3 produced coordinates. Contract checks and advisory
measurements are reported separately. Mosaic retains flagged structures; the
user decides whether to relax, refold, rank or discard them.

## Common mistakes

| Mistake | Correct action |
| --- | --- |
| one YAML per structure | write one YAML and set `sampling.designs` |
| supplying both interface sides as one peptide connection | use `terminal-extensions` for a non-covalent dimer |
| always using cyclic `+1` | use `nearest_adjacent`, which evaluates `+1` and `-1` per pose |
| expecting `designs` to change pose without `initial_pose` | declare a radius/orientation distribution |
| using `masked` while side-chain atoms remain fixed | use the initializer or enable motif side-chain redesign as intended |
| combining glycine and side-chain redesign | use either glycine conditioning or masked redesign |
| treating a compiled input CIF as a generated design | generated structures end in `*_model_0.cif[.gz]` |
| treating an advisory flag as deleted/failed output | inspect the retained CIF and audit details |
| running a large campaign before a canary | validate, then run a small pilot before scaling |

## Recommended campaign progression

1. Run `doctor`, `plan` and `validate`.
2. Generate 2-10 designs at 50 timesteps as an engineering canary.
3. Inspect fixed geometry, continuity, clashes, topology and output naming.
4. Generate 20-50 designs at 200 timesteps for a scientific pilot.
5. Compare fixed, masked and glycine conditioning only when relevant.
6. Scale to the required campaign size after the task definition and output
   behavior are frozen.

For exact argument spelling, use `rfd3-mosaic init --help`. For expert graph
assembly, quotient or polyhedral declarations, consult the
[command-line reference](USER_CLI.md) and the current
[project status](PROJECT_STATUS.md) before allocating a production campaign.
