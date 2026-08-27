# RFD3-Mosaic quick start

This guide covers the supported public workflow. It does not assume access to
any particular institution, server, scheduler or GPU model.

## 1. Check the installation

```bash
rfd3-mosaic doctor --profile local
rfd3-mosaic capabilities
```

`doctor` checks the Python environment, RFD3 imports, packaged resources,
selected executor and checkpoint without running inference.

## 2. Choose a design task

The user writes **one YAML file per scientific design request**, not one YAML
per generated structure. `sampling.designs: 1000` remains one request and
produces up to 1000 independently named outputs in the run directory.

### Preserve a supplied interface

Use this workflow when the input already contains the interface geometry that
must remain unchanged:

```bash
rfd3-mosaic init my-design.yaml \
  --task supplied-interface \
  --input interface-seed.pdb \
  --side-a A165-194 \
  --side-b B211-241 \
  --symmetry C3
```

Mosaic treats the complete supplied interface as one geometric object. It
does not independently rearrange the two sides of that interface.
Use `--component-motion free` when that complete object may translate and
rotate relative to the symmetry axis while its internal interface remains
exact.

There are two distinct ways to scaffold a supplied interface:

- `adjacent-linker` creates a declared covalent polymer connection between
  symmetry-neighbour motifs;
- `terminal-extensions` grows each non-covalent partner independently and
  never joins the supplied interface sides with a peptide bond.

For a joint-rigid dimer that should assemble into a new higher-order Cn
oligomer, create the second form explicitly:

```bash
rfd3-mosaic init dimer-oligomer.yaml \
  --task supplied-interface \
  --input light-dependent-dimer.cif \
  --side-a A1-153 \
  --side-b C1-26 \
  --symmetry C3 \
  --interface-scaffold terminal-extensions \
  --new-oligomer-interface \
  --sequence-conditioning masked \
  --redesign-motif-sidechains \
  --ligand-selector B1 \
  --component-motion guided \
  --pose-radius-minimum 20 \
  --pose-radius-maximum 32 \
  --pose-axial-minimum -4 \
  --pose-axial-maximum 4 \
  --pose-orientation uniform_so3 \
  --pose-seed 1000 \
  --designs 50
```

The supplied A/C interface and ligand remain one rigid seed. The explicit
`--new-oligomer-interface` switch asks only the generated residues to form an
additional C3 interface. Omitting it preserves the supplied interface without
inventing a second one.

The pose interval is explicit rather than inferred from a protein-specific
template. With the default `--replicates-per-pose 1`, the command above
compiles 50 independent rigid assembly poses and gives each pose one RFD3
diffusion trajectory. Omit all pose options to retain the supplied assembly
placement exactly.

### Create a new interface around a motif

Use this workflow when the input provides a motif but the surrounding
interface should be generated:

```bash
rfd3-mosaic init my-design.yaml \
  --task central-motif \
  --input motif.pdb \
  --motif-selector A12-20 \
  --symmetry C3
```

The motif remains fixed internally. Depending on the selected component
motion policy, its assembly pose can remain locked or undergo bounded
translation and rotation while generated regions are guided toward packing.

Use `--component-motion guided` for constrained packing-aware translation and
rotation, or `--component-motion free` for bounded SE(3) motion. The default
is `locked`, which keeps the complete supplied arrangement fixed while
guidance acts only on generated atoms.

To browse rather than initialize from arguments:

```bash
rfd3-mosaic examples
rfd3-mosaic examples --copy central-motif --output my-design.yaml
```

The copied YAML contains a resolved input path and a local run directory, so
it remains valid outside a source checkout.

## 3. Edit the user configuration

`init` already writes safe defaults. At minimum, check:

- `name`;
- input structure path;
- chain/residue selectors;
- target symmetry;
- generation lengths;
- output directory;
- whether supplied geometry must be preserved or a new interface created.

Optional native RFdiffusion3 conditioning is placed under `conditioning`:

```yaml
conditioning:
  sequence:
    - {selector: A1-153, mode: masked}  # or glycine
  ligands:
    - {selector: B1, coupling_group: supplied_interface}
  buried:
    - {selector: B1, atoms: ALL}
  hotspots:
    - {selector: A42-45, atoms: TIP}
  hbond_acceptors:
    - {selector: B1, atoms: O1,O2}
  redesign_motif_sidechains: false
  origin_strategy: hotspots

sampling:
  is_non_loopy: true
  plddt_enhanced: true
```

`buried`, `partially_buried`, `exposed`, `hotspots`, `hbond_acceptors` and
`hbond_donors` use RFdiffusion3's documented atom-selection vocabulary.
Mosaic remaps the selected source fragments after symmetry compilation and
RFD3 validates atom names during `validate`. These fields are optional; their
omission retains the standard RFD3 defaults.

Ordinary users should not need to select group-transform identifiers or tune
individual low-level packing losses. Expert declarations remain available for
explicit component, port, connection and mobility control.

If the structure's interface selectors are not known, inspect it first:

```bash
rfd3-mosaic inspect assembly.cif --output-dir inspection
```

## 4. Plan and validate

```bash
rfd3-mosaic plan my-design.yaml
rfd3-mosaic validate my-design.yaml
```

`plan` explains the resolved components, symmetry, interfaces, constraints
and execution mode. `validate` compiles the design and performs finite runtime
feature prevalidation before GPU time is used.

## 5. Run

Run directly on any compatible machine or allocated compute node:

```bash
rfd3-mosaic run my-design.yaml
```

The `local` profile name means direct synchronous execution; it does not mean
that the machine must be a personal computer. For Slurm, copy and edit the
generic profile described in [INSTALLATION.md](INSTALLATION.md), then pass its
path through `--profile`.

List the profiles visible to the installed copy or create a scheduler file:

```bash
rfd3-mosaic profiles
rfd3-mosaic profiles --copy-slurm my-cluster.yaml
rfd3-mosaic doctor --profile my-cluster.yaml
```

## 6. Inspect the result

```bash
rfd3-mosaic status RUN_ID_OR_DIRECTORY
rfd3-mosaic report RUN_ID_OR_DIRECTORY
```

A completed inference and its structural checks are reported separately.
`GENERATED` means that a coordinate output exists; contract checks report
whether declared geometry, symmetry and continuity invariants were met; and
advisory measurements support subsequent ranking without deleting outputs or
claiming experimental success.

## Supported scope

The current public release target is Cn/Dn fixed-motif, supplied-interface,
generated-interface packing and bounded-mobility design. Polyhedral and more
general automatic cage-solving paths are research features and should be
evaluated explicitly rather than assumed stable.
