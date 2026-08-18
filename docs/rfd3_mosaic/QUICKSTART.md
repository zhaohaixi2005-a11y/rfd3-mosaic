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

### Preserve a supplied interface

Use this workflow when the input already contains the interface geometry that
must remain unchanged:

```bash
cp examples/rfd3_mosaic/simple_interface_seed.yaml my-design.yaml
```

Mosaic treats the complete supplied interface as one geometric object. It
does not independently rearrange the two sides of that interface.

### Create a new interface around a motif

Use this workflow when the input provides a motif but the surrounding
interface should be generated:

```bash
cp examples/rfd3_mosaic/simple_central_motif.yaml my-design.yaml
```

The motif remains fixed internally. Depending on the selected component
motion policy, its assembly pose can remain locked or undergo bounded
translation and rotation while generated regions are guided toward packing.

## 3. Edit the user configuration

At minimum, check:

- `name`;
- input structure path;
- chain/residue selectors;
- target symmetry;
- generation lengths;
- output directory;
- whether supplied geometry must be preserved or a new interface created.

Ordinary users should not need to select group-transform identifiers or tune
individual low-level packing losses. Expert declarations remain available for
explicit component, port, connection and mobility control.

## 4. Plan and validate

```bash
rfd3-mosaic plan my-design.yaml --profile local
rfd3-mosaic validate my-design.yaml
```

`plan` explains the resolved components, symmetry, interfaces, constraints
and execution mode. `validate` compiles the design and performs finite runtime
feature prevalidation before GPU time is used.

## 5. Run

Run directly on any compatible machine or allocated compute node:

```bash
rfd3-mosaic run my-design.yaml \
  --profile local \
  --run-root "$PWD/runs"
```

The `local` profile name means direct synchronous execution; it does not mean
that the machine must be a personal computer. For Slurm, copy and edit the
generic profile described in [INSTALLATION.md](INSTALLATION.md), then pass its
path through `--profile`.

## 6. Inspect the result

```bash
rfd3-mosaic status RUN_ID_OR_DIRECTORY
rfd3-mosaic report RUN_ID_OR_DIRECTORY
```

A completed inference is not automatically a passed design. The final verdict
also requires every task-specific geometry, symmetry, interface and scaffold
audit to pass.

## Supported scope

The current public release target is Cn/Dn fixed-motif, supplied-interface,
generated-interface packing and bounded-mobility design. Polyhedral and more
general automatic cage-solving paths are research features and should be
evaluated explicitly rather than assumed stable.
