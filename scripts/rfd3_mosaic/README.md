# Developer and validation scripts

The supported user entry point is the `rfd3-mosaic` CLI. This directory
contains campaign launchers, result collectors, release checks and historical
research helpers. A script is not part of the public execution contract unless
this document identifies it as maintained.

## Review many CIF structures in PyMOL

Loading many CIF files as separate PyMOL objects does not create a trajectory,
so frame/state keyboard shortcuts cannot switch between them. Load them as
independent states of one discrete object instead:

```text
run /path/to/rfd3-mosaic/scripts/rfd3_mosaic/load_cif_ensemble.py
load_cif_ensemble /path/to/cif_directory, mosaic_batch
load_cif_ensemble /path/to/generated_structures_cif.zip, mosaic_batch_zip
```

The loader accepts a directory/glob of `.cif` and `.cif.gz`, or the
`generated_structures_cif.zip` written by Mosaic. It searches directories
recursively by default and binds Left/Right and PageUp/PageDown to the
previous/next design. It refuses to overwrite an existing PyMOL object with
the requested name.

## Maintained campaign and collection helpers

- `submit_gpu_release_gates.py`: freezes and submits the non-redundant current
  GPU evidence matrix through `rfd3-mosaic run/submit`. Each gate carries a
  machine-readable acceptance list and writes its evidence into the run
  report.
- `submit_packing_replicates.py`: creates matched independent-pose
  locked/guided C3 packing evidence through the normal CLI. On memory-limited
  Slurm login nodes, `--defer-runtime-preflight` keeps lightweight planning
  local and performs complete RFD3 prevalidation in the allocation.
- `submit_mosaic_lhd101_c3_1000.py`: shards the LHD101 comparison campaign;
  every shard uses the normal CLI and current per-design pose semantics.
- `collect_packing_campaign.py`: collects generated-output, runtime-contract,
  runtime CA-window and post-hoc backbone-heavy-atom packing diagnostics
  without deleting raw outputs or assigning a scientific verdict. Its table
  separates interface-guidance runtime completion from the overall hard
  contract, lists the exact contract flags, and reports observed SE(3) motion
  plus committed proposals when a mobility audit is available.
- `compare_hoyeung_lhd101_backbones.py`: creates the backbone-only comparison
  report.
- `check_public_surface.py` and `release_smoke.sh`: release checks.
- `setup_local_cpu_dev.sh` and `activate_local_dev.sh`: local development
  environment helpers.
- `pymol_fixed_orbit_alignment.py`: visualization only.

Historical direct-execution scripts and personal campaign records are kept
outside the public source tree. They do not define the current
compiler/worker/report contract.

## Rule for new scripts

New launchers must freeze a public YAML and invoke `rfd3-mosaic run` or
`submit`. A direct call to `rfd3.run_inference` is permitted only for an
explicitly labelled low-level RFD3 test.
