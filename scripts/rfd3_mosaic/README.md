# Script classification

The supported user entry point is the `rfd3-mosaic` CLI. Scripts in this
directory are development, validation or historical research helpers; their
presence does not make every script a current product path.

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
  GPU evidence matrix through `rfd3-mosaic run/submit`. The `closure` tier and
  every gate's machine-readable acceptance list are maintained in
  `docs/internal/GPU_ACCEPTANCE_MATRIX_2026_08_25.md`.
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

## Historical direct-execution scripts

The following files call the adapter or `rfd3.run_inference` directly. They
are isolated below `archive/legacy_direct/` to explain old runs and reproduce
historical experiments. Do not use them to validate the current public
compiler/worker/report contract.

- `archive/legacy_direct/lhd101_c3_central_motif_probe_p100.sbatch`
- `archive/legacy_direct/lhd101_c3_full_h100.sbatch`
- `archive/legacy_direct/lhd101_c3_full_single.sbatch`
- `archive/legacy_direct/lhd101_c3_smoke.sbatch`
- `archive/legacy_direct/lhd101_c5_mobile_pilot_p100.sbatch`
- `archive/legacy_direct/lhd101_cn_full_p100.sbatch`
- `archive/legacy_direct/prism_c3_g2_fixed_mosaic.sbatch`
- `archive/legacy_direct/validate_lhd101_d3_two_orbit.sh`

Their associated submission wrappers and focused research matrices are also
historical/diagnostic unless a current document explicitly names them. They
may use valid code for their original experiment, but they do not provide the
complete current provenance and reporting contract.

## Rule for new scripts

New launchers must freeze a public YAML and invoke `rfd3-mosaic run` or
`submit`. A direct call to `rfd3.run_inference` is permitted only for an
explicitly labelled historical comparison or low-level RFD3 test.

See `docs/internal/EXECUTION_PATH_AUDIT_2026_08_21.md` for the full path and
version audit.
