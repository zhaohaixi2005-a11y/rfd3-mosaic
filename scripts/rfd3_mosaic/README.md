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

- `sync_generated_cifs.py`: verifies and downloads final generated CIFs over
  SSH using Python's standard library; see the example below.
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

## Download final structures after a run

```bash
python scripts/rfd3_mosaic/sync_generated_cifs.py \
  --host USER@HOST --socket /path/to/existing-ssh.sock \
  --ssh-option UserKnownHostsFile=/path/to/known_hosts \
  --run-dir /remote/path/to/task/job \
  --relative-destination task/job \
  --destination-root /local/collection
```

For several runs on one host, replace `--run-dir` and
`--relative-destination` with `--manifest runs.json`:

```json
{
  "runs": [
    {"remote_run_dir": "/remote/run-one", "destination": "task-one/job-one"},
    {"remote_run_dir": "/remote/run-two", "destination": "task-two/job-two"}
  ]
}
```

Each destination receives a `generated_structures_cif/` directory containing
plain final CIFs and the script's `.mosaic_sync.json` verification record.
Only structures with matching final `*_model_0.json` metadata qualify;
noisy/denoised trajectories are excluded. The gzip source is preferred when
both compressed and plain versions exist. Each run uses one SSH inventory
request and at most one batched tar transfer. No remote files are modified.

Downloads stay in a temporary directory until source size/SHA256, gzip
integrity and decompressed size/SHA256 have been verified. Each final CIF is
replaced atomically; interrupted downloads cannot truncate an existing final
file. Matching local files are verified and skipped. Different unrecognized
or locally modified files are refused, not overwritten. Rerun the same
command to collect newly completed designs; this is a one-shot collector,
not a background service. It verifies transfer integrity, not scientific
quality or full mmCIF syntax. A failed request exits nonzero; already verified
outputs remain available.

Use `--jobs 2` to sync independent runs concurrently (default 1, maximum 4).
A failed run does not stop collection from the others. Incomplete result JSON
is retried three times, 0.2 seconds apart, then explicitly reported as
`pending`; completed structures in that run are still collected. Any failed
or pending run makes the command exit nonzero after all runs are processed.

## Rule for new scripts

New launchers must freeze a public YAML and invoke `rfd3-mosaic run` or
`submit`. A direct call to `rfd3.run_inference` is permitted only for an
explicitly labelled low-level RFD3 test.
