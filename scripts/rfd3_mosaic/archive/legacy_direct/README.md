# Historical direct-execution scripts

These scripts are preserved for provenance only. They predate the unified
public `rfd3-mosaic run` path and may call the adapter or
`rfd3.run_inference` directly.

They are intentionally outside the normal script directory so that users do
not mistake them for maintained launchers. Running one requires its full
archive path. New development must not add files here unless the purpose is
to reproduce a historical run.

For current work use one of:

```text
rfd3-mosaic run DESIGN.yaml
scripts/rfd3_mosaic/submit_gpu_release_gates.py
scripts/rfd3_mosaic/submit_packing_replicates.py
scripts/rfd3_mosaic/submit_mosaic_lhd101_c3_1000.py
```

