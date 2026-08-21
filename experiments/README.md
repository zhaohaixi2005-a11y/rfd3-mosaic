# Experiment configuration map

This directory contains both current GPU evidence inputs and historical
diagnostic canaries. A filename containing `canary`, `smoke`, `v100` or `h100`
does not by itself identify the current product version.

## Current release-gate source of truth

The authoritative mapping is `GATES` in
`scripts/rfd3_mosaic/submit_gpu_release_gates.py`. At the time of this audit it
selects:

- `lrz_public_fixed_components_v100_canary.yaml`
- `lrz_public_c3_locked_packing_patch_capture_v100_50step.yaml`
- `lrz_public_c3_joint_packing_patch_capture_v100_50step.yaml`
- `lrz_public_d3_two_orbit_mobility_v100_canary.yaml`
- `lrz_public_c4_c2_quotient_orbit_v100_canary_s943.yaml`
- `lrz_public_t_two_orbit_initialized_short_v100_smoke.yaml`
- `lrz_public_t_designed_interface_packing_v4_v100_canary.yaml`
- `lrz_public_o_static_runtime_t50_large_gpu_canary.yaml`
- `lrz_public_i_static_runtime_t50_large_gpu_canary.yaml`

The current LHD101 supplied-interface template is
`lrz_mosaic_lhd101_c3_guided_50step_template.yaml`.

## Ordinary-intent fixtures

Files named `lrz_simple_*_intent.yaml` exercise the resolver for incomplete
ordinary-user topology declarations. They must be resolved to a strictly
replayable public design before RFD3 execution.

## Historical and diagnostic inputs

All other tracked YAMLs remain useful regression or provenance artifacts, but
they are not automatically current release gates. In particular, do not
substitute an older `short` O/I input for the maintained 50-step runtime gate,
and do not compare results across files without checking the frozen commit,
resolved configuration and pose manifest.

New current gates should be added to the launcher mapping and documented in
`docs/internal/GPU_VALIDATION_PLAN_2026_08_21.md`; do not create an unindexed
look-alike YAML and rely on its filename.

