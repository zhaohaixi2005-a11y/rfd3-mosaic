# Experiment configurations

This directory contains both current GPU evidence inputs and historical
diagnostic canaries. A filename containing `canary`, `smoke`, `v100` or `h100`
does not by itself identify the current product version.

These files are source-tree validation assets, not packaged user examples.
Their output roots are deliberately portable placeholders. Supply the actual
location at execution time with `--run-root` rather than committing a personal
or institutional filesystem path.

## Release-gate source of truth

The authoritative mapping is `GATES` in
`scripts/rfd3_mosaic/submit_gpu_release_gates.py`. That mapping records the
current configuration, resource class and machine-readable acceptance
criteria for each gate. It is intentionally not duplicated here, because a
static filename list becomes stale whenever the validation matrix changes.

The current LHD101 supplied-interface template is
`lrz_mosaic_lhd101_c3_guided_50step_template.yaml`.

## Ordinary-intent fixtures

Files named `lrz_simple_*_intent.yaml` exercise the resolver for incomplete
ordinary-user topology declarations. They must be resolved to a strictly
replayable public design before RFD3 execution.

## Historical and diagnostic inputs

Superseded inputs that have a direct maintained replacement are isolated in
`archive/superseded/`. Other active-directory YAMLs may remain useful
development regressions, but they are not automatically current release
gates. In particular, do not substitute an archived `short` O/I input for the
maintained 50-step runtime gate, and do not compare results across files
without checking the frozen commit, resolved configuration and pose manifest.

New release gates must be added to the launcher mapping with explicit
machine-readable acceptance criteria. Do not create an unindexed look-alike
YAML and infer its maturity from the filename alone.
