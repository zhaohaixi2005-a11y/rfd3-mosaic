# Decision transparency GPU pilot

Three paired conditions, two designs per condition, 50 diffusion steps. Run
the same commit and input on each site: six outputs per site, twelve total.
Each job requests one GPU through the site profile, eight CPUs, 64 GB host
memory and a two-hour limit. No scientific threshold is tuned by this pilot.

| Condition | Purpose |
|---|---|
| `c3_locked.yaml` | Generated-interface guidance with fixed component poses |
| `c3_guided.yaml` | Current joint rigid/interface guidance, including automatic capture |
| `c3_guided_no_intra.yaml` | Same guided setup with intra-chain compactness/support weight zero |

The guided/locked comparison changes both mobility and its automatic capture
mechanism. It measures their combined effect; it does not isolate capture.
The no-intra condition keeps safety, routing where applicable and capture;
it does not switch off the entire core controller.

All arms use pose seed 70000 and diffusion seed 949 with one replicate per
pose. Verify the actual pose manifests and compiled initial coordinates
before treating outcomes as matched: seeds alone do not establish identical
initial geometry, especially across different GPUs and software environments.

## Execution

Use a clean checkout at the published commit. Verify the checkpoint identity,
environment, import paths and scheduler resources on each site first. For
each condition, invoke the existing CLI with a site-local profile and a real
run root, for example:

```bash
python -m rfd3_mosaic.cli submit \
  experiments/decision_validation/c3_locked.yaml \
  --profile /absolute/path/to/site-profile.yaml \
  --run-root /absolute/path/to/decision-validation-runs \
  --defer-runtime-preflight
```

Deferred preflight performs expanded runtime validation inside the allocated
worker before loading the model. Record the returned Slurm job ID immediately;
if submission output is interrupted, inspect the queue/run index before retrying.

## Evidence to retain

- Source commit and snapshot hashes, checkpoint identity, GPU/environment,
  resolved configs and both seeds; compare actual initial poses.
- All generated structures, logs, per-design audits and decision explanations.
- Separate generation completion, geometry contracts and advisory outcomes.
- For active controllers: effective parameters, candidate trial decisions,
  acceptance/rollback counts, final coordinate metrics and declared motion bounds.
- Compare continuity and full available backbone geometry, contacts, compactness
  and interface measurements independently of the optimized total loss.

Two samples per arm only exercise execution and expose gross differences.
They cannot calibrate defaults, establish statistical superiority, or prove
foldability. A larger, task-stratified study with independent downstream
validation follows only after inspecting this pilot.
