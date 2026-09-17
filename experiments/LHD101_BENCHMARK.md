# LHD101 benchmark, September 2026

`scripts/rfd3_mosaic/prepare_lhd101_benchmark.py` prepares task YAMLs and a
manifest, without submitting GPU jobs. The default matrix requests 22 tasks,
50 designs per task and 50 diffusion steps: 1,100 backbones if every family
has enough feasible poses. Preparation records a shortfall instead of filling
it with invalid or duplicate poses.

| Family | Tasks | Seed scope |
| --- | ---: | --- |
| C2 / C4 / C6 adjacent-copy interface scaffolding | 6 | Complete LHD101 A/B interface, locked and free |
| C3 adjacent-copy interface scaffolding | 6 | Complete A/B interface, two poses, locked / guided / free |
| C3 generated-interface packing | 2 | A165–194 motif, locked and guided |
| C3 complex terminal extension | 1 | Complete A/B interface, guided |
| C3 three-component connection graph | 1 | Three independent A-chain fragments, free |
| D2 / D3 / D4 two-orbit mobility | 3 | Two independent A-chain fragments, free |
| T / O two-orbit mobility | 2 | Two independent A-chain fragments, free |
| I static continuity | 1 | A165–194 plus 20 generated residues per copy |

All coordinates originate from the maintained 7MWR input. Fragment tasks do
not claim to preserve the complete A/B interface. D/T/O fragment tasks test
orbit mobility and continuous generation, not a connected interface-seeded
cage or useful cage packing. I is static and does not test dynamic I motion.
The C4/C2 quotient task is excluded: no validated intrinsic-C2 LHD101 component
is supplied. Native conditioning combinations and cross-seed generalization
are outside this cohort.

Preparation samples 12 candidate poses per family within the explicit ranges
in the script, uses the existing hard-feasibility checks and geometry ranking,
and freezes the selected coordinates. Those ranges are experiment settings,
not universal physical thresholds. C3's two selected poses must be separated
by at least 1 Å in the documented inter-seed CA distance-spectrum descriptor.
Motion arms reuse identical initial coordinates and diffusion seed schedules;
all 50 outputs in a task use the same initial pose. The complete coordinate
arrays are compared before writing the manifest. Motion changes the whole
policy, including automatic capture where applicable, not just one force.

The finite-group fragment tasks explicitly bound SE(3) motion to 3 Å and
10 degrees, matching the existing engineering canaries. Ordinary interface
tasks use their resolved public motion policies. Compare motion arms within
families; these are not identical cross-family mobility budgets.

Run all arms of each geometry family on the same site. Freeze the source
commit, checkpoint SHA256, environment, configs, input hash, pose reports and
job IDs. Retain failed generation attempts and all outputs. Report generation
completion, seed/symmetry/continuity contracts, collisions, packing metrics,
backbone diversity and GPU cost separately. Fifty samples per fixed pose are
an initial cohort, not proof of general success across poses or seeds.

Results use the site-local `runs/rfd3-mosaic` root. Campaign definitions and
submission receipts live under `requests`, separate from the results root.
