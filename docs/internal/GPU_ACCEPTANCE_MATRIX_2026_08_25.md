# Remaining GPU acceptance matrix (2026-08-25)

## Frozen decision

This matrix records the GPU work that remains after the representative C3/D3
core paths and static T/O gates closed. It does not rerun closed static gates,
and it does not turn advisory structural preferences into user-level failure.

Every submission must be frozen from one clean Git revision. The generated
`gpu_validation_manifest.json` records that revision, the exact YAML, profile,
design count, submission output and the acceptance criteria below.

The C3 topology and C4/C2 quotient gates defer complete RFD3 feature
prevalidation to the allocated worker because the LRZ login-node memory limit
can kill AtomWorks/RFD3 input construction. Lightweight planning still runs
before `sbatch`; complete prevalidation remains mandatory and inference cannot
start if it fails.

In this matrix, `chain_break_count` means a non-contiguous or broken CA trace,
matching the native RFD3 raw-backbone chain-break measurement. Missing
backbone atoms remain a hard contract. Numeric C--N outliers are retained as
advisory peptide-geometry observations; they are not relabelled as raw
generation failure before sequence design, refolding or relaxation.

Current C4/C2 evidence: job `5761820` generated one 50-step `C4 x2` result,
met the complete fixed-orbit contract, retained exact symmetry
(`8.40e-5 A` coordinate RMSD) and had zero CA clashes. Its two CA traces were
continuous. Each copy contained the same `2.119 A` C--N advisory outlier at
residues 44--45; the earlier schema incorrectly counted those two
stereochemical observations as CA chain breaks.

The first dynamic polyhedral submissions did not constitute scientific
failures. Job `5761813` (T) reached valid 12-copy runtime features but stopped
before diffusion because scaffold-driven mobility unconditionally invoked a
Cn/Dn-only primary-axis solver. Job `5761814` (O) failed preflight because its
`[1, 1, 0]` master direction has an O-group two-fold stabilizer, producing
only 12 unique placements. The 2026-08-26 correction makes axis-free
polyhedral `bounded_se3` explicit and changes the O canary to a generic
24-image direction. Both YAMLs now pass complete local compile/RFD3 input
validation; the same two gates must be resubmitted from the corrected frozen
revision.

## Gates

| gate | question | accelerator | outputs | closure condition |
| --- | --- | --- | ---: | --- |
| `cross-chain-topology` | Does the current chain-local initialization, per-step continuity projection and segment repulsion prevent C3 supplied-interface weaving? | H100/A100/V100/P100 | 6 | fixed seed recovered; zero chain breaks, CA clashes and cross-chain CA-segment collisions in all six |
| `d3-dynamic` | Can two D3 rigid motif orbits move through the bounded controller without losing exactness? | H100/A100/V100/P100 | 1 | motion executes within bounds; exact orbit, symmetry, continuity, clash and topology contracts meet |
| `c4-c2-quotient` | Does a quotient physical interface orbit execute on GPU with the compiled multiplicity? | H100/A100/V100/P100 | 1 | output produced; quotient multiplicity and all hard contracts meet |
| `t-dynamic` | Does bounded two-orbit mobility work through all twelve T actions? | H100/A100 80G | 1 | motion executes within bounds; exact orbit, symmetry, continuity, clash and topology contracts meet |
| `o-dynamic` | Does bounded two-orbit mobility work through all twenty-four O actions? | H100/A100 80G | 1 | motion executes within bounds; exact orbits, symmetry, continuity, clash and topology contracts meet |
| `i-continuity` | Does the current per-step polymer projection close the historical I generated/fixed junction defects? | H100/A100 80G | 2 | 60 copies produced; fixed orbit and symmetry meet; zero breaks, clashes and segment collisions in both |
| `locked-packing` + `guided-packing` | Does C3 generated-interface guidance improve a matched independent-pose population? | H100/A100/V100/P100 | 4 + 4 | hard runtime contracts meet; final interface measurements are retained as advisory comparative evidence |
| `t-packing` | Does T graph-interface guidance execute and report the complete physical edge orbit? | H100/A100 80G | 1 | exact runtime contracts meet; final interface measurements are retained as advisory evidence |

Static T and static O are already closed and are deliberately absent. Existing
generated structures remain valid evidence for their frozen revisions; the
small current-revision topology gates above are regression checks for the new
sampling protection, not a request to repeat every historical trajectory.

## Submission

### Binary engineering closure (submit now)

This is the non-interface acceptance batch. D3 dynamic is not repeated because
its representative fixed/mobile GPU path is already retained as closed. Static
T and static O are also not repeated. On LRZ, after synchronizing the branch
and confirming a clean checkout:

```bash
cd /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic

PY=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/conda_environment/rc-foundry/bin/python
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"
export RFD3_MOSAIC_RUN_ROOT=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic
export DEBUG=false TYPE_CHECK=false NAN_CHECK=true

"$PY" scripts/rfd3_mosaic/submit_gpu_release_gates.py \
  --tier closure --submit
```

This command creates one timestamped campaign directory and one durable
manifest. Do not manually rename run directories or submit the frozen YAMLs a
second time. Use the job identifiers in the manifest for collection.

The closure tier contains exactly five questions: current-revision C3
cross-chain topology, C4/C2 quotient execution, dynamic T, dynamic O and I
continuity.

### Advisory interface experiments (do not mix into binary closure)

Generated-interface quality has no universal RFdiffusion/RFD3 backbone-level
absolute pass threshold. Mosaic therefore keeps its hard geometry/runtime
contracts separate from contact/shape observations. Run the following only as
a comparative scientific campaign:

```bash
"$PY" scripts/rfd3_mosaic/submit_gpu_release_gates.py \
  --gate locked-packing \
  --gate guided-packing \
  --gate t-packing \
  --submit
```

These jobs may show whether guidance improves a matched population; they do
not contribute a binary “interface generator is 100%” claim.

## Interpretation

- `generated`: RFD3 wrote a coordinate output.
- `contract met`: exact fixed/joint-rigid/symmetry/topology requirements met.
- `review`: the coordinate output exists but one hard contract is flagged.
- packing coverage, shape and contact-patch measurements are advisory. They
  compare locked and guided populations and do not decide whether a user is
  allowed to keep a generated backbone.
- a gate is closed only against the question written in this matrix. Passing
  static O never silently proves dynamic O or O generated-interface quality.
