# RFD3-Mosaic project status

## Release stage

RFD3-Mosaic is an actively developed **research preview**. It is installable,
has a unified public CLI and executes the same compiler/sampler/audit path
independently of the host site. Direct and Slurm launch adapters are currently
built in; neither restricts the server or GPU model. The project is not yet a
stable production release.

## Supported release target

The current supported target comprises:

- Cn and Dn symmetric execution;
- exact fixed-motif and supplied-interface preservation;
- multiple rigid components and motif orbits;
- generated-interface packing guidance;
- locked and bounded component mobility, including guided translation and
  rotation;
- deterministic lowering to RFD3 inputs;
- strict replay, explicit geometry contracts and non-destructive advisory
  screening;
- guided configuration creation and portable packaged examples;
- per-design assembly-pose instantiation with explicit pose/diffusion seed
  provenance and one-load RFD3 multi-example execution;
- local execution and configurable Slurm execution.

These capabilities have extensive CPU regression coverage. Representative GPU
validation exists for important paths, but broader multi-seed and packing
quality calibration is still in progress.

Current retained GPU evidence includes exact C3 and D3 fixed/mobile paths, a
complete static tetrahedral 12-copy result, a six-design LHD101 C3
supplied-interface pilot and a 40-design independent-pose LHD101 cohort.  In
the 40-design cohort, all outputs preserve the supplied joint-rigid interface
and remain continuous; 27 meet the historical configured-check bundle and 28
are free of CA clashes under the current coarse screen.  None meets all
current advisory monomer-core controller targets, but those targets are not a
published RFdiffusion/RFD3 backbone-generation acceptance standard.  Several
clash-free outputs combine good tertiary support with no unintended
generated--generated inter-chain contacts.  Generated-new-
interface packing remains the principal scientific blocker: the most recent
H100, A100 and RTX 3070 evidence contains no output meeting the complete
online packing-proxy target bundle across 20 completed structures.  Those
outputs remain generated backbones; this result diagnoses controller
calibration rather than assigning user-level rejection. Runtime reciprocal
CA-window continuity and stricter post-hoc backbone-heavy-atom continuity are
now reported separately, because they are different measurements and must not
be collapsed into one acceptance label.

The historical 20-structure packing evidence all reused one supplied input
pose within each job and therefore did not test the per-design assembly-pose
sampler. Maintained locked/guided canaries now use paired independent 20--30 A,
Haar-SO(3) pre-diffusion poses while preserving their distinct runtime motion
contracts. The controller now also combines an early, quadratically decayed
RFdiffusion-style all-pair contact prior with Mosaic's later contiguous-patch
refinement. CPU contracts are complete; a new frozen 50-step GPU campaign is
required before claiming improved generated-interface yield.

## Experimental capabilities

The following capabilities are implemented at varying compiler or CPU
validation levels but are not currently presented as stable release features:

- tetrahedral, octahedral and icosahedral assembly paths;
- component stabilizers, cosets and quotient orbits;
- unknown-relative-pose multi-interface assembly solving;
- higher-participant interface hyperedges;
- fully automatic architecture selection.

Experimental means that the software fails closed when it cannot prove a
valid executable lowering. It does not mean that an unvalidated candidate is
silently accepted.

The current finite-group GPU maturity is deliberately asymmetric. Static T
has a complete 12-copy PASS. Static O now also has a documented 24-copy,
50-timestep PASS: RFD3 inference completed, the exact fixed orbit was
recovered and scaffold validity passed. The frozen 50-step I canary completed
and materialized all 60 copies. Its original high fixed-orbit flag was a
high-order chain-label ordering bug in the audit; preserving RFD3
materialization order recovers the fixed orbit at `0.000132 A` RMSD. One real
ASU generated-to-fixed peptide-junction defect is reproduced across all 60
chains, so complete I scaffold-continuity closure remains open. These are
execution and fixed-geometry-scaffolding milestones, not production-quality
generated-interface O/I cage claims.

## Known incomplete areas

- repeated multi-seed GPU validation across diverse real inputs;
- scientific calibration of generated-interface packing quality;
- general interface-edge stabilizers and mixed multiplicities;
- fully general native polymer-path lowering;
- stable schema migration and long-term release compatibility;
- polished tutorials and a broader public example library.

Sequence design and refolding are intentionally outside the current release
scope.

## Development policy

Validated workflows are extended incrementally rather than replaced. New
features must preserve exact constraints, compile to a replayable input and
report required result contracts. Scientific proxy flags remain visible
recommendations rather than user-independent rejection labels. Site-specific
cluster results are treated as validation evidence, not as dependencies of the
public software.
