# RFD3-Mosaic project status

## 2026-08-25 cross-chain topology safety (CPU implementation)

- Scaffold-core sampling now measures finite CA--CA backbone segments across
  chains, rather than relying only on CA endpoint distances. A generated
  segment crossing can therefore receive a differentiable repulsive gradient
  even when all four endpoints are farther apart than the point-clash radius.
- Supplied fixed--fixed interface geometry remains excluded from that runtime
  term. Generated--fixed and generated--generated segment approaches remain
  covered, and exact motif/symmetry projection remains authoritative.
- The final scaffold audit records broad segment proximities as advisory
  measurements but treats a sub-angstrom interior/interior backbone-segment
  collision as a topology contract violation.
- Every completed run writes a structure-only ZIP of plain CIF members plus a
  separate count/hash manifest. A PyMOL helper loads a CIF directory into one
  discrete multi-state object for keyboard navigation.
- Python compilation, archive streaming, and the complete 927-test CPU unit
  suite pass locally. A matched GPU canary remains required before calling the
  new sampling protection GPU-validated.

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

The current-revision C4/C2 quotient gate produced the declared two physical
quotient copies, preserved the complete fixed orbit and exact symmetry, and
contained no CA clash. Both chains had a CA-continuous trace;
the previous audit nevertheless classified one `2.119 A` C--N outlier per
copy as a hard chain break. The scaffold audit now follows the native RFD3
generation-stage distinction: non-contiguous numbering, missing backbone
atoms and broken CA traces remain contracts, while numeric C--N outliers are
reported as peptide-geometry advice for downstream relaxation. This changes
reporting, not the generated coordinates or quotient implementation.

The historical 20-structure packing evidence all reused one supplied input
pose within each job and therefore did not test the per-design assembly-pose
sampler. A subsequent four-output paired gate used two independent 20--30 A,
Haar-SO(3) poses. It improved minimum contact coverage to four or five
residues but still produced zero complete physical interface edges. Guided
transactions committed 12 and 22 times yet displaced the motif orbit by less
than 0.3 A and 0.8 degrees, establishing under-responsive early pose motion
rather than a missing runtime path. The controller now combines an early,
quadratically decayed RFdiffusion-style all-pair contact prior with Mosaic's
later contiguous-patch refinement and scales bounded motif response by the
observed capture/expansion/polish phase. CPU contracts are complete; a new
paired 50-step gate confirms that the phase scaling raises observed guided
motion to 0.33--1.10 A and 1.02--2.42 degrees, but still closes zero physical
interface edges. One pose removes three heavy-atom clashes; the other gains
two post-hoc contact residues but introduces three heavy-atom clashes and a
continuity flag. Runtime acceptance currently protects CA geometry while the
final audit observes backbone-heavy-atom and C--N geometry, so closing that
representation gap is the next generated-interface blocker.

## Experimental capabilities

The following capabilities are implemented at varying compiler or CPU
validation levels but are not currently presented as stable release features:

- tetrahedral, octahedral and icosahedral assembly paths;
- component stabilizers, cosets and quotient orbits;
- unknown-relative-pose multi-interface assembly solving;
- higher-participant interface hyperedges.

Experimental means that the software fails closed when it cannot prove a
valid executable lowering. It does not mean that an unvalidated candidate is
silently accepted.

The current finite-group GPU maturity is deliberately asymmetric. Static T
has a complete 12-copy PASS, including repeated exact fixed-orbit,
scaffold-continuity and clash-free results. Static O also has a documented
24-copy, 50-timestep PASS: RFD3 inference completed, the exact fixed orbit was
recovered and scaffold validity passed. A longer 50-step I canary completed
and materialized all 60 copies. Its exact fixed orbit, sub-milliangstrom
symmetry-coordinate residual and zero-CA-clash checks met their contracts, but
five ASU-local generated/fixed boundary defects were reproduced across all
copies, so I scaffold-continuity closure remains open. Exact run identifiers
and paths are retained only in the internal validation history. These are
static execution and fixed-
geometry-scaffolding milestones; dynamic T/O/I mobility and production-
quality generated-interface polyhedral cages are separate, still-unvalidated
scientific capabilities.

The first local-anchor correction was shown by two GPU reruns to execute but
to be erased by the high-noise diffusion lifecycle.  A superseding,
packing-independent generated-polymer continuity projection is now CPU
validated and awaits one frozen I 50-step rerun.  Until that rerun passes,
the product status remains “I inference and exact-orbit execution available;
I generated-scaffold continuity not yet closed.”

## Known incomplete areas

- repeated multi-seed GPU validation across diverse real inputs;
- scientific calibration of generated-interface packing quality;
- stable schema migration and long-term release compatibility;
- polished tutorials and a broader public example library.

Sequence design and refolding are intentionally outside the current release
scope.

The following are also intentionally outside the current product plan and do
not count as incomplete release work: automatic inference of symmetry/order,
component connectivity, interface multiplicity or a supposedly "best" cage
architecture from arbitrary structures; cooperative functional-site design;
and unrestricted generalization of every quotient-edge or native-polymer-path
case. Users declare the intended symmetry and architecture. Mosaic preserves,
samples, compiles and audits that declared problem and fails closed outside
the supported executable domain. Existing implementation work in these areas
is retained where it supports validated paths, but no speculative roadmap
item is advertised as a missing v0.1 feature.

## Development policy

Validated workflows are extended incrementally rather than replaced. New
features must preserve exact constraints, compile to a replayable input and
report required result contracts. Scientific proxy flags remain visible
recommendations rather than user-independent rejection labels. Site-specific
cluster results are treated as validation evidence, not as dependencies of the
public software.
