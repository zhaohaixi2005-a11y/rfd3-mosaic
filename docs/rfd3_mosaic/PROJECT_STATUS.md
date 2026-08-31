# RFD3-Mosaic project status

## 2026-08-31 layered SE(3) capture correction

- Ordinary `create_symmetric_interface` and explicit cross-chain
  supplied-interface designs now enable intra-chain scaffold-core guidance by
  default. An explicit zero remains a supported native-RFD3 ablation.
- The core controller now combines the chain-wide tertiary-support mean with
  a normalized smooth maximum over contiguous sequence windows. This targets
  long unsupported generated arms without inventing a universal structure
  pass threshold.
- Movable cyclic supplied interfaces receive a pre-RFD3 feasibility gate for
  cyclic-wedge occupancy, local tangential orientation, linker reachability
  and endpoint-corridor geometry. Rejected independent poses are resampled;
  feasible poses are not collapsed to one optimum.
- Early full-SE(3) capture probes signed radial, tangential and axial
  translation/rotation directions. A design-seeded near-optimal gain pool
  preserves replayable pose diversity while accepting only objective-lowering
  candidates. Settle and polish retain projected-gradient line search.
- Supplied-interface capture transitions from a Ho-Yeung-style neighbouring
  chain midpoint to tertiary-support-weighted core centers, reducing the
  leverage of long unsupported arms. Locked components remain immobile and
  complete joint-rigid seed geometry remains exact.
- The complete Mosaic CPU unit suite passes `990` tests plus `119` subtests.
  A matched 50-step GPU cohort is still required before claiming a scientific
  improvement in helix/strand content, unsupported-run length or monomer
  packing.

## 2026-08-28 completion snapshot and I-continuity closure

The current branch is ready for an internal research-preview release.  The
supported Cn/Dn compiler, fixed-geometry runtime, supplied-interface workflow,
per-design pose semantics, CLI, reports, CIF archives and portable execution
profiles are implemented and regression-tested. The complete local CPU gate
passes 978 tests, and the maintained public documentation passes its automated
boundary check. Release CI on the laboratory branch and frozen GPU pilots for
new conditioning combinations remain release evidence rather than missing
architecture.

The latest two-output I canary materially changes the previous I status.  Both
written structures contain 60 chains of 50 residues, with zero CA chain breaks
and zero cross-chain CA-segment collisions.  Copy-internal distance matrices
agree to a maximum error of approximately `4.07e-5 A`.  I generation,
copy-consistent geometry, generated-scaffold CA continuity and cross-chain
topology are therefore closed at the structure-output level.

This is not yet a claim that every local geometry check is clean.  The two
independent ASU backbones contain respectively two and three non-neighbour CA
overlaps; exact I copying repeats them to 120 and 180 reported observations.
The smallest ASU-local distances are approximately `1.08--1.40 A`, and one
generated/fixed junction per ASU has an advisory C--N outlier.  These outputs
remain generated, continuous and scientifically reviewable; Mosaic does not
discard them or convert an advisory local-repair recommendation into a user-
independent failure.  A clash-free I quality cohort remains open.  The
downloaded structure-only bundle did not include the frozen RFD3 input and
result JSON, so fixed-target RMSD and transform-coordinate residuals must
still be taken from the retained run audit rather than inferred from the two
CIFs alone.

Current delivery gates are:

| Delivery slice | Current state | Remaining evidence |
| --- | --- | --- |
| v0.1 research-preview engineering | release candidate | one green laboratory-branch release workflow and tagged frozen release |
| Cn/Dn exact fixed-motif and supplied-interface core | engineering/stable | preserve golden GPU replays across future changes |
| Generic supplied-interface expansion plus native conditioning | CPU validated, public | small mask/glycine/ligand GPU pilots on real inputs |
| Static and bounded-mobile T/O | GPU engineering closed | later multi-design secondary-structure benchmarking is advisory |
| I generation, copy consistency, CA continuity and cross-chain topology | structure-output closed | retained full run audit plus clash-free/local-repair quality evidence |
| Generated-new-interface packing | implemented | comparative multi-seed scientific calibration |
| Route-ownership anti-weaving protection | CPU validated | matched current-revision C3 GPU cohort |

These gates replace a misleading single project-completion percentage.  They
support releasing the present backbone-design system for laboratory pilots
without claiming that generated-interface designability, sequence design,
refolding or experimental success has already been scientifically validated.

## 2026-08-27 public conditioning and interface-seed product closure

- Supplied-interface scaffolding is a generic public workflow, not an
  LHD101-specific template. Any two declared interface partners can be kept
  in one joint-rigid component, grown as separate non-covalent polymer paths,
  and optionally used as the building block of a second Cn oligomerization
  interface.
- `sampling.designs` retains its assembly-design meaning: movable tasks may
  instantiate an independently seeded feasible SE(3) pose per design, while
  `replicates_per_pose` explicitly requests several diffusion trajectories
  from one pose. The generated multi-example input is executed with one RFD3
  model load.
- Public conditioning now covers native RFD3 sequence masking, all-glycine
  backbone conditioning, symmetric ligands, RASA state, hotspots, hydrogen-
  bond donors/acceptors, motif side-chain redesign, COM/hotspot origin,
  non-loopy conditioning, pLDDT enhancement and optional trajectory output.
  Side-chain redesign leaves the supplied protein backbone fixed and is not
  silently cancelled by an all-atom fixed mask.
- Raw native parser/frame controls that would invalidate a compiler-owned
  exact assembly (`allow_realignment`, arbitrary `unindex`, raw `ori_token`
  and `partial_t`) remain fail-closed rather than being exposed as unsafe
  passthrough strings. The public native-capability matrix records both the
  supported mappings and these deliberate boundaries.
- The maintained example `supplied-interface-oligomer` and the `init` command
  demonstrate one complete non-covalent interface-seed path. The general
  assembly graph supports multiple rigid components and interface relations.
  It remains a scientific user choice whether to request an additional
  generated interface,
  mask/redesign the supplied surface, include a ligand or sample different
  initial poses; none is enabled implicitly for existing designs.

## 2026-08-27 endpoint and route-ownership protection (CPU implementation)

- Strict executable compilation now rejects any non-break generated link for
  which the maximum declared residue count cannot span the fixed endpoint
  distance at the documented 3.8 A CA contour reference. Diagnostic relaxed
  compilation still records the infeasible copies and tie groups.
- Two-fixed-anchor generated paths now carry compiler-derived route ownership
  into the frozen RFD3 input. During sampling, generated CA coordinates are
  penalized when they enter another chain's closer endpoint corridor; fixed
  atoms are excluded, legal inter-chain contacts are not prohibited, and no
  inward/outward ring morphology is prescribed.
- The pre-existing cross-chain CA-segment barrier remains responsible for
  literal backbone-segment collisions. Route ownership, segment collision,
  continuity and fixed-motif preservation remain separate recorded terms.
- A principal-axis cone sampler is implemented as an explicit optional prior,
  but all maintained workflows retain Haar-uniform `uniform_so3`. Ho-Yeung's
  broad interface-seed orientation sampling does not support silently making
  a 45-degree cone the LHD or cyclic default.
- The mathematical definitions, normalization, claim boundary and limitations
  are recorded in `RIGID_MOBILITY_MATHEMATICAL_CONTRACT.md`. Static checks and
  the complete 978-test CPU suite pass. A matched old/new GPU comparison is
  still required before the route-ownership change is called scientifically
  validated.

## 2026-08-26 polyhedral bounded-mobility correction (CPU implementation)

- Dynamic T/O `bounded_se3` no longer asks the runtime to invent one global
  cyclic primary axis. Polyhedral groups have several equivalent axes; the
  previous unconditional Cn/Dn axis extraction stopped T before diffusion.
- The polyhedral path retains junction, clash, pose-prior, exact-orbit and
  atomic-transaction terms. Only the physically undefined global-axis tilt
  term is absent. Axis-dependent radial, axial and tilt-only mobility remains
  restricted to Cn/Dn and fails closed for T/O/I.
- The dynamic O canary now uses a generic master direction with 24 distinct O
  images. Its former `[1, 1, 0]` direction lies on a two-fold stabilizer and
  collapsed the nominal 24-copy orbit to 12 unique placements.
- Targeted sampler/controller tests and complete CPU input validation pass for
  both dynamic T (12 copies) and dynamic O (24 copies). Corrected 50-step GPU
  canaries subsequently closed both engineering gates: bounded mobility
  executed, the complete fixed orbits and exact symmetry were retained, and
  both written CA traces had zero breaks and zero clashes. Exact run identifiers
  remain in the internal GPU evidence matrix; the single outputs are not
  secondary-structure benchmarks.

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
- Python compilation, archive streaming, and the complete 978-test CPU unit
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
generated--generated inter-chain contacts. Generated-new-interface packing is
implemented, while its comparative scientific calibration remains open: the
most recent H100, A100 and RTX 3070 evidence contains no output meeting the
complete online packing-proxy target bundle across 20 completed structures.
This is not a missing execution path or a binary release failure. Those
outputs remain generated backbones; the result diagnoses controller
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
representation gap is the next generated-interface calibration question.

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
recovered and scaffold validity passed. Historical 50-step I canaries
materialized all 60 copies but reproduced ASU-local generated/fixed boundary
defects.  The latest two-output I canary removes those CA discontinuities in
all 60 copies and has no cross-chain segment collision.  It retains two or
three unique ASU-local CA overlaps, so I continuity is now closed while
clash-free local geometry remains a separate quality gate. Exact run
identifiers and paths are retained only in the internal validation history.
These are static execution and fixed-
geometry-scaffolding milestones. Dynamic T/O bounded mobility is GPU-closed
after correcting the polyhedral-axis and O-stabilizer initialization errors.
Dynamic I mobility and production-
quality generated-interface polyhedral cages remain separate, unvalidated
scientific capabilities.

The first local-anchor correction was shown by two GPU reruns to execute but
to be erased by the high-noise diffusion lifecycle.  The superseding,
packing-independent generated-polymer continuity projection is now supported
by two 60-copy outputs with zero CA breaks.  The accurate product status is
therefore “I inference and generated-scaffold continuity available; local
clash-free quality and dynamic-I mobility remain open.”

## Known incomplete areas

- repeated multi-seed GPU validation across diverse real inputs;
- scientific calibration of generated-interface packing quality;
- stable schema migration and long-term release compatibility;
- polished tutorials and a broader public example library.

Sequence design and independent refolding are planned downstream workflow
stages. Their integration remains under development and is evaluated
separately from the current backbone-generation release gates.

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
