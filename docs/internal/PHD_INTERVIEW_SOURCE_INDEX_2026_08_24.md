# RFD3-Mosaic PhD interview source index

Last reviewed: 2026-08-24  
Current evidence revision: `34a79e9`

This page is the starting point for preparing a PhD-interview presentation.
It separates current evidence from historical planning notes so that old GPU
results, superseded execution paths and provisional scientific screens are not
presented as current product claims.

## One-sentence project statement

RFD3-Mosaic is an assembly-aware compiler, constrained RFD3 sampler and audit
layer for generating symmetric protein backbones while preserving supplied
motifs or complete interface seeds under explicit fixed, joint-rigid and
bounded-mobile geometry contracts.

## Read these first

1. `README.md` -- public project definition, two main user workflows and the
   current product boundary.
2. `docs/rfd3_mosaic/PROJECT_STATUS.md` -- current maturity, GPU evidence and
   remaining gaps.
3. `docs/rfd3_mosaic/HOYEUNG_BACKBONE_COMPARISON.md` -- the LHD101 background,
   matched task, paper-aligned measurements and Mosaic cohort results.
4. `docs/internal/EXECUTION_PATH_AUDIT_2026_08_21.md` -- the canonical execution
   spine, per-design pose semantics and separation of fixed geometry, mobility
   and packing.
5. `docs/internal/INTERFACE_PACKING_STATUS_2026_08_20.md` -- why generated-new-
   interface packing remains the main scientific blocker and how the current
   controller combines RFdiffusion-style broad contact capture with Mosaic
   patch refinement.
6. `docs/rfd3_mosaic/BACKBONE_EVALUATION_EVIDENCE.md` and
   `docs/rfd3_mosaic/STRUCTURE_METRIC_PROVENANCE.md` -- published or explicitly
   sourced metric formulas and the boundary between generation, advisory
   backbone screening and sequence-conditioned validation.
7. `docs/internal/GPU_VALIDATION_PLAN_2026_08_21.md` -- retained T/O/I and C3
   packing evidence, including the corrected interpretation of I job
   `5756755`.

## Background and method sources

### Ho-Yeung / Interface-Seed background

- `docs/rfd3_mosaic/HOYEUNG_BACKBONE_COMPARISON.md` is the concise current
  source for the LHD101 task and fair comparison boundary.
- `docs/internal/INTERFACE_SEED_RFD1_UPGRADE_AUDIT.md` is the detailed code
  archaeology of the RFdiffusion1 interface-seed implementation: random seed
  orientation, displacement, symmetry expansion, COM dragging, native contact
  potentials, saved pose metadata and known mathematical/software risks.
- The long audit is useful for answering detailed questions, but its archived
  implementation is not the current Mosaic runtime.

### Mosaic architecture

- `docs/internal/EXECUTION_PATH_AUDIT_2026_08_21.md` is the best architecture
  reference.
- `docs/internal/RFD3_MULTI_INTERFACE_SEED_FINAL_PLAN.md` contains the full
  Spec/Instance/State model, interface ports/edges, motion groups, symmetry
  orbits, scaffold links and mapping registry. It is an archived architecture
  record; use selected diagrams/concepts, not its historical status claims.
- `docs/internal/RFD3_MOSAIC_PRODUCTIZATION_PLAN.md` explains invariants,
  compiler/runtime boundaries, Foundry fork policy and validation tiers. It is
  also an archived plan rather than the current project-status authority.

### User-facing software

- `docs/rfd3_mosaic/QUICKSTART.md` -- shortest maintained workflow.
- `docs/rfd3_mosaic/USER_CLI.md` -- complete CLI lifecycle.
- `docs/rfd3_mosaic/PACKING_GUIDANCE.md` -- current packing controls and
  diagnostics.
- `docs/rfd3_mosaic/INSTALLATION.md` -- installability and execution profiles.
- `DEVELOPMENT_STATUS.md` -- compact public capability boundary.

## Current quantitative evidence

### Software

- Complete local CPU regression: `915` tests passed at revision `34a79e9`.
- The same revision is present on personal branch
  `origin/refactor/product-core-v1` and lab branch
  `lab/hx/rfd3-mosaic-product-core`.
- Public CLI, wheel smoke, direct execution, Slurm execution, frozen source
  provenance, status/report/audit and per-design multi-example execution are
  implemented.

### Supplied-interface LHD101 cohort

The first independent-pose cohort contains 40 generated backbones:

- supplied joint-rigid interface preserved: `40/40`;
- continuous chains: `40/40`;
- CA-clash-free under the current coarse screen: `28/40`;
- historical configured-check bundle met: `27/40`;
- observed whole-seed motion: `0.081--0.683 A` translation and
  `0.483--3.422 degrees` rotation;
- normalized chain Rg: `2.729--4.383`, median `3.426`;
- tertiary-support fraction: `0.235--0.694`, median `0.512`;
- central-pore p05 diameter: `11.02--39.41 A`.

These numbers prove independent pose sampling, fixed-interface recovery and
assembly-level geometric diversity. They do not prove sequence designability
or experimental success. The name `accepted_strict_27` is historical and means
only that 27 structures met the checks configured at that revision.

### Finite-group GPU evidence

- T: complete static 12-copy GPU closure.
- O: job `5755569`, 24 copies, 50 steps, fixed orbit and scaffold validity
  passed.
- I: job `5756755`, 60 copies, 50 steps, coordinate generation completed.
  The original `62.535 A` fixed-orbit flag was an audit chain-order false
  positive; corrected materialization-order matching gives `0.000132387 A`
  joint RMSD and `0.000241701 A` maximum error. One real ASU
  generated-to-fixed peptide-junction defect (`C--N = 3.1511 A`) is copied 60
  times, so scientific scaffold-continuity closure remains open.

T/O/I evidence demonstrates executable finite-group symmetry and constraint
handling. It is not evidence that production-quality T/O/I cages are already
generated reliably.

### Generated-new-interface packing

The independent-pose C3 campaign at revision `a935529` generated all 12
requested structures but none met the complete advisory interface target
bundle. Its strongest output had 37 reciprocal residue pairs, coverage 16,
shape loss about `0.0784`, no coarse clash and only contiguous-patch length 2.
The paired guided output was similar, demonstrating that the old bounded
controller moved too little to turn scattered contacts into a broad continuous
interface.

Revision `caabd25` therefore added an early RFdiffusion-style all-pair contact
prior with quadratic decay, while retaining Mosaic's exact constraints,
continuity/shape refinement and rollback. Revision `34a79e9` contains that
implementation. A new frozen GPU campaign is still required before claiming
improved generated-interface yield.

## Main technical contributions to present

1. **A structural-intent compiler.** User motifs, complete interface seeds,
   polymer connections and symmetry declarations become a deterministic,
   replayable Assembly IR and RFD3 input.
2. **Hierarchical geometry semantics.** Internal motif geometry, relative
   interface geometry, assembly pose and generated scaffold are distinct
   layers; `fixed` and `joint-rigid but mobile` are not contradictory.
3. **Exact symmetry-aware runtime control.** A master component is updated,
   complete copies are reconstructed by declared group actions, and hard
   constraints are restored throughout diffusion.
4. **Assembly-level diversity.** Variable-pose `designs=N` now means N
   independently seeded feasible assembly poses plus N independent diffusion
   trajectories, compiled into one RFD3 multi-example input so the checkpoint
   is loaded once.
5. **Multi-interface and finite-group representation.** Multiple supplied
   interface identities, participant hyperedges, component orbits,
   stabilizers/cosets and user-declared polymer connections share one compiler
   model instead of one script per cage.
6. **Packing as capture plus refinement.** Broad RFdiffusion-style contact
   capture provides range; Mosaic then evaluates contiguous coverage,
   orientation, shape, junctions, clashes and multi-edge balance under exact
   symmetry.
7. **Evidence and provenance.** `GENERATED`, geometry-contract status and
   advisory quality screening are reported separately; raw outputs are not
   deleted or called experimentally successful/failed by an arbitrary proxy.

## Development difficulties worth discussing

- translating informal biological intent into non-contradictory fixed/mobile
  geometry contracts;
- preventing a complete supplied interface from being confused with a task
  that must generate a new interface;
- preserving exact symmetry while moving several joint-rigid components
  during diffusion;
- restoring assembly-pose diversity after discovering that one compiled pose
  plus many diffusion seeds explores only `P(backbone | fixed pose)`;
- resolving historical scripts, frozen run snapshots and several cluster
  environments into one canonical execution spine;
- diagnosing packing capture range rather than weakening thresholds to make
  poor candidates appear successful;
- separating software correctness from backbone quality, sequence
  designability, refolding and experimental validation;
- auditing high-order O/I outputs whose legacy RFD3 chain identifiers include
  punctuation and blank labels.

## Recommended interview slide sequence

1. **Problem and biological motivation** -- preserve functional/interface
   geometry while completing symmetric assemblies.
2. **Ho-Yeung background** -- interface-seeded RFdiffusion1, orientation
   sampling, COM drag, contact potentials, generate-many/filter-later.
3. **The gap** -- native RFD3 is all-atom but conditions on a supplied pose;
   neither system provides Mosaic's explicit multi-interface compiler,
   hierarchical fixed/mobile semantics and complete audits.
4. **Mosaic concept** -- RFD1 sampling breadth + RFD3 generation + Mosaic
   constraints, assembly search and evidence tracking.
5. **Two workflows** -- preserve a supplied interface versus create a new
   symmetric interface around a motif.
6. **Canonical pipeline** -- intent -> resolve -> Assembly IR -> per-design
   pose -> RFD3 -> projection/mobility/guidance -> CIF + audits.
7. **How fixed geometry is maintained during diffusion** -- master/group
   mapping, prediction, hard restoration, exact-copy reconstruction and
   bounded joint-rigid proposal acceptance.
8. **Diversity and efficiency** -- independent pose and diffusion seeds,
   multi-example RFD3 execution and frozen provenance.
9. **Results** -- LHD101 cohort plus T/O/I technical closure, with exact
   evidence boundaries.
10. **Current scientific blocker** -- generated-new-interface packing;
    scattered contact versus broad continuous interface and the new
    broad-contact controller.
11. **Software/product contribution** -- installable CLI, portable executors,
    strict replay, tests, reports and non-destructive screening.
12. **Limitations and next experiments** -- packing GPU calibration, broader
    multi-seed validation and later sequence/refolding work.

## Claims to avoid

- Do not call the project a universally solved automatic cage designer.
- Do not call all advisory-flagged generated CIFs failed proteins.
- Do not call CPU prevalidation GPU evidence.
- Do not claim that T/O/I technical symmetry closure proves high-quality cage
  design.
- Do not compare Mosaic directly with bare RFD3 as if bare RFD3 implemented
  the same interface-seed task.
- Do not use provisional normalized-Rg or tertiary-support thresholds as
  published universal acceptance standards.
- Do not claim sequence designability, AlphaFold recovery or experimental
  success; those stages are currently outside scope.

## Existing presentation assets

- `reports/RFD3-Mosaic_Project_Update_2026-08-18_EN.pptx`
- `reports/RFD3-Mosaic_Project_Update_2026-08-18_EN.pdf`
- `reports/RFD3-Mosaic_English_Presentation_Guide_CN_2026-08-18.md`
- `reports/RFD3_Mosaic_Keynote_Compatible_Flattened.pptx`
- `reports/ppt_preview/`

These are useful visual/layout references but are not current factual
authorities. They predate independent per-design pose execution, the 40-design
LHD101 cohort, O/I 50-step evidence, broad-contact packing and the 915-test
suite. Any new interview deck must update those claims from this source index
and the current status documents above.

## Historical documents: consult, do not quote as current status

- `docs/internal/DEVELOPMENT_STATUS_HISTORY.md`
- `docs/internal/CURRENT_PRODUCT_STATUS_HISTORY.md`
- `docs/internal/USER_CLI_DEVELOPMENT_HISTORY.md`
- `docs/internal/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md`
- `docs/internal/RFD3_MULTI_INTERFACE_SEED_FINAL_PLAN.md`
- `docs/internal/RFD3_MOSAIC_PRODUCTIZATION_PLAN.md`
- `experiments/archive/superseded/`

They preserve design reasoning and old evidence, but the current authority is
`README.md`, `DEVELOPMENT_STATUS.md`, `PROJECT_STATUS.md`, the canonical
execution-path audit and the dated GPU evidence documents.
