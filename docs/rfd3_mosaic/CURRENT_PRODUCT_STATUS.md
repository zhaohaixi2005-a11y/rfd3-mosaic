# RFD3-Mosaic Current Product Status

Last updated: 2026-08-11

## Executive status

RFD3-Mosaic is a usable research-grade alpha, not yet a general automatic
protein-cage product. Its strongest completed layer is exact, audited motif
and supplied-interface scaffolding. Its largest remaining gap is converting
several supplied interface seeds into a geometrically feasible cage and then
generating consistently high-quality new packing interfaces around them.

The product has one compiler/runtime spine:

```text
ordinary intent or expert design
        -> UserDesignSpec
        -> AssemblySpecification + ConstraintPlan + SamplingPlan
        -> Mosaic-RFD3
        -> provenance + required audits + report
```

Ordinary and expert modes do not use different samplers.

## 2026-08-11 verified module closeouts

### Static finite-quotient exact scaffolding: completed and GPU validated

The first finite-quotient runtime slice is closed for its declared static
scope.  The demonstrated contract is a `C4` assembly with a `C2` seed
stabilizer.  Mosaic materializes the two physical cosets
`{e,r2}` and `{r1,r3}` rather than pretending that the seed has four
independent physical copies.

Fresh frozen V100 runs `5742936` and `5742947` both completed.  Run `5742936`
is the retained golden evidence:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/public-c4-c2-quotient-orbit-canary-8_11-v2/public-c4-c2-quotient-orbit-v100-t50-s943/5742936
```

The result recovered `144/144` fixed atoms, recorded zero authoritative
runtime fixed-target error, produced two continuous chains, had zero CA
clashes, and passed exact-constraint, symmetry, continuity and compactness
audits.  This also closes the earlier audit defect that re-applied group
transforms to an already materialized quotient source and falsely reported
about `17.23 A` joint RMSD.

This closeout does **not** claim dynamic quotient mobility, partial diffusion
of a quotient seed, or mixed full-orbit and quotient-orbit components in one
native task.  Those are separate modules and must not inherit this GPU claim.

### Orbit mobility mechanics: implemented; packing-coupled design is open

The mobility runtime already supports complete-orbit rigid translation and
rotation, exact internal motif geometry, reconstruction of all symmetry
copies from one master pose, synchronized fixed targets/conditioning,
cumulative and per-step bounds, scheduled updates, and atomic simultaneous
multi-orbit acceptance or rollback.  GPU evidence exists for C3 single- and
multi-orbit execution and D3 two-orbit execution.  Axis-restricted radial and
radial/axial translation have also run successfully.

Recent runs `5742223`, `5742231` and `5742921` confirm that the mechanics
remain active: all fixed atoms and symmetry audits pass, non-zero translation
and rotation are recorded, and no CA clashes are introduced.  They do not
form the requested interface (`0/3` instances) and have three chain breaks.
Therefore the motion engine is not the remaining scientific blocker; its
current scaffold objective is not yet a joint interface-packing objective.

The next mobility module is one unified joint SE(3) controller:

```text
interface packing + linker/junction + continuity + global clash
+ orientation + shape + cavity/compactness
    -> simultaneous orbit proposals
    -> radial/axial/azimuth/tilt/twist projection
    -> exact symmetry, target and conditioning refresh
    -> atomic accept or rollback
```

Local timestep guidance is not expected to rescue a globally infeasible
initial pose.  The tetrahedral run `5742211` kept exact T symmetry and clean
chains but began with interface partners about `94.92 A` apart and formed
`0/12` interfaces.  Such cases require continuous global pose optimization
before diffusion, followed by the local controller.

### Evidence that must not be conflated

- Run `5741270` formed `3/3` C3 interface relations, but it did so with three
  chain breaks.  It is failure evidence for contact-by-tearing, not a packing
  success.
- Run `5741324` proves the pre-positioned two-seed resolver can compile,
  replay and preserve all six supplied interface instances, but it retains
  six real CA clashes.  Candidate feasibility/ranking remains open.
- A completed CIF, exact motif RMSD, clean symmetry or a passing mobility
  audit is not by itself evidence of interface packing or a valid cage.

## Required module closeout record

Every core module must end with a written closeout in this file and the
detailed history in `DEVELOPMENT_STATUS.md`.  The closeout must state:

1. the exact scope that is complete;
2. CPU/unit and GPU evidence, including run IDs and retained paths;
3. known failures and excluded scope;
4. the next independent module and its acceptance gate.

No capability is promoted merely because code exists or a structure file was
written.  A module is complete only for the scope named in its closeout.

For the three-day demonstration, product status is reported by **evidence
gates**, not completion percentages. A generated CIF proves only that RFD3
inference finished. A workflow is demo-ready only when its frozen input and
YAML are identifiable, every required audit passes, and the final morphology
matches the task that was requested.

## The two ordinary-user tasks

The public interface deliberately exposes two tasks. They share the same
compiler and sampler, but their geometry contracts are different:

1. **Preserve a supplied interface** (`task: preserve_supplied_geometry`).
   The input already contains the functional cross-fragment or cross-subunit
   interface. All fragments in one coupling group are restored by one joint
   rigid correspondence while RFD3 generates only the declared missing
   protein.
2. **Create a new symmetric interface around a motif**
   (`task: create_symmetric_interface`). The input supplies an internally
   exact motif, not the desired final interface. RFD3 grows the declared
   regions and the packing controller must create new neighbour contacts.
   Exact motif recovery is therefore necessary but does not prove that the
   generated interface, pore or global assembly shape is satisfactory.

Interface creation does not imply fixed-component motion. The independent
`fixed_arrangement` contract defaults to `locked`: the complete supplied
motif/orbit arrangement remains exact while guidance moves generated atoms
only. `fixed_arrangement: optimize_components` must be requested explicitly
to permit bounded rotation, translation or radius changes of exact rigid
components. Both modes use the same graph packing controller and audit path.
For a single-ASU motif on a symmetry stabilizer, the compiler may first
resolve one deterministic non-overlapping assembly pose; `locked` means that
resolved pose cannot move during diffusion. A usable supplied assembly pose
is never replaced by this fallback.

The templates are respectively
`examples/rfd3_mosaic/simple_interface_seed.yaml` and
`examples/rfd3_mosaic/simple_central_motif.yaml`. Expert assembly graphs are
an optional authoring surface; they do not create a third runtime path.

## Usable now

- Exact complete-orbit restoration for central motifs and cross-protomer
  interface seeds in the established Cn workflow.
- Jointly fixed fragments and independently coupled fixed components.
- Static initial poses and bounded rigid translation/rotation of motif
  orbits.
- Atomic multi-orbit updates without declaration-order dependence.
- Public `plan`, `validate`, `render`, `run`, `submit`, `status`, `report` and
  `runs` commands.
- Required motif, symmetry, continuity, clash, mobility and interface audits.
- C3 engineering evidence plus successful D3 static/dynamic and tetrahedral
  static/public-graph GPU canaries.

These capabilities are appropriate for engineering campaigns when their
declared capability level is `engineering` or `stable` and every required
audit passes.

## Implemented locally, awaiting the current LRZ gate

- `inspect`: deterministic PDB/mmCIF contact-patch detection and a short
  ordinary cage-intent YAML.
- `resolve`: the first executable ordinary architecture slice. It enumerates
  all chain-direction and adjacent-copy alternatives for one binary
  `preserve_exact` seed in a Cn ring, compiles/ranks them, writes standard
  `UserDesignSpec` YAML and requires strict replay/hash identity.
- A bounded multi-seed path-cover primitive. It enumerates deterministic
  interleaved cycles for disjoint binary seeds, uses every seed side exactly
  once and removes global rotation/reversal duplicates. Its output is
  topology-only and explicitly `executable: false`.
- An experimental **pre-positioned multi-binary Cn bridge**. For several
  disjoint binary `preserve_exact` seeds that already share one authoritative
  input coordinate frame, it binds path-cover hypotheses to component/port/
  interface/connection graphs, enumerates chemical direction, closing seam
  and Cn winding, validates the expanded interface/unit graph, and sends each
  surviving candidate through normal static ranking, strict YAML replay and
  native RFD3-adapter preflight. This is a narrow executable frontend, not
  automatic cage pose discovery, and remains `schema_only` until the LRZ and
  real-run gates below.
- Cross-seam fixed components now retain the selectors actually materialized
  in the ASU and resolve each supplied-seed member through its own relative
  native group action. A runtime prevalidation regression covers the real
  two-seed C3 seam that previously compiled but referred to a non-ASU
  copy-zero selector. The corrected snapshot still requires LRZ replay before
  this item advances in maturity.
- Packing-guidance v5 lifecycle/continuity/shape/anti-collapse changes and
  finite runtime-feature preflight. These need the complete LRZ suite and new
  50-step GPU evidence before promotion.

## Not complete

- General executable binding of several supplied interface seeds. The narrow
  pre-positioned binary Cn case is joined end to end locally, but arbitrary
  side ownership, unknown relative seed poses, homomer equivalence,
  Dn/T/O/I actions and mixed component multiplicities are not.
- Native three-or-more-participant relation/hyperedge lowering and runtime.
- Reliable packing-quality generation. Current guidance is more than a COM or
  radius pull, but repeated broad, well-oriented, all-atom interface evidence
  is still required.
- Continuous joint optimization of radius, axial offset, azimuth, tilt and
  twist for several interfaces.
- Vertex/edge/face stabilizers, cosets and mixed-multiplicity components.
- Dynamic T production evidence; O and I GPU closure; helical semantics;
  high-order local-neighbourhood GPU equivalence.
- ProteinMPNN, multimer refolding, interface-energy/designability ranking and
  a single downstream acceptance gate.
- Clean-checkout release packaging, CPU/GPU CI, schema migration and automated
  upstream Foundry compatibility replay.

## Evidence-gated maturity

| Capability | Evidence already in hand | Next gate before it is advertised |
|---|---|---|
| Exact C3 motif/supplied-seed restoration | Multiple GPU runs with complete heavy-atom recovery, exact C3 and continuity/clash audits | Keep frozen 200-step golden replays across Foundry upgrades |
| D3 static and multi-orbit mobility | Static and dynamic GPU canaries with six-copy group-action audits | Add an independent input and production-length replay |
| Static tetrahedral execution | Three independent 12-chain, two-orbit GPU runs passed exact and scaffold audits | Dynamic T and a real multi-face packing result remain separate gates |
| Static C4/C2 quotient orbit | Frozen V100 runs 5742936 and 5742947 passed exact-target, two-coset, continuity, clash and scaffold gates | Dynamic quotient mobility and mixed full/quotient tasks remain separate modules |
| New C3 generated interface | Run 5741271 completed; exact, interface-relation and scaffold audits passed | Final packing proxy and global pore/shape gate must pass on a new frozen run |
| Pre-positioned two-seed C3 resolver | Run 5741324 completed inference; post-hoc multi-chain fixed audit recovered 273/273 atoms and all 6/6 supplied interface instances | Remove the six real CA clashes and obtain a newly frozen full PASS |
| General multi-interface cage solver | Schema, graph and bounded Cn path-cover primitives only | Unknown-pose solving, multi-interface packing, stabilizer/coset and non-Cn GPU evidence |
| O, I and helical production workflows | Registry/compiler or planned pieces only | Dedicated end-to-end GPU gates; they are not demo claims |

Passing an earlier row must not be used as evidence for a later row. In
particular, exact motif RMSD does not certify interface packing, and a static
T run does not certify an arbitrary tetrahedral cage design.

## Frozen run evidence for the three-day demo

### Run 5741271: generated-interface baseline, not a supplied seed

- Run directory:
  `/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/public-contig-inferred-interface-packing-v5-8_10/public-c3-contig-inferred-interface-t50-s930/5741271`
- Public YAML:
  `experiments/lrz_public_c3_contig_inferred_interface_v100_50step.yaml`
- Source structure:
  `examples/rfd3_mosaic/inputs/Prism_C3_G2_fixed_motif.pdb`
- Scientific purpose: preserve `A12-20`, generate 35 residues from each
  terminus and test whether the generated regions form a new C3-neighbour
  interface. This was never the multi-seed 7mwr experiment.
- Proven success: 270/270 fixed heavy atoms, joint RMSD 0.0000185 A; three
  exact C3 chains; zero CA clashes and zero chain breaks; all three final
  heavy-atom interface-relation instances passed.
- Unclosed gate: `final_proxy_targets_satisfied` is `false`. A retrospective
  assembly-axis measurement gives a central radial clearance of about
  17.9 A. The run therefore proves exact/symmetric generation and final
  neighbour contacts, but not the desired compact pore or a generally solved
  packing objective. It is a partial baseline, not the final poster result.

### Run 5741324: two supplied seed patches, inference completed

- Run directory:
  `/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/ordinary-two-seed-c3-canary-8_10/ordinary-two-seed-c3-t50-s940-c3-0007/5741324`
- Selected YAML source:
  `ordinary-two-seed-c3-resolution-fixed-8_10/20260810T144753Z/selected/rank_0001_candidate_000007.yaml`
- Source structure:
  `examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb`
- Seed definition: two disjoint engineering patches cut from the same 7mwr
  A/B contact (`A186-189` with `B238-240`, and `A191-192` with `B234-235`).
  They are a resolver/runtime test; they are not claimed to be two independent
  biological interfaces.
- Proven success: RFD3 inference completed. The generalized post-hoc audit
  recovered 273/273 fixed heavy atoms across two ASU chains, with maximum
  constraint RMSD about `1.02e-5 A`; all six supplied interface instances
  pass their declared geometry contract. The PyMOL helper now uses the same
  compiler/runtime provenance rather than assuming one chain per C3 action.
- Unclosed gate: the scaffold audit detects six real CA clashes, repeated by
  C3 symmetry (A-B, C-D and E-F; minimum 0.896 A). The run therefore proves
  multi-seed compilation, exact preservation and cross-seam provenance, but
  it is not a complete design PASS. The clash gate has not been relaxed.

## Immediate three-day acceptance sequence

1. Correct the two-seed linker/endpoint clash exposed by 5741324, then run one
   newly frozen P100/V100 replay through the complete audit gate.
2. Add a final global morphology audit (central clearance/pore and outer
   extent) so a visually open assembly cannot pass only pairwise-interface
   metrics.
3. Re-run the C3 `create_symmetric_interface` canary with the morphology gate
   and require both final all-atom interface and runtime proxy targets.
4. Freeze one 200-step ordinary example for each of the two public tasks and
   generate its HTML report from the same source snapshot.
5. Keep O/I/H, unknown-pose general cage solving and sequence/refolding out of
   the three-day demo claim. Their schemas may remain visible only with their
   true capability levels.

## Pre-positioned multi-binary Cn acceptance gate

The capability may be advertised for engineering use only after all of the
following are true for one frozen source snapshot:

1. The complete LRZ unit suite passes, including deterministic path-cover,
   candidate-budget, input-contact, backbone-anchor, multiplicity and strict
   replay tests.
2. A real input plus an inspected/edited intent containing at least two
   disjoint binary `preserve_exact` seeds completes `plan -> resolve`; the
   manifest retains every
   non-equivalent candidate, reports `automatic_selection: false`, and has no
   replay failure for each advertised runnable YAML.
3. At least one selected YAML passes public `validate`, RFD3 runtime-feature
   prevalidation, expanded interface/unit-graph validation, linker/clash/group-
   closure gates and immutable structure/source hashes.
4. A newly rendered 50-step V100 or P100 run passes every required fixed-seed,
   symmetry, continuity, clash and scaffold audit. A second independent input
   or a second Cn order reproduces the result without source-specific code.
5. Documentation and reports identify the input coordinates as pre-positioned
   and never imply automatic radius/orientation/tilt optimization.

Even after this gate, the capability is not a general cage solver. Unknown
relative seed poses, three-or-more-participant relations, homomer equivalence,
stabilizer/coset orbits, T/O/I winding, dynamic multi-seed refinement and
sequence/refolding validation remain separate milestones.

The authoritative detailed history remains in `DEVELOPMENT_STATUS.md`; the
long-term architecture and release gates are in
`docs/rfd3_mosaic/RFD3_MOSAIC_PRODUCTIZATION_PLAN.md`.
