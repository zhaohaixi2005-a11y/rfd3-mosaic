# RFD3-Mosaic Current Product Status

Last updated: 2026-08-17

## Local development continuity

Cluster availability is no longer a blocker for CPU engineering. A repository-
local Python 3.12/CPU-PyTorch environment now runs the same editable Mosaic,
RFD3 and Foundry sources used by the AI-cluster snapshot. Its acceptance gate
is `make local-test`; GPU and scientific-quality claims still require frozen
cluster canaries and are not inferred from local CPU success.

## Current implementation checkpoint

### Mixed-multiplicity finite-group components: first CPU slice closed

Ordinary mode can now execute one complete, pre-positioned
oligomer--oligomer interface whose two participants have different
stabilizers. The reference tetrahedral case derives six C2 components and four
C3 components from one user-supplied C2--C3 interface used twelve times. It
expands all twelve physical interface incidences without inventing a new
interface identity, strictly reproduces the ranked structure, traverses the
native RFD3 adapter and passes AtomWorks runtime-feature prevalidation.

The selected manifest explicitly records:

- `physical_interface_count: 12`;
- component orbit multiplicities `component__c2: 6` and
  `component__c3: 4`;
- both stabilizer subgroups, coset representatives and the complete physical
  edge incidence map;
- the original supplied interface ID and `invented_interface_count: 0`.

This is a static `seed_layout: preserve_input` contract. Stabilizer-aware
unknown-pose placement, dynamic mixed-orbit motion, several independent
mixed-valency interfaces and GPU/scientific validation are not implied by the
CPU closeout.

### Packing guidance CPU closeout (GPU evidence pending)

Generated-interface guidance now uses observed `capture -> expand -> polish`
phases rather than timestep alone, protects the worst declared interface in a
multi-interface transaction, fails impossible generated-patch capacity before
sampling, and emits final heavy-atom residue-pair density, depth, fragmentation,
void and hydrophobic-composition proxies. The runtime/audit data contract is
schema v8. Detailed semantics and evidence boundaries are recorded in
`docs/rfd3_mosaic/PACKING_GUIDANCE.md`.

This closes the non-GPU implementation slice, not the scientific validation
gate. Stable broad interfaces still require repeated frozen 50-step CUDA runs.

The latest compatibility-preserving pass closes three CPU execution gaps:

- linker ranges are restored to exact symmetry-safe lengths before strict
  replay, including one common length for every `tie_group`;
- candidates outside the pose-optimization compute shortlist are no longer
  rejected when they already pass the complete compiler contract;
- ordinary diameter/cavity ranges now drive pose ranking and are checked
  again on the final RFD3 structure.

These changes retain all existing C3/D3/T, static quotient, fixed/mobile and
packing paths. The Python 3.12 CPU gate now passes 798 tests.
The three-user-seed T resolver now accepts user-authoritative finite-group
relations on polymer connections. The reference intent freezes `T:g01` and
`T:g03`, produces one topology rather than eight equivalent generator
assignments, selects that one candidate, and independently validates the
frozen YAML as 10,872 atoms, 1,812 residues, 36 runtime chains and 24 physical
polymer units with finite RFD3 runtime features. The remaining gate for this
exact candidate is 50-step CUDA execution and result audits.

The closeout order is deliberately one path, not another auxiliary sampler or
submission script:

```text
complete local CPU unit suite
-> ordinary three-seed T resolve
-> exact restored-link lengths printed and stored in the manifest
-> selected YAML strict replay + native RFD3 prevalidation
-> one frozen 50-step GPU canary
-> required supplied-interface, constraint, continuity, clash and shape audits
```

If a future resolve reports `accepted > 0` but `selected = 0`, the candidate's
`replay_error` is the remaining compiler/runtime defect and must be fixed in
the shared path. It must not be bypassed by manually editing the emitted
contig, weakening a hard audit or submitting the provisional assembly.

## Latest ordinary multi-interface evidence

Both the real three-seed C3 engineering intent and the first real three-seed
tetrahedral intent are CPU closed. C3 resolution
`three-seed-user-connected-c3-20260812T120807Z` produced 48 joint
topology/pose candidates, accepted four, selected three strict-replay YAML
files and validated rank 1 with three exact constraints, 2199 atoms, 357
residues, nine chains and finite RFD3 runtime features. The T reference then
closed the polyhedral CPU gate with two three-face units, three supplied
interface identities, four user-declared polymer connections and zero invented
interfaces. Its one resolved candidate passes strict structure replay, native
adapter construction and runtime-feature prevalidation. The next explicit
gate is one frozen 50-step GPU result whose complete required audit set passes.

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
   interface. Its two or more participants form one indivisible
   `joint_rigid` hyperedge: their internal coordinates, relative orientation
   and packing are restored together while RFD3 generates only the declared
   missing protein. Participant termini may be scaffold endpoints, but are
   never independently moved or re-paired. Different complete interface
   seeds may still receive different whole-body poses during offline assembly
   resolution; that does not alter either seed internally.
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
- A retained compatibility **pre-positioned multi-binary Cn bridge**. For several
  disjoint binary `preserve_exact` seeds that already share one authoritative
  input coordinate frame, it binds path-cover hypotheses to component/port/
  interface/connection graphs, enumerates chemical direction, closing seam
  and Cn winding, validates the expanded interface/unit graph, and sends each
  surviving candidate through normal static ranking, strict YAML replay and
  native RFD3-adapter preflight. This is the authoritative-input-layout branch
  selected by `seed_layout: preserve_input`; it is no longer the boundary of
  the unknown-pose resolver.
- An explicit unknown-relative-pose contract for supplied seed libraries.
  `seed_layout: auto` solves multiple independent source frames and preserves
  a shared input frame; `solve` forces joint pose resolution even when the
  seeds came from one PDB/mmCIF; `preserve_input` requires one meaningful
  shared frame. Every complete seed is canonicalized as one rigid interface,
  so the solver cannot alter its natural packing. Candidate metadata records
  all supplied interface IDs and an invariant requires zero invented,
  omitted or merged identities.
- Multi-seed participants may contain any number of ordered, disjoint fixed
  helices/fragments from one source chain. They remain one complete rigid
  interface face; Mosaic generates only the ordered intra-chain gaps and the
  user-declared or resolved links between seeds. Cross-chain covalent order
  is never inferred in ordinary mode.
- Deterministic global starts and continuous joint rigid-pose refinement are
  now wired into that resolver. Cn/Dn retain the ring/layer initializer;
  full-orbit T/O/I use a non-axis-biased spherical initializer. Every pose is
  evaluated on the fully expanded assembly against required interface,
  linker-contour, clash, closure and static-objective hard contracts before
  strict YAML/hash replay. Straight endpoint-chord clearance is retained as
  a soft routing/ranking signal because a flexible generated linker is not
  constrained to that chord. Explicit stabilizer/coset
  components still fail closed because they require stabilizer-aware local
  frames.
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

### Supplied multi-interface seed resolver: staged checkpoint (2026-08-12)

This module is not finished. Its current stage boundaries are:

1. **Input and identity contract — implemented locally.** Users supply every
   natural interface seed and its `auto`/exact/range physical usage.
   `seed_layout` distinguishes authoritative shared coordinates from an
   unknown relative-pose problem. Canonicalization moves a complete seed as
   one rigid hyperedge, and candidate metadata requires exactly the supplied
   interface IDs with `invented_interface_count: 0`.
2. **Topology and finite-group hypothesis generation — implemented locally.**
   The resolver enumerates polymer path/unit covers, assigns finite-group
   relations, expands the complete interface/unit graph and rejects
   disconnected lifts. A participant may now contain several ordered,
   disjoint fixed ranges from one source chain; implicit covalent order across
   unrelated source chains still fails closed. Automatic homomer equivalence
   and heteromer component ownership remain unsupported.
3. **Unknown relative pose — CPU executable for user-declared C3 and the
   reference three-seed T cage, not yet GPU hardened.**
   Deterministic Cn/Dn/T/O/I full-orbit starts and joint radius, azimuth,
   axial and three-axis rotation refinement are wired to complete-assembly
   linker/interface/clash/closure evaluation. The current bounded
   coordinate/pattern search is not yet a globally reliable cage optimizer;
   explicit stabilizer/coset component frames and cavity objectives still
   fail closed.
   The new authoritative `polymer_connections` path no longer enumerates
   alternative participant pairings. LRZ run
   `user-connected-two-seed-c3-20260812T114354Z` evaluated 32 global
   topology/pose starts, accepted four and froze four strict-replay designs
   while retaining the declared A1--B2/A2--B1 connection directions.
4. **Freeze and replay — CPU validated on real two-seed C3, three-seed C3 and
   three-seed T intents.** Surviving
   candidates are lowered to the normal expert graph, frozen as public YAML,
   reloaded and required to match structure hashes and the RFD3 adapter. The
   synchronized snapshot passed all 754 tests. Resolution directory
   `two-seed-semantic-replay-20260812T103548Z` produced 16 hypotheses, accepted
   four and validated rank 1 with 873 atoms, 153 residues, six chains and
   finite runtime features. The frozen YAML explicitly records two complete
   `joint_rigid` supplied interfaces as `preserve_input` and never emits a
   generated `contact` target.
5. **GPU and scientific quality — not yet closed for unknown-pose multi-seed
   designs.** At least one 50-step canary must pass all required audits and
   show the supplied interfaces, continuous chains and acceptable geometry.
   A 200-step run plus an independent input/symmetry reproduction is required
   before scientific validation.

The multi-participant public representation is now atomic: a supplied
interface may declare two or more ports in one `between` list. A connected
compiler-generated `contact_pairs` tree feeds binary RFD3 compatibility
features, while identity, requested use, symmetry multiplicity and audit
grouping remain attached to one hyperedge. LRZ focused tests now validate the
real PI25 three-participant C3/C3 quotient through standard-YAML standalone
compile, strict replay and native RFD3 prevalidation. The complete stabilized
trimer is represented as one preexpanded ASU with three transform-annotated
polymer paths, avoiding both symmetry loss and erroneous nine-chain
re-expansion. GPU execution is the next gate; native variadic sampler tensors
are not claimed.

For one multi-participant seed, ordinary resolution is executable only when
each participant supplies an explicit same-chain ordered fragment path. This
uses source-chain continuity as user-provided topology and generates only its
missing intervals. Isolated interface sides do not authorize a guessed
covalent graph. The portable real-structure gate is
`lrz_simple_three_participant_c3_quotient_v100_50step_intent.yaml`, derived
without coordinate changes from the PI25 C3 trimer contact patches; LRZ
strict replay now passes and GPU closeout is pending.

The immediate order is therefore: run one newly frozen 50-step canary from
the validated rank-1 YAML, require complete supplied-interface, continuity,
clash and scaffold audits, and only then continue with stabilizer-aware
placement and component-equivalence inference. The CPU stages before that GPU
gate are now complete.

- General executable binding of several supplied interface seeds is CPU closed
  for user-declared C3 examples and the reference three-seed T cage. The T
  path now completes symmetry-safe linker restoration, strict replay and
  native RFD3 prevalidation. Unknown-relative full-orbit Cn/Dn/T/O/I starts
  and joint pose optimization are implemented; Dn/O/I still need equivalent
  real-input replay and GPU evidence. Homomer equivalence, automatic heteromer ownership and
  stabilizer-aware unknown-pose placement remain open and fail closed.
- The two primary workflows remain intentionally distinct: supplied
  multi-fragment interfaces are preserved exactly, while motif-only
  `create_symmetric_interface` runs create new packing contacts. A hybrid run
  that preserves some supplied interfaces and simultaneously creates other
  new interface types in the same cage is not yet an ordinary-user contract;
  it requires a mixed preserve/contact assembly task and joint GPU evidence.
- Native variadic sampler tensors for three-or-more-participant relations.
  Public hyperedge schema, compatibility lowering, physical multiplicity and
  grouped audit are implemented locally; LRZ full-suite/strict replay and a
  representative GPU closeout remain required.
- Reliable packing-quality generation. Current guidance is more than a COM or
  radius pull, but repeated broad, well-oriented, all-atom interface evidence
  is still required.
- Continuous joint optimization already searches component-wide translation
  and rotation for several interfaces. Explicit stabilizer-frame placement
  and systematic calibration of radius/axial/azimuth/tilt/twist schedules
  across T/O/I remain open.
- General vertex/edge/face combinations, several mixed-multiplicity interface
  types and unknown-pose stabilizer-frame optimization. The first static,
  pre-positioned T C2--C3 incidence is CPU closed.
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
| General multi-interface cage solver | User-only seed identity invariant, hyperedge/path-cover topology, atomic public multi-participant relations, binary compatibility lowering, finite-group relations, explicit seed-layout policy, global full-orbit Cn/Dn/T/O/I starts, continuous pose refinement, earlier full 754-test LRZ suite and real two-seed C3 strict replay pass | Re-run the expanded LRZ suite, representative full PASS GPU result, then stabilizer-aware poses and native variadic sampler tensors |
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
5. Documentation and reports record the resolved `seed_layout` contract. For
   `preserve_input`, the shared input coordinates are authoritative. For
   `solve`, the report records canonicalization, initialization, continuous
   pose optimization and the frozen replay candidate without implying that an
   unvalidated candidate is a successful cage.

Even after this gate, the capability is not yet a universally validated cage
solver. Full-orbit unknown relative seed poses are implemented locally, but
three-or-more-participant native runtime, homomer/heteromer equivalence,
stabilizer/coset component orbits, broad T/O/I GPU evidence, dynamic multi-seed
refinement and sequence/refolding validation remain separate milestones.

The authoritative detailed history remains in `DEVELOPMENT_STATUS.md`; the
long-term architecture and release gates are in
`docs/rfd3_mosaic/RFD3_MOSAIC_PRODUCTIZATION_PLAN.md`.
