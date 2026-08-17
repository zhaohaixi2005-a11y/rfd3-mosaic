# RFD3 Mosaic Development Status

Last updated: 2026-08-17

This file is the persistent project memory for resuming development after a
new login or a new Codex session. Update it whenever a milestone changes.

## 2026-08-17 local CPU development environment

AI-cluster downtime no longer blocks compiler/runtime development. The
repository now has a reproducible `.venv-local` path built by
`scripts/rfd3_mosaic/setup_local_cpu_dev.sh`: standalone CPython 3.12,
CPU-only PyTorch, editable RFD3-Mosaic/Foundry, AtomWorks, Biotite, Pydantic,
pytest and ruff. `activate_local_dev.sh` fixes the local desktop's invalid
`DEBUG=release` value to the boolean `false` expected by Foundry and exports
the two editable source roots. `make local-test` is the single complete CPU
gate. No checkpoint is downloaded and existing LRZ YAML/profiles are unchanged.

This environment can close schema, compiler, resolver, strict replay, RFD3
feature construction and audits. It cannot replace CUDA execution or
scientific packing validation; those remain queued canaries after cluster
recovery.

A separate short-lived laboratory workstation now provides an additional
compatibility buffer while the AI cluster is unavailable.  A repository-local
Python 3.12 environment with PyTorch 2.7.1/CUDA 12.6 on an 8 GB RTX 3070 passes
the same complete 787-test CPU gate.  An externally supplied
`rfd3_latest.ckpt` is readable and its two available copies have identical
SHA256
`9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`.
This workstation is not a replacement execution backend: temporary inputs,
profiles and results stay outside the tracked repository, existing LRZ
profiles/YAML remain unchanged, and no GPU evidence is claimed while another
user occupies most of the device memory.  When the GPU becomes free, the
first gate is one batch-1, low-memory C3 10-step compatibility smoke before
any 50-step or polyhedral run.

The first complete local gate passed **786 tests** before the extended-chain
runtime correction and **787 tests** afterward. Local execution also closed
the real three-user-seed T path that previously stopped after static
acceptance: eight finite-group candidates compile, one frozen candidate is
selected, and the selected YAML independently validates as 10,416 atoms,
1,752 residues and 24 physical polymer chains with finite RFD3 runtime
features. RFD3's empty-selection construction now accepts extended mmCIF
chain identifiers, and strict replay now performs actual RFD3 feature
prevalidation before publishing any YAML under `selected/`.

## 2026-08-12 compatibility-preserving executable-candidate closeout

This pass extends the existing compiler/runtime spine; it does not replace
any previously validated fixed, mobile, quotient, packing or resolver path.
Legacy designs without assembly connections or shape intent retain their
previous lowering and sampler settings.

The candidate boundary now performs deterministic feasibility restoration
before a YAML is published. Every generated connection range is evaluated
over all physical symmetry instances and frozen to one exact contour-safe
length. Connections sharing `tie_group` are solved jointly over the
intersection of their user-authorized ranges and receive the same exact
length. Endpoint identity, supplied-interface geometry, component ownership,
usage, symmetry and copy relations are guarded as immutable invariants. A
restored candidate is recompiled and the restored structure—not its
provisional ranged predecessor—is the one ranked and hash-replayed. This
directly addresses the T three-seed failure in which static compilation
accepted a 10--45 range but the adapter independently chose length 27 for a
physical instance requiring a longer contour.

Ordinary `goal.diameter_angstrom` and `goal.cavity_diameter_angstrom` now
lower to normal required Assembly IR objectives, participate in full-assembly
pose optimization, remain in frozen RFD3 provenance, and are evaluated again
against final-output CA morphology by `scaffold_validity_audit.json`.
`preferences.cavity` remains a soft compact/auto/open initialization bias;
an explicit numeric range is a hard output contract. Run reports show both
requested and observed values.

The pose-optimization shortlist is no longer a hidden rejection filter.
Candidates outside the compute shortlist may still be selected when their
fully expanded compiler and strict-replay contracts already pass. Straight
linker-chord obstruction remains a soft routing preference; actual fixed
clashes, insufficient maximum contour, required supplied-interface failure,
required size failure and replay mismatch remain hard. The next evidence gate
is the synchronized LRZ full suite followed by a fresh three-seed T resolve;
only then should a selected 50-step GPU canary be submitted.

`resolve` now prints every range-to-exact linker restoration beside the
selected design and records the complete decision in
`resolution_manifest.json` (configured range, worst physical requirement,
selected length, policy, tie group and physical instance IDs). This makes the
adapter decision inspectable and prevents a repaired candidate from looking
like an unexplained manual YAML edit.

## 2026-08-12 three supplied interfaces CPU replay

The ordinary user-declared connectivity path is no longer limited to two
supplied interface identities.  LRZ resolution
`three-seed-user-connected-c3-20260812T120807Z` consumed three complete
two-participant seeds, retained the three user-declared cross-seed polymer
connections, evaluated 48 topology/pose candidates, accepted four and wrote
three strict-replay YAML files.  Rank 1 is
`selected/rank_0001_candidate_000032.yaml`.  Public validation passed three
exact constraints, geometry with 2199 atoms/357 residues/nine chains, and
finite RFD3 runtime features.  No interface identity was invented.

This closes the three-seed C3 CPU integration gate, not a tetrahedral cage
claim.  The next CPU gate is three supplied interface identities arranged as
two user-declared three-face protein units under T.  Its executable intent is
`experiments/lrz_simple_three_seed_t_user_connections_v100_50step_intent.yaml`;
T must assign two independent finite-group generators, expand 24 physical
polymer units, preserve all three natural interfaces and survive strict
replay before a GPU job is authorized.

## 2026-08-12 supplied-interface semantic boundary

A user-supplied interface is one complete physical contact face with two or
more participants. It is not a bag of independently reusable "sides". The
ordinary resolver now emits `task: preserve_supplied_geometry` explicitly,
stores all participants of each supplied identity in one `joint_rigid`
component, retains `relation.mode: preserve_input`, and rejects either a
generated `contact` target or independently rigid components under that task.

Scaffold compilation may reference a participant's real N/C terminus, but it
must not detach, independently move, re-pair or invent a participant-level
interface. Whole supplied hyperedges may be translated/rotated during global
assembly pose solving while their complete internal geometry remains exact.
This workflow is distinct from `task: create_symmetric_interface`, where the
input is a central motif without the desired interface and packing guidance
must create contacts in generated regions.

The public assembly graph now also accepts one variadic supplied-interface
declaration with `between: [port_A, port_C, port_D, ...]`. Resolver-generated
`contact_pairs` must be unique, contained in `between` and connect every
participant. Lowering creates compatibility member constraints for the
current binary RFD3 tensors, but preserves one hyperedge ID and counts physical
usage once. Multi-participant `mode: contact` fails closed: this slice is for
preserving a supplied face, not inventing a cooperative generated interface.

The ordinary resolver now has one additional fail-closed executable case for
a single cooperative supplied hyperedge. If every participant selector
contains two or more disjoint fragments on one source chain, those source
chain orders define the covalent scaffold paths exactly. Mosaic generates
only the missing interval within each participant; it never connects one
interface participant to another. A single hyperedge made only of isolated
fragments remains ambiguous and still requires another supplied seed or
expert `connections`. The real PI25 C3 three-participant canary is the LRZ
execution gate for this slice. Its focused resolver test now passes on LRZ:
one cooperative seed is recognized as one C3/C3 quotient hyperedge, its three
authoritative same-chain paths compile, and the resulting standard public
YAML survives strict standalone replay. Internal binary compatibility members
use distinct runtime port aliases while retaining one public hyperedge
identity; the ranker also treats absent inter-group atom pairs in a single
`joint_rigid` component as a valid `None` measurement. The native adapter now
executes the mathematically valid C3/C3 one-coset case as a preexpanded
stabilized ASU: the three authoritative paths are frozen onto `C3:e`,
`C3:r1` and `C3:r2`, materialized exactly once, and annotated as one RFD3
symmetry entity rather than incorrectly expanded into nine chains. Focused
LRZ adapter and prevalidation tests pass with three chains and three runtime
transforms. A 50-step GPU result remains the separate pending gate.

Local `py_compile` and `git diff --check` pass. The synchronized LRZ snapshot
then passed the complete suite (`754` tests). A real two-seed C3 semantic
replay at
`/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/two-seed-semantic-replay-20260812T103548Z`
enumerated 16 candidates, accepted four and froze four replayable YAML files.
Rank 1 is `selected/rank_0001_candidate_000012.yaml`; it explicitly contains
`task: preserve_supplied_geometry`, two `joint_rigid` components and two
`preserve_input` relations. Public validation passed geometry (873 atoms, 153
residues and six chains) and finite RFD3 runtime features. This closes the
supplied-interface identity and CPU compile/replay gate; representative GPU
and final-scaffold quality remain separate gates.

## 2026-08-12 unknown-relative-pose multi-seed resolver checkpoint

The user-authoritative connectivity slice is now CPU closed for the real C3
engineering input. Ordinary intents may declare `polymer_connections` using
`interface.participant` endpoints. Once present, the resolver does not
enumerate alternative participant pairings and does not reverse the declared
chain directions; it only assigns the finite-group seam/relation and solves
component poses. Independent seed materialization remaps those endpoints into
canonical chain IDs together with their complete rigid interface geometry.

LRZ resolution
`user-connected-two-seed-c3-20260812T114354Z` discarded the mutual input pose,
evaluated 32 topology/pose states, accepted four and selected four strictly
replayable public YAML files. Rank 1 is
`selected/rank_0001_candidate_000016.yaml`; its topology is explicitly marked
`declared`, contains six physical polymer units, and preserves the requested
cross-seed A1--B2/A2--B1 connectivity. This is CPU executable evidence, not
yet GPU/scientific-quality evidence.

The ordinary multi-interface frontend now distinguishes supplied interface
geometry from arbitrary file placement. A new `seed_layout` intent field has
three explicit meanings:

- `auto`: solve relative pose when seeds use different source files; preserve
  the overall pose when all seeds share one input structure;
- `solve`: canonicalize every complete supplied seed and jointly solve their
  relative placement even when they came from one PDB/mmCIF;
- `preserve_input`: retain one shared input frame and reject multiple unrelated
  source frames.

This is not an interface-discovery mode. The user supplies every interface
identity and physical usage. Materialization preserves the full intra-seed
participant geometry; resolver metadata records all supplied IDs and asserts
that emitted hyperedges are exactly the supplied set. Mosaic may enumerate
polymer paths and finite group relations, but may not invent a new noncovalent
seed combination.

One supplied interface participant may now contain several ordered,
non-overlapping residue ranges from the same source polymer chain (for
example, two or three helices forming one interface face). Independent-file
materialization keeps all ranges in one canonical participant, the complete
multi-fragment interface remains one `joint_rigid` hyperedge, internal gaps
become ordered generated links, and cross-seed paths bind only the outer N/C
fragments. Ordinary mode still refuses to infer covalent links across
different source chains; that topology must be user-declared in expert mode.

The execution path is now wired as:

```text
supplied rigid interface hyperedges
-> contact/connectivity validation
-> canonical independent seed frames when requested
-> polymer unit/path-cover enumeration
-> finite-group relation assignment and expanded-graph validation
-> deterministic global full-orbit Cn/Dn/T/O/I starts
-> joint radius/azimuth/axial/rotation pattern search
-> interface/linker-contour/clash/closure hard contracts
-> straight-chord clearance, terminal tangent and packing soft ranking
-> frozen UserDesignSpec
-> strict YAML/hash replay and RFD3 adapter prevalidation
```

Cn/Dn preserve their existing ring/layer initializer. T/O/I full-orbit
components use a deterministic Fibonacci-sphere family that avoids named
symmetry axes. Components carrying explicit stabilizer/coset actions fail
closed because their unknown-pose initialization needs stabilizer-aware local
frames; a generic full-orbit start would be geometrically false.

Local evidence: `py_compile`, `compileall` and `git diff --check` pass. New
tests cover shared-file `solve`, shared-file `auto`, invalid multi-file
`preserve_input`, single-seed misuse, exact supplied interface identity and
polyhedral initialization. The full LRZ suite and the real C3 strict replay
described above now pass. Representative GPU evidence is still required
before this module is promoted from CPU validated to GPU validated.

## 2026-08-11 authoritative engineering checkpoint

### Module closeout protocol

After each core module, record the completed contract, unit/CPU evidence, GPU
run IDs and paths, failures/exclusions, and the next acceptance gate.  Do not
infer scientific completion from a generated CIF, a single passing audit or
an implementation-only change.  `docs/rfd3_mosaic/CURRENT_PRODUCT_STATUS.md`
is the concise current report; this file retains the detailed development
memory.

### Static finite-quotient exact scaffolding closed

The first static quotient slice is complete for `C4/C2`: C4 is the full
group, the supplied seed has stabilizer `{e,r2}`, and the physical orbit is
the two cosets `{e,r2}` and `{r1,r3}`.  Fresh frozen V100 jobs `5742936` and
`5742947` completed.  Job `5742936` is retained as the golden run at:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/public-c4-c2-quotient-orbit-canary-8_11-v2/public-c4-c2-quotient-orbit-v100-t50-s943/5742936
```

It recovered 144/144 fixed atoms with authoritative runtime target RMSD and
maximum error equal to zero, generated two continuous chains, introduced no
CA clashes, and passed constraint-orbit and scaffold-validity gates.  The
runtime now records direct final fixed-target residuals.  The audit consumes
that authoritative contract and retains transform-reconstructed values only
as legacy diagnostics.  The compactness policy derives its lower bound from
the immutable fixed geometry without weakening clash rejection.

Excluded from this closeout: dynamic quotient mobility, quotient partial
diffusion, and mixing full and quotient physical orbits in one native task.

### Mobility implementation and evidence checkpoint

The implemented engine provides complete-orbit rigid SE(3) motion; exact
per-copy internal motif geometry; reconstruction from one master pose;
synchronized symmetry copies, fixed targets and conditioning; per-step and
cumulative bounds; schedules; radial and radial/axial translation subspaces;
and snapshot-synchronous multi-orbit proposals with atomic joint acceptance
or rollback.  Validated GPU examples include jobs `5733341`, `5733680`,
`5733718`, `5733719`, `5733788` and `5733972` across C3 and D3.

The executable public declaration is currently
`fixed_xyz.pose.mode: bounded_mobile`.  Standalone `kind: bounded_mobile` and
`kind: cylindrical` declarations still stop at schema/plan because the first
public backend lowering path only admits `fixed_xyz`.  Internal `tilt_only`
metadata also exists but is rejected by the runtime controller and is not a
public `FixedComponentPoseSpec` option.  Explicit azimuth-, twist-, tilt- and
radial-plus-rotation-only subspaces therefore remain unimplemented.  Dynamic
mobility currently requires one design per process, low-memory/chunked pair
conditioning, exact orbit-average state and coupled symmetry noise.  The
local-neighbourhood execution backend rejects dynamic mobility.  Cn and Dn
have GPU evidence; dynamic T/O/I/H and dynamic quotient execution do not.

The older C5 pilot scripts and configuration under
`scripts/rfd3_mosaic/lhd101_c5_mobile_*` and
`configs/rfd3_mosaic/experimental/lhd101_c5_mobile.yaml` are retained for
historical reproduction.  They are not the recommended public product entry
and must not become another parallel sampler workflow.

Runs `5742223`, `5742231` and `5742921` further prove non-zero bounded motion
and exact constraints/symmetry without CA clashes, but all fail the requested
generated interface (`0/3`) and contain three chain breaks.  This separates
the completed motion mechanics from the unfinished scientific controller.

The main design defect is architectural: motif mobility optimizes junction,
CA-clash, tilt and pose-prior terms, while generated-interface packing is a
separate post-Euler patch controller.  They can therefore make individually
valid moves that do not jointly improve the complete assembly.  The next
core implementation must combine packing, linker/junction, continuity,
global clash, orientation, shape and cavity/compactness into one simultaneous
multi-orbit SE(3) proposal, project it into the declared motion subspaces,
refresh exact symmetry/targets/conditioning and accept or roll back the whole
update atomically.

Run `5742211` demonstrates a separate global-pose blocker: exact tetrahedral
runtime and clean chains passed, but the selected interface partners remained
about 94.92 A apart and all 12 interfaces failed.  A local trust-region
controller cannot repair this.  Global continuous radius/rotation/tilt/
translation and neighbour-relation optimization must precede timestep-local
packing refinement for such candidates.

Do not use run `5741270` as packing evidence: it formed 3/3 relations while
breaking all three chains.  Do not use `5741324` as a complete cage PASS: it
preserved all six supplied interface instances but retained six real CA
clashes.  Both runs are valuable failure evidence and define acceptance
tests for the unified controller and candidate feasibility gates.

## Current software maturity snapshot

RFD3-Mosaic is currently a research-grade alpha with a strong execution core,
not yet a finished general protein-cage product.

## Operational baseline: use the software now

Development no longer assumes that every roadmap feature must be finished
before Mosaic can be used. The project now has an operational v0.1 baseline
that is protected while experimental capabilities advance independently.

The recommended routine workflow today is:

```text
one input PDB/mmCIF
  -> simple YAML (input, symmetry, generation, fixed selectors, lengths)
  -> rfd3-mosaic plan
  -> rfd3-mosaic validate
  -> rfd3-mosaic run
  -> rfd3-mosaic status/report
  -> inspect only candidates whose required audits pass
```

Use now as the default engineering path:

- supplied-interface scaffolding: provide all fixed interface fragments in
  one `coupling_group`, generate the connecting region and preserve the whole
  interface orbit exactly;
- fixed central-motif scaffolding in Cn: preserve the complete motif orbit and
  generate N/C extensions; new-interface quality remains a ranked design
  outcome rather than a one-shot guarantee;
- C3 for the most extensively exercised exact runtime, with audited C5--C7
  supplied-interface campaigns available as broader engineering evidence;
- 50 steps for a canary and 200 steps for a real design campaign;
- V100/P100 according to queue availability, with `validate` before every
  submission and all declared audits treated as mandatory.

Use only as opt-in engineering/experimental functionality:

- bounded motif mobility and multiple independently moving orbits;
- D3 and T assembly graphs;
- automatic graph pose search and newly generated interface packing guidance;
- local-neighbourhood execution for high-order symmetry.

Do not yet treat as routine product capabilities:

- general O/I/helical generation, mixed stabilizer/coset cage components;
- automatic recovery of a globally optimal cage architecture from an
  arbitrary motif;
- sequence design, multimer refolding or experimental designability claims.

The release policy is therefore additive. A feature progresses through
`planned -> schema_only -> cpu_validated -> gpu_canary -> engineering ->
stable -> scientifically_validated`. Anything below `engineering` remains
opt-in and may not change the protected operational defaults without a golden
regression. This lets normal design campaigns run now while cage search,
packing and downstream validation improve incrementally.

The v0.1 baseline is considered releasable when installation instructions,
one simple supplied-interface example, one simple central-motif example,
full CPU tests, two GPU golden replays, provenance capture and portable HTML
reports all pass from a clean checkout. O/I/H and sequence design are not v0.1
release blockers.

Current delivery view:

| Release layer | Current state | Remaining gate |
|---|---|---|
| v0.1 supported exact runtime | Core implementation complete for audited C3 central/interface scope | preserve the existing golden behavior |
| v0.1 product/release hardening | 80--85% | clean-checkout installation, freeze two GPU golden replays, concise tutorial |
| v0.2 generated-interface packing | 45--55% | packing-v4 LRZ suite, repeated 50/200-step GPU evidence, all-atom output metrics |
| v0.3 multi-face cage graphs | 35--45% | continuous joint pose refinement, three-plus-interface GPU campaigns, stabilizer/coset IR |
| v0.4 O/I/high-order/helical | 20--30% | O/I GPU closure, bounded neighbourhood equivalence, separate helical semantics |
| v0.5 sequence/fold validation | 5--10% | ProteinMPNN/refolding adapters, provenance, ranking and failure gates |

### Queued-run input freezing correction (2026-08-10)

Jobs submitted before this correction could fail before RFD3 startup when the
live public-design YAML was edited while the job waited in Slurm.  The source
code already ran from `source_snapshot.tar.gz`, but the worker still reloaded
the mutable authoring YAML from the shared project checkout.  LRZ job
`5736812` exposed this contradiction as a size-identity error (`1537 != 1562`);
it is not a diffusion, GPU or structure-quality failure.

Rendered public runs now contain a normalized immutable
`public_user_design.yaml`, and `resolved_config.yaml` points to that submission
artifact.  Experiment/profile/compatibility files are retained as authoring
provenance but are not executable dependencies after their values have been
resolved.  The frozen public design, structure inputs, source archive and
checkpoint remain fail-closed hashed runtime dependencies.  A regression test
requires edits to the original YAML/profile to leave a queued render valid and
requires tampering with the frozen design to fail.  This correction is locally
syntax-validated and still requires the complete LRZ unit suite plus one newly
rendered canary before promotion.

The first newly frozen P100 retry (`5740944`) passed RFD3 input construction
but exposed a separate runtime-feature contract error before its first
diffusion step.  Optional assembly-interface COM distance targets had used
`NaN` as an absent-value sentinel; Foundry correctly rejects any non-finite
model feature.  The transform now emits finite zero placeholders plus an
explicit `assembly_interface_has_distance_target` mask, and graph guidance
consults that mask.  Prevalidation now also rejects any non-finite floating
runtime feature.  This is a backend-independent adapter bug, not a P100
kernel/precision failure; all submissions rendered with the older transform
must be replaced after LRZ unit validation.

Public `validate`, `render`, `run` and `submit` preflight now use the complete
production path (`UserDesignSpec -> AssemblySpecification -> RFD3 adapter ->
AddSymmetryFeats`) rather than stopping after standalone static geometry.
Validation therefore checks the exact runtime feature dictionary, including
its finite-value contract, on CPU before a scheduler slot is consumed.  The
CLI reports a separate `RFD3 input: PASSED` line so users can distinguish
schema/static-assembly success from backend-input construction success.  This
closes the validation hole that allowed job `5740944` to pass public validate
while carrying a NaN runtime feature.

### Ordinary cage inspection hardening (current local slice)

The ordinary `inspect` frontend now separates residue-disconnected contact
patches on the same pair of chains instead of merging them into one false
interface seed. It also records the observed chain-to-interface incidence
graph and detected port count, and accepts subunit, outer-diameter and cavity-
diameter ranges directly on the command line. These numeric ranges now lower
to required Assembly IR objectives, participate in complete-assembly pose
ranking and are rechecked on final RFD3 CA morphology. The qualitative
`preferences.cavity` field remains a soft initializer/ranking preference;
only explicit numeric ranges are hard output contracts.

Simple cage plans now report `resolution_stage: intent` and
`executable: false` in machine-readable output, plus the variables still
blocking a frozen assembly. This removes an important product ambiguity:
successful `inspect/validate` proves the input contacts and request are
self-consistent, not that component ownership, scaffold paths, symmetry
neighbours and continuous poses have already been solved. The current slice
is syntax-validated locally and requires the full LRZ unit suite before its
capability evidence is updated.

The supported v0.1 runtime is not "15--20% scientifically missing". Its
remaining percentage is release engineering around an already demonstrated
scope. The later percentages describe additional product layers and are not
discounts applied to the existing successful structures. The immediate
product target is v0.1; v0.2 work may continue in parallel but must not delay
or destabilize the exact-scaffolding release.

Validated or engineering-ready foundations:

- one public YAML/CLI path lowering through `UserDesignSpec` into a common
  `AssemblySpecification`, RFD3 backend, provenance record and audit gate;
- exact complete-orbit restoration for central motifs and cross-protomer
  interface seeds;
- grouped versus independently coupled fixed fragments;
- bounded rigid motif translation/rotation and atomic multi-orbit updates;
- Cn, Dn and tetrahedral declared group actions, including successful C3, D3
  and static T GPU canaries;
- public run/submit/status/report/index operations and fixed-orbit PyMOL
  alignment;
- retained ordinary and expert authoring levels. Executable compact motif-
  scaffolding YAML and explicit components/ports/interfaces/connections both
  use the same compiler and sampler; the broader ordinary cage intent is a
  pre-execution discovery document until its resolver freezes that same graph.

Active experimental core:

- `rfd3-mosaic inspect` now performs deterministic PDB/mmCIF chain/contact
  analysis, separates residue-disconnected patches on the same chain pair,
  reports observed component/port incidence and emits a short ordinary-user
  cage intent. The user supplies the
  input, broad cage goal and per-interface usage (`auto`, exact or range);
  detected interface selectors and inspection thresholds are recorded for
  reproducible validation. The intent itself is never submitted directly. A
  user must first choose one standard YAML produced by a supported resolver.
  The resolver now supports single-seed Cn plus supplied multi-seed layouts in
  `preserve_input` and `solve` modes. The latter canonicalizes every supplied
  seed independently and jointly solves candidate topology, symmetry relation
  and continuous component poses before freezing a strict-replay YAML. This is
  implemented locally but remains behind the LRZ full-suite/replay/GPU gate;
- the ordinary intent schema represents an interface as two or more named
  participants rather than a hard-coded pair. Exact selector validation uses
  a connected participant contact graph, allowing cooperative A-C-D-style
  sites without demanding an all-to-all clique. The public expert graph now
  preserves that variadic relation as one atomic hyperedge. Compiler lowering
  emits a connected binary compatibility tree for the current left/right RFD3
  tensors while physical usage, provenance and audit remain hyperedge-level;
- explicit assembly-graph interfaces now carry a physical multiplicity
  contract. After group expansion the compiler counts unique physical edge
  instances and rejects an expert symmetry/copy relation that cannot satisfy
  the requested usage;
- ordinary intents now receive a conservative Cn/Dn/T/O/I generic full-orbit
  compatibility report. It filters group orders using interface usage and
  optional homomeric subunit bounds, but explicitly leaves polymer ownership,
  connection order, neighbour transforms, continuous pose and mixed
  stabilizer/coset multiplicities unresolved;
- ordinary resolution retains the narrow single-binary Cn frontend and now has
  a general supplied multi-seed frontend. `seed_layout: preserve_input` keeps a
  shared reference layout; `seed_layout: solve` deliberately discards
  cross-seed placement while preserving each seed's internal geometry. It
  enumerates chemical path covers and group relations, initializes full-orbit
  Cn/Dn/T/O/I placements, jointly refines radius/azimuth/axial/rotation, runs
  hard linker/clash/closure checks, and freezes ranked candidates through the
  common compiler/replay path. It never invents an interface identity and does
  not silently select a candidate. Stabilizer/coset component orbits still fail
  closed until their geometry-aware pose solver is implemented;

- simple terminal-contig designs now infer that generated symmetry-neighbour
  regions must form an interface, choose a concrete nonidentity neighbour and
  automatically move a motif away from a degenerate symmetry stabilizer;
- output-interface guidance derives its own balanced residue coverage and
  contiguous-patch targets, so ordinary users do not specify a contact count
  or packing location;
- final audits independently check generated-interface coverage rather than
  allowing exact fixed-motif restoration to hide a missing new interface.
- packing-guidance v4 augments attraction/coverage/continuity with an
  orientation proxy, contact-depth uniformity, local C-alpha geometry
  protection and smooth worst-interface pressure. These terms are joint over
  compiler-declared graph edges and remain inside the unified sampler.

Major unfinished product/science layers:

- single-input multi-interface-seed cages: the public graph can separately
  name any number of supplied interface identities and polymer units carrying
  arbitrary ordered subsets such as `A-C-D` or `B-C-D`. A local topology
  analyzer now checks the general interface--unit incidence
  cycle and records pair/unit ownership, but explicit input-copy-to-group-action
  assignment, strict public validation and a three-pair GPU canary remain
  unfinished. Ordered same-chain paths are supported separately and are not
  the flagship cage topology;
- the general ordinary architecture resolver can now optimize unknown
  full-orbit seed poses after the user supplies every interface identity and
  usage. Remaining inverse-design gaps are automatic component-equivalence
  inference for heteromers, geometry-aware stabilizer/coset assignments,
  native higher-participant hyperedge runtime, and broad LRZ/GPU evidence;
- repeated GPU evidence that automatic guidance produces broad, well-oriented
  and sequence-designable new interfaces, rather than merely geometric
  contact;
- true solvent-accessible area, side-chain-aware shape complementarity,
  cavity/porosity and exposed-hydrophobe terms in the joint packing objective;
- calibration and GPU validation of joint continuous pose optimization for
  several seed/interface orbits;
- vertex/edge/face stabilizers, cosets and mixed-multiplicity cage components;
- dynamic T, O/I GPU closure, high-order local-neighbourhood execution and
  helical symmetry;
- integrated sequence design, multimer refolding and final candidate ranking;
- packaged releases, GPU CI, schema migration and upstream Foundry upgrade
  automation.

Approximate maturity is therefore: exact symmetry/constraint runtime 80--85%,
public compiler/execution/audit spine 70--80%, automatic new-interface packing
40--50%, general cage architecture solving 30--40%, and complete experimental
design pipeline 25--35%. These ranges describe engineering coverage, not
scientific success rates.

The latest simple/expert authoring slice and subsequent graph-runtime fixes
have passed the complete LRZ unit suite. Packing-guidance v4 is a newer local
change and must pass that suite plus a 50-step V100/P100 canary before its
runtime diagnostics schema is promoted.

## Ultimate multi-interface-seed cage acceptance target

One flagship product goal is not tied to a literal count such as two, three or
eight. It is an **N-interface-relation cage** contract: one input PDB/mmCIF may
contain any number of fixed interface identities, and each relation contains
two or more participants. The original two-protomer interface seed is the
common binary special case; cooperative A/C/D-like sites are native
hyperedges, not three unrelated successful pairs. A protein unit may carry
ports from any number of different relations. The non-covalent relation
hypergraph preserves each supplied interface, while the independent covalent
scaffold graph creates arbitrary ordered polymer units. Component/interface
ownership is not one-to-one: heterogeneous component types may expose
different interface sets and one relation type may have many physical
instances under symmetry. Mosaic must either scaffold those two graphs into
one compatible symmetric assembly or return a precise infeasibility
explanation. No implementation layer may contain a scientific maximum number
of relations or participants; practical limits must arise only from validated
topology, memory and compute budgets and must be reported explicitly.

This target is reached only when all of the following are true:

1. The public input layer binds all selected chains/residue ranges from one
   structure with unambiguous component, interface-pair and polymer-unit
   ownership. The original two-helix input is one pair, not one chain unit;
   additional pairs use the same representation rather than a new script.
2. Each supplied interface is represented independently from generated
   polymer paths. The interface relation is never
   inferred from covalent contig order and the contig compiler never consumes
   or rewrites the interface edge.
3. If input chains already represent different symmetry copies, the compiler
   records their group-transform identities and must not copy the complete
   multi-chain seed again as though every input chain were an independent ASU.
4. The generated-chain compiler discovers polymer units from scaffold edges,
   including units carrying three or more different interfaces, and emits
   every unit once. Ordered path is a native property of every unit, not a
   hard-coded two-interface or adjacent-pair rule.
5. The compiler assigns components to valid group orbits, including
   stabilizer/coset semantics when future vertex, edge and face objects have
   different multiplicities.
6. A joint pose/topology solver chooses or validates symmetry neighbours,
   radius, rotation, tilt, axial offset and chain connections, and rejects
   contradictory seeds through group-closure, clash and linker-feasibility
   diagnostics. Static enumeration alone is insufficient.
7. The unified timestep runtime restores every fixed seed relation exactly,
   or moves every explicitly bounded orbit atomically, without declaration-
   order dependence. The implementation is list-based, but GPU evidence is
   currently limited to two independently controlled C3/D3 orbits.
8. Independent final audits report atom completeness and fitted relative-pose
   error for every seed, complete-assembly symmetry, continuity, clashes and
   interface quality. A single aggregate RMSD cannot hide one failed edge.
9. A real campaign with at least three distinct supplied interface classes is
   replayable at 50 and 200 steps; the later full cage gate must also include
   sequence design and multimer refolding before experimental-design claims.

The already implemented exact projector, two-edge-type assembly graph and
atomic multi-orbit transaction are the correct foundation for this target.
They do not need a second sampler. Ordered non-branching multi-link path
lowering is a useful local addition awaiting LRZ validation, but it is not the
flagship blocker. The immediate missing work is cross-pair unit derivation,
explicit input-chain/group-action ownership, per-interface-pair audits and
broader N-pair GPU validation. Consequently
`multi_chain_interface_seed_cage` remains an implementation-stage capability
until strict replay and GPU evidence pass. The earlier pre-positioned graph is
retained as a compatibility/evidence path, not as the limit of the resolver.

## 2026-08-07 single-input ordered seed-path compiler slice

The native adapter no longer assumes that every generated scaffold link owns
two disjoint fixed endpoints. Copy-zero continuous links are assembled into
directed open N-to-C paths. For `A -> B -> C`, the emitted RFD3 contig is
`A,linker,B,linker,C`; B is selected, fixed and materialized once. Link order
is determined by endpoint connectivity rather than YAML or identifier order,
so the representation remains data-driven for longer 5/6/7-fragment paths.

This slice applies only when several fixed fragments belong to the same
polymer chain. It does **not** redefine an interface seed as sequence-adjacent
fragments and is not the core solution for the original Interface-Seed cage
topology. In that topology, fixed edges are `A_i <-> B_i` while the generated
unit is `B_(i-1) -- A_i` (or the equivalent reverse indexing).

The compiler fails closed for two cases that cannot be represented by one
linear protein chain: two outgoing/incoming links at one terminus and a closed
peptide cycle. Independent open paths and declared chain breaks retain their
existing multi-ASU-chain behavior. Adapter metadata now records the ordered
link IDs, source IDs and selector path while preserving singular legacy keys
for one-link designs.

New regression coverage checks path ordering, single materialization of an
internal seed, arbitrary path length, branch rejection, cycle rejection and a
single-file three-fragment end-to-end C3 compilation. Local `compileall`,
`py_compile` and `git diff --check` pass. The workstation Python environment
lacks `pydantic`, so the feature remains unpromoted until the complete LRZ
`rc-foundry` unit suite passes. The following step is explicit ownership of
input chains that already correspond to different group actions; without that
mapping, a complete pre-expanded multi-chain cage can still be over-expanded.

## 2026-08-07 original Interface-Seed topology re-audit

The original Ho-Yeung implementation confirms the interleaved two-graph
semantics directly. Its input PDB contains at most chains A and B, which form
one asymmetric non-covalent interface pair. Cyclic expansion renames each
copy as `(A,B)`, `(C,D)`, `(E,F)`, and so on. The contig template is
`Y...generated...X`; after comparing the two neighbouring directions, the
runtime substitutes X/Y with halves from adjacent interface copies. For C3,
one direction yields units equivalent to `F--A`, `B--C`, and `D--E`, rather
than the non-covalent pairs `A<->B`, `C<->D`, and `E<->F`.

Mosaic's existing IR separation between `InterfaceEdgeInstance` and
`ScaffoldLinkInstance` is therefore conceptually correct. The missing product
feature is a first-class topology pass that validates the alternating
interface/scaffold cycle, assigns every physical fragment to exactly one
interface pair and one polymer unit, and then maps equivalent units to the
declared symmetry action without duplication.

## 2026-08-07 packing-guidance v4

The designed-interface controller is being upgraded from contact formation
to a broader differentiable backbone-packing objective without adding a new
sampler path. In addition to nearest-pair attraction, balanced per-side
coverage, contiguous patches, clash repulsion and optional centroid distance,
v4 adds four terms:

- an orientation term penalizes selected contact patches whose local C-alpha
  tangents point end-on through the opposing patch;
- a contact-shape term penalizes strongly non-uniform nearest-contact depths,
  discouraging one protruding point from standing in for a broad interface;
- a backbone term protects adjacent generated C-alpha spacing from local
  token-by-token collapse during guidance;
- a log-mean-exp source-interface term puts more gradient pressure on the
  worst currently unsatisfied declared interface while preserving equal
  weighting across symmetry multiplicities.

All four values and their per-edge evidence are emitted in sampler diagnostics
schema v4 and required by the independent runtime audit. They are deliberately
described as backbone-level packing proxies. They do not calculate SASA,
atomic shape complementarity, hydrophobic burial or sequence designability.
Those require all-atom/sequence-aware evaluation later in the pipeline.

## 2026-08-07 unified graph-interface sampler guidance

The public authoring surface is now explicitly divided into two retained
levels without dividing the backend. `UserDesignSpec.user_mode` reports
`simple` for contig-style `generation + constraints`, and `expert` when the
user declares `components / ports / interfaces / connections`. The CLI plan
shows the selected level. Both lower to the same Assembly IR and timestep
runtime. Simple terminal designs now choose their closest non-identity group
neighbour from geometry instead of taking the first registry element. A motif
on a symmetry stabilizer receives a deterministic, clash-avoiding generic
orbit initialization; an already usable input pose is preserved.

The public interface intent is now inferred for the two ordinary contig
topologies rather than delegated to user tuning. Terminal generation around a
fixed central motif lowers to an internal output-stage symmetry-neighbour
interface objective. Between-generation linking supplied fixed fragments does
not invent a second interface objective. A bare `mode: contact` is valid in
the advanced graph API; contact count is no longer mandatory. Compiler output
adds an `auto` interface-quality contract, and the sampler derives bounded
coverage and contiguous-patch targets from the number of generated residues
on both sides. The final interface audit independently derives and checks the
same residue-scale targets. Explicit ports, neighbour relations and numerical
thresholds remain advanced cage/replay controls, not requirements for simple
motif scaffolding.

T job `5735772` completed the 50-step public multi-face graph run on V100 in
23:02. It passed the assembly-interface, exact constraint-orbit, RFD3 input and
scaffold audits. All 2316 fixed heavy atoms across 12 tetrahedral copies were
retained with 0.000054 A joint RMSD; the output had 12 continuous chains, zero
CA clashes and exact T symmetry. This is strong evidence for the supplied
interface mode: it preserves a declared `preserve_input` relation. It does not
show that an arbitrary motif can induce a newly packed generated interface.

The second behavior is now represented without a parallel compiler or sampler.
Public `preserve_input` relations lower as input-stage hard constraints;
public `contact` relations lower as output-stage design targets. Required
output-stage graph edges automatically enable a shared RFD3 sampler field. The
compiler-expanded concrete neighbour pairs are converted to generated-chain
CA masks, all edge energies are evaluated jointly, updates are bounded and
token-rigid, fixed atoms never receive the update, and the existing unified
symmetry/fixed-orbit projector runs again after every guidance step. This is
intentionally different from a global compactness force or Ho-Yeung-style
radius drag: only graph-declared neighbour edges act, and unrelated copies are
not pulled toward the center.

The final assembly-interface audit now distinguishes the two semantics. Input
relations are evaluated on declared fixed port atoms. Output contact relations
are evaluated on generated heavy atoms of the resolved output chains after
excluding all fixed residues from `diffused_index_map`. Missing generated
contacts therefore fail closed even when motif restoration and symmetry are
perfect. `preserve_input` also defaults to at least one sub-4.5 A heavy-atom
contact; setting the minimum to zero is an explicit geometry-only opt-out.

This implementation is locally syntax-clean but the workstation Python lacks
the Foundry dependencies (`pydantic` and `torch`). It therefore requires the
complete LRZ unit suite and a V100/P100 contact-design GPU canary before the
runtime field can advance beyond experimental status. Remaining scientific
work includes stronger orientation-aware packing terms, sequence-aware
side-chain evaluation and downstream design/refolding validation.

## 2026-08-07 graph-aware inverse-search slice

The first executable inverse-assembly search layer is now implemented above
the common public compiler. `rfd3-mosaic search` accepts a normal public
components/ports/interfaces/connections design, enumerates canonical
nonidentity group transforms for selected interface edges and optionally
samples the already-declared component pose ranges with deterministic seeds.
Every candidate is lowered into the same AssemblySpecification and evaluated
by the production standalone compiler. Ranking is feasibility-first: required
interface and linker failures, objective failures and hard clashes precede
interface-contact, linker-span and clearance diagnostics. Combinatorial growth
is bounded before compilation.

The search writes all compiler artifacts, one machine-readable
`graph_search.json`, and ordinary resolved public YAML files for the top
candidates. Those YAMLs replay the concrete transform assignments and pose
seeds through the existing validate/submit/RFD3 path; there is no separate
cage sampler. Selection now fails closed: only statically accepted candidates
are serialized, reloaded through the public loader and strictly recompiled;
the replayed initialized CIF must hash identically to the ranked assembly.
Replay failures remain diagnostic search records and are never exposed as
submittable selected designs. The same search can now compare an explicit set
of candidate Cn, Dn, T, O or I symmetries; selected YAMLs freeze the winning
group as well as neighbour relations and pose seeds. This is genuine discrete
architecture comparison using complete-assembly feasibility, but it does not
claim that one local relation uniquely identifies a global group. Automatic
symmetry-family proposal, component topology, stabilizer/coset assignments and
generated connectivity remain unresolved.
The generic graph-search compiler/replay layer is CPU validated. Individual
architecture families retain their own evidence level: a finite-group
candidate is not promoted to GPU/scientific readiness until a frozen YAML for
that family passes native adapter replay and its representative output audits.

## 2026-08-07 independent component initialization

- While the corrected joint-orbit T GPU smoke is running, the public static
  pose API is being generalized from one implicit motif pose to independently
  named component poses.  The backwards-compatible singular
  `sampling.initial_pose` remains valid for a one-component design.  New
  `sampling.initial_poses` is keyed by the user's explicit `coupling_group`;
  every component carries its own radius, axial offset, radial direction,
  orientation and seed.  Lowering rejects unknown groups and ambiguous use of
  both spellings.  Per-group seeds are preserved in Assembly IR and sampled
  with independent RNGs, making realized poses invariant to initialization
  declaration order.  The first intended gate is
  `lrz_public_t_two_orbit_initialized_short_v100_smoke.yaml`, which restores
  two genuinely independent T seed orbits instead of merging them into the
  temporary joint-seed workaround.  This slice is locally syntax/YAML clean
  but requires the LRZ unit suite and strict geometry preflight before GPU
  submission.
- The first strict preflight of that independent two-orbit T configuration
  correctly rejected the proposed 70/80 A, 20/30 degree placement: 72 atom
  pairs were below 2 A with a 0.273 A minimum.  This is a static candidate
  geometry failure, not a compiler or T-runtime failure.  The smoke candidate
  now places the two master centers about 47.0 A apart (70 A at 20 degrees and
  100 A at 45 degrees), still inside the reach of its 30-residue generated
  segment.  Standalone clash errors now report the worst concrete motion-group
  copy pairs rather than only the global count, so future pose correction does
  not require an ad-hoc diagnostic script.  The revised pose must pass LRZ
  preflight before it is submitted.

## 2026-08-06 science-core convergence: native Dn runtime

- Public-design validation now includes a strict, topology-neutral static
  geometry preflight.  `validate` lowers the public declaration, expands the
  complete symmetry assembly in a temporary directory and applies the same
  standalone clash/interface gates used by execution.  `render`, `run` and
  `submit` call the identical preflight before creating a persistent request
  directory, so callers cannot bypass it and consume a GPU slot with invalid
  source geometry.  Temporary compiler artifacts are discarded; successful
  validation reports only assembly atom, residue and chain counts.  Regression
  coverage includes both a separated C3 expansion and a deliberately
  coincident expansion that must fail closed.
- T job `5734023` is the motivating negative control, not a runtime failure:
  its unchanged C3-embedded Prism coordinates produced 2676 inter-group atom
  pairs below 2 A (minimum 0.461 A) during standalone expansion, before RFD3
  inference.  Job `5734024` uses the same invalid geometry and should be
  cancelled rather than allowed to repeat the result.  The corrected first T
  experiments use one jointly coupled rigid seed plus a sampled 80 A initial
  pose; validate them with the new preflight before submission.

- Static native D3 has crossed the GPU-canary boundary.  The 50-step V100
  run `5733912` (`public-d3-two-orbit-v100-s915`) completed on `dgx-002` in
  **00:16:25** with scheduler exit `0:0`.  RFD3 input prevalidation, the
  two-orbit exact-constraint audit and the scaffold validity audit all passed,
  and the run emitted a complete structure.  This promotes `dn_static` from
  `cpu_validated` to `gpu_canary`; it does not yet promote Dn to `stable`.
- Dynamic D3 job `5733940` is a successful runtime-kernel result but a failed
  end-to-end candidate.  Both independently mobile orbits executed seven
  atomic joint updates over all six declared D3 actions, translated by
  0.065817 A and 0.123855 A, rotated by 0.478063 and 0.256409 degrees,
  preserved all 1158 fixed heavy atoms at at most 0.000014 A per-copy internal
  RMSD, and passed the mobility, constraint-orbit, continuity, compactness and
  exact-symmetry gates.  The final scaffold nevertheless contained one
  symmetry-equivalent generated--generated CA clash orbit: residue 33 in
  chain pairs A--D, B--F and C--E was separated by about 1.816 A.  Output
  residue 33 lies in the generated 10--94 interval rather than either fixed
  1--9 or 95--106 interval.  The strict scaffold gate therefore correctly
  rejected this sample.  Do not attribute this event to loss of motif
  rigidity or incomplete D3 actions, and do not relax the 3 A clash cutoff;
  replay a different diffusion seed before deciding whether additional
  assembly-context scaffold guidance is required.
- That controlled replay has now passed.  V100 job `5733972`
  (`public-d3-two-orbit-mobility-v100-s917`) completed all required gates with
  zero CA clashes and zero chain breaks.  Both independently mobile components
  executed seven atomic joint scaffold updates over all six D3 actions.  They
  translated by 0.139269 A and 0.081815 A and rotated by 0.404452 and 0.473126
  degrees.  The runtime matched all 1158/1158 fixed heavy atoms, with maximum
  per-copy internal RMSD 0.0000138 A.  Final symmetry coordinate RMSD was
  0.0000351 A and maximum symmetry error was 0.0000535 A.  This establishes a
  separate `dn_dynamic_multi_orbit` GPU-canary capability; it does not yet
  establish broad Dn order/topology generalization or scientific pose quality.
- The first polyhedral-cage foundation is now implemented locally without
  changing the GPU sampler.  `SymmetryTransformSetSpec` and the common
  registry support the 12 proper tetrahedral, 24 proper octahedral and 60
  proper icosahedral rotations with stable identity-first transform IDs,
  arbitrary declared center and a deterministic oriented group frame.  The
  icosahedral action is constructed from the 60 directed edges of a canonical
  icosahedron rather than a hand-maintained matrix table.  CPU regressions
  require unique identity, proper orthogonal matrices, center preservation,
  pairwise group closure and complete instance expansion for T/O/I.  Public
  T/O/I designs now lower into the common AssemblySpecification; nonzero
  cyclic `orbit_offset` is rejected because a polyhedral group has no
  canonical ring ordering.  Native RFD3 adapter/runtime execution remains
  intentionally fail-closed until this registry/compiler slice passes the LRZ
  suite and declared-frame transport is implemented end to end.
- The LRZ `rc-foundry` suite subsequently passed **468/468 tests** in 18.827
  seconds.  This promotes `polyhedral_groups` from `schema_only` to
  `cpu_validated` for the registry/compiler boundary only.  Native RFD3
  declared-frame input construction, GPU denoising and cage-scale memory
  behavior remain separate maturity gates.
- The next local slice begins the native RFD3 declared-frame transport needed
  for those separate gates.  The RFD3 frame resolver now consumes a
  compiler-declared T/O/I registry before invoking the legacy Cn/Dn frame
  generator, validates the complete 12/24/60 transform count, and preserves
  the declared transform order.  Two-dimensional entity reannotation uses
  symmetry multiplicity without trying to synthesize legacy frames.  The
  Mosaic adapter emits native symmetry IDs `T`, `O` and `I`, always includes
  declared matrices for polyhedral inputs, and prevalidation recognizes their
  finite multiplicities.  Focused regressions cover the multiplicity contract,
  legacy-generator bypass, incomplete-registry rejection, generic RFD3 virtual
  frame round trips for all 12/24/60 actions and a tetrahedral terminal-design
  adapter build.  The complete LRZ `rc-foundry` suite passed **477/477 tests**
  in 16.030 seconds, accepting this declared-frame transport at the CPU
  boundary.  Partial diffusion remains
  outside this slice because its legacy symmetry verifier still constructs
  Cn/Dn frames directly; declared-frame T/O/I partial inputs now fail closed
  with an explicit unsupported-mode error instead of reaching that verifier.
- The first post-CPU execution gate is declared in
  `experiments/lrz_public_t_two_orbit_a100_canary.yaml`.  It reuses the public
  design language and common Assembly IR, expands two static motif orbits over
  all 12 proper tetrahedral actions and runs 50 denoising steps through the
  explicit-all-copy reference backend.  The 80 GB A100 profile is intentional:
  a 12-copy reference trajectory is roughly four times the pair-state scale of
  the six-copy D3 canary.  Do not use a 10-step output as a scaffold-quality
  gate and do not claim local-neighbourhood T support from this experiment.
- A separate memory-oriented race configuration is
  `experiments/lrz_public_t_two_orbit_short_v100_smoke.yaml`.  It retains the
  same two static exact orbits and all 12 T actions but shortens the generated
  segment from 85 to 30 residues, bringing the token count near the already
  demonstrated six-copy D3 scale.  It may be submitted independently with
  `--profile v100` and `--profile p100`; passing it proves small-GPU runtime
  transport, not full-length tetrahedral scaffold quality.  Reducing timestep
  count alone is not a memory remedy because it does not reduce peak pair-state
  size.
- The first two-orbit T submissions were rejected correctly before GPU
  inference: job `5734023` reported 2676 inter-group atom pairs below 2 A,
  with a 0.461 A minimum.  The Prism coordinates encode a C3 placement and
  cannot be reinterpreted directly as a generic tetrahedral master pose.
  Because one public `initial_pose` deliberately cannot ambiguously reposition
  two independent coupling groups, the corrected first T gate couples both
  fixed fragments into one complete rigid seed and samples one generic pose at
  80 A radius before applying all 12 actions.  The replacement configurations
  are `lrz_public_t_joint_orbit_short_v100_smoke.yaml` and
  `lrz_public_t_joint_orbit_a100_canary.yaml`.  They prove one exact joint T
  orbit; independent multi-orbit T initialization remains a subsequent API and
  runtime milestone.
- The multi-orbit scaffold controller now treats proposals as one atomic
  Jacobi update: every orbit reads the same immutable pre-update state and the
  combined candidate is committed only when the joint assembly objective
  improves. The joint objective now includes an explicit inter-orbit CA clash
  term, closing the case where two individually acceptable movable motifs
  collide only after both proposals are materialized.
- Runtime mobility diagnostics now record the exact group-action count and
  ordered transform IDs for every orbit. The mobility audit compares these
  against the compiled orbit declarations and fails closed if, for example, a
  declared six-action D3 orbit is accidentally executed as only its three-copy
  C3 subgroup.
- Added CPU regressions for two independently moving D3 orbits: both proposals
  are atomic and declaration-order independent, and each accepted master pose
  reconstructs all six D3 actions exactly. The V100 dynamic canary is
  `experiments/lrz_public_d3_two_orbit_mobility_v100_canary.yaml`; it must be
  submitted only after the static V100 D3 canary passes.

- Product operations (`run`, `runs`, `status`, `report`) are sufficiently
  complete for current development.  The active priority has returned to the
  scientific runtime: exact non-cyclic group actions and multi-orbit control.
- Exact orbit expansion, joint projection and atomic multi-orbit updates were
  already group-action based.  The remaining hard Cn assumption was in
  scaffold-derived pose guidance, which required every transform to share one
  cyclic axis.  That is false for Dn because its secondary two-fold coset has
  different axes.
- Axis-dependent guidance now resolves the primary Cn subgroup from a declared
  Cn or Dn runtime registry.  Dn keeps all `2n` transforms for materialization
  and exact projection; only its first `n` rotation-subgroup transforms define
  the principal axis used by radial/axial/tilt objectives.
- Added CPU regressions proving that a D3 registry yields the correct primary
  axis while orbit expansion still produces all six group copies, plus public
  lowering for two independently controlled D3 components.
- Added `experiments/lrz_public_d3_two_orbit_h100_canary.yaml`.  Its first GPU
  objective is deliberately static: prove six-copy D3 denoising, two exact
  constraint orbits, continuity, clash freedom and final group closure before
  enabling D3 dynamic mobility.  P100 is not the primary target because the
  earlier full D3 attempt exceeded its practical memory envelope.

The complete LRZ `rc-foundry` unit suite passed **420/420 tests** on
2026-08-06 after adding public scaffold-proposal selection, axis-aware
radial/radial-axial mobility, joint whole-interface-seed lowering and explicit
rigid-component/symmetry-copy reporting in `rfd3-mosaic plan`.

## 2026-08-05 public sampling and capability ledger

- Upgraded public `fixed_xyz` from an ambiguous coordinate label to explicit
  rigid-geometry component semantics. A comma-separated declaration is one
  joint component; separate declarations are independent unless they share a
  `coupling_group`. Compiler motion groups, runtime constraint orbits and
  result audits now carry the same component identity.
- Fixed components are gauge-invariant: every component is accepted only when
  all of its atoms across the complete symmetry orbit superpose under one
  common rigid transform. Absolute laboratory-frame displacement is neither a
  constraint nor an acceptance metric. This matches the intended use: all
  jointly fixed relative positions must remain unchanged, while an irrelevant
  common translation/rotation is allowed.
- V100 canary `5732041` already provided positive geometry evidence before
  this semantic cleanup: all 579 selected atoms jointly aligned at
  **0.000016 A RMSD** (orbit distance-matrix RMSD **0.000015 A**). Its Slurm
  failure was caused by the deliberately short 10-step scaffold audit (195
  chain breaks and 12 CA clashes), not by loss of the fixed component. The
  former 7.847 A raw offset was only a common coordinate-frame translation.
- Added the follow-up 50-step V100 canary
  `experiments/lrz_public_fixed_components_v100_canary.yaml`. It declares the
  two Prism fixed selections separately with the shared coupling group
  `prism_site`, so one GPU result exercises the new public component contract
  and must jointly superpose both selections across all C3 copies.
- The component contract now includes a user-selected pose policy. A fixed
  component defaults to `pose.mode: fixed`; `pose.mode: bounded_mobile`
  preserves the component's complete atomic geometry while allowing that
  component to translate and rotate independently inside explicit cumulative
  and per-step bounds. The compiler emits per-component orbit mobility and
  the experiment worker enables the existing multi-orbit denoiser-fit
  controller automatically. This removes the previous mismatch in which the
  compiler/audit recognized independent components but the sampler still
  restored every component to its original pose.
- Bounded component designs now carry complementary geometry, symmetry and
  motion evidence. Within each symmetry copy, the constraint-orbit audit
  jointly superposes every fixed fragment in one `coupling_group`; the normal
  scaffold audit verifies the relationship among symmetry copies; and the
  component-mobility audit proves that the GPU controller executed, refreshed
  conditioning, and stayed inside every user-declared cumulative translation
  and rotation bound. Fixed-pose components retain the stronger initial
  complete-orbit joint-fit contract.

- Added a machine-readable capability registry and
  `rfd3-mosaic capabilities [--format json]`. It separates schema presence,
  CPU validation, GPU canaries, engineering support, stable behavior and
  scientific validation instead of treating all code paths as equivalent.
- Added a topology-neutral `SamplingPlan`. Public `sampling.initial_pose`
  now describes one pre-diffusion rigid pose independently of the diffusion
  seed and timestep configuration.
- Radius, axial offset and fixed or Haar-uniform SO(3) orientation lower into
  `AssemblySpecification.initialization` for the complete motif motion group.
  No initial-pose declaration means no coordinate repositioning.
- This slice implements bounded timestep mobility for compiler-declared fixed
  components. It does not yet expose cylindrical timestep projection or a
  general ensemble/QD dispatcher; those remain separate capabilities.

## 2026-08-05 current phase: platform-first productization

- The current development objective is a general, professional and extensible
  software platform. It is not yet the selection of one metal site, one
  protein target or one flagship scientific benchmark.
- The supported spine is `UserDesignSpec -> AssemblySpecification +
  ConstraintPlan + SamplingPlan -> BackendPlan -> Mosaic-RFD3 -> execution,
  provenance and audits`.
- A capability is not complete merely because its schema exists. Completion
  requires user configuration, selector binding, compiler lowering, sampler
  runtime, result audit, tests and user documentation.
- Functional-site and inverse-architecture work remains an important future
  scientific layer, but it is frozen while the public execution spine is
  completed. The internal schemas and relation-compatibility kernel are kept
  as extension points and are not public product claims.
- After the first platform slices, the complete LRZ unit suite passed
  **394/394 tests** on 2026-08-05. The CLI error printed by the
  `test_submit_cli_rejects_unlowered_public_design` test is intentional
  fail-closed behavior.
- The next slice replaced topology-labelled audit selection with
  explicit `AuditRequirement` declarations. Exact fixed constraints use the
  topology-neutral `constraint_orbit_audit.json`; the historical central
  audit remains a compatibility engine while its implementation is gradually
  generalized. This slice is now LRZ CPU-validated by the 394-test suite; a
  central golden GPU canary and a public `fixed_xyz` GPU canary remain.
- While the P100 canary is queued, the numeric constraint-orbit audit engine
  has been moved into the topology-neutral module. The historical
  `rfd3_central_motif_audit` now only translates legacy argument/report names;
  a regression assertion requires both APIs to return identical thresholds
  and numeric summaries. This compatibility refactor passed the complete
  **394/394** LRZ unit suite and does not alter already submitted job
  snapshots.
- Added a declarative public `fixed_xyz` P100 canary for the Prism C3 motif.
  It uses only the public `UserDesignSpec`, fixes A12-20 and A26-37 as one
  complete C3 constraint orbit, generates an 85-residue between segment and
  deliberately omits `initial_pose` so the supplied geometry is not moved.
  It must be validated and planned now, but submitted only after the preceding
  central P100 audit canary succeeds.
- Added a V100 execution profile with an explicit `gpu:1` GRES request. The
  10-step public fixed-XYZ canary may now be rendered from the same immutable
  design for P100, A100, H100 and V100. These runs test backend portability
  and the complete software/audit path; they are not independent scientific
  designs and must retain per-run execution provenance.
- The first V100 public fixed-XYZ canary (`5731968`) correctly failed closed
  during native prevalidation before inference. The public between-segment IR
  contained a fixed motion group and a constraint orbit, but the adapter
  incorrectly routed every between-segment design through interface-edge
  group lowering. Because a topology-neutral public design has no synthetic
  `InterfaceEdge`, it emitted empty `motif_constraint_groups` and
  `motif_constraint_orbits`.
- The adapter now lowers fixed groups directly from
  `ConstraintOrbitInstance`, supporting several fixed fragments in one joint
  group per symmetry copy. Interface-edge designs retain their existing
  cross-protomer group semantics. A regression fixture covers a two-fragment
  C3 public between path, three complete runtime groups, exact transform IDs
  and declared-frame execution. This fix requires LRZ unit validation before
  the failed canary is resubmitted.

## Future scientific roadmap: functional-site-first assembly design

- A future scientific lead is cooperative functional-site design: compile
  multi-subunit functional geometry into symmetry-coupled constraint orbits
  and preserve or refine that site throughout all-atom diffusion while the
  surrounding assembly is generated.
- Constraint-orbit diffusion is the method core. Symmetry/topology discovery
  is a later search layer that proposes candidate interpretations; it is not
  the sole scientific claim or a replacement sampler.
- Added the first internal `FunctionalGeometrySpec`: rigid/soft-rigid
  functional fragments, stable symbolic atoms, distances, angles, periodic
  dihedrals, chirality, relative-pose bounds and true multi-fragment
  coordination hyperedges with optional ownership declarations.
- Added a backend-independent residual evaluator. It emits observed values,
  raw errors, pass/fail state and non-negative normalized violation per
  relation. Structure selector binding and GPU constraint-orbit lowering are
  deliberately still absent, so capability maturity remains `schema_only`.
- A possible first flagship target is one cooperative site spanning three
  symmetric subunits, exact every-timestep site-orbit restoration,
  complete-assembly generation, clash-free output and downstream
  sequence/refolding evidence.
- The preliminary Cn code is now an internal relation-compatibility kernel,
  not a public CLI and not an architecture solver. It distinguishes the
  observed cyclic subgroup from a proposed full group and reports unobserved
  cosets and unresolved topology/chemistry evidence.
- Discrete group/topology changes remain forbidden inside one diffusion
  trajectory. Later inverse search creates separately identified candidate
  trajectories; bounded mobility refines only continuous variables.

## Project identity

- Repository: `zhaohaixi2005-a11y/rfd3-mosaic`
- Upstream: `RosettaCommons/foundry`
- Active branch: `refactor/product-core-v1`
- Server working tree: `/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic`
- Local mirror: `/home/haixi/Documents/mosaic`
- Goal: build a generator-independent `AssemblySpecification` compiler and a
  constraint-aware RFD3 symmetry sampler in which fixed or bounded-mobile
  motif orbits, generated segments and exact group actions are updated through
  one runtime contract. LHD101 and individual Cn/Dn runs are regression
  fixtures, not product-specific code paths.

## 2026-08-04 architecture migration

The architecture has been reset around three planes:

```text
compile:   user intent -> AssemblySpecification -> RFD3 runtime features
infer:     (assembly state X_t, orbit poses G_t) -> denoise/control/project
evaluate:  motif/symmetry/scaffold/assembly audits -> acceptance gate
```

The backwards-compatible migration now has three completed local slices:

- `AssemblySpecification` is now the topology-neutral public schema name;
- `InterfaceSeedSpec` remains an alias, so existing configs and imports keep
  working;
- `load_assembly_config` accepts new `assembly:`, legacy `interface_seed:`, or
  an unwrapped payload and rejects ambiguous dual wrappers;
- ports and interface edges may be empty for non-interface motifs;
- `generated_segments` can represent N/C terminal extensions using the same
  assembly schema used by between-motif scaffold links;
- central and interface compatibility frontends now return one immutable
  `CompiledAssembly` artifact, so prevalidation and inference no longer branch
  on how the input was compiled;
- each `CompiledAssembly` now carries immutable, topology-neutral
  `CompiledAudit` command descriptions; the experiment worker no longer
  branches on central motif versus interface seed during evaluation;
- the runtime public API is now `ConstraintGroup`, `ConstraintOrbit` and
  `ConstraintOrbitLayout`; the original `InterfaceConstraint*` names remain
  compatibility aliases, and the motif-mobility controller consumes the
  neutral API;
- `UnifiedJointProjector` now owns the ordered runtime contract
  `symmetry projection -> complete constraint restore -> closure validation`;
  exact Euler updates, fixed-motif projection and final exact projection are
  wired through it while the previous sampler methods remain compatibility
  wrappers;
- mobility is now a property of the topology-neutral symmetry orbit contract,
  not intrinsically of an interface edge. `OrbitMobilitySpec` declares the
  allowed subspace (`radial`, `radial_axial`, `tilt_only`, `bounded_se3`),
  cumulative bounds, timestep window, per-step translation/rotation trust
  region, proposal source and objective references;
- `ConstraintOrbitInstance` is now part of the compiled assembly IR. Legacy
  non-fixed interface mobility is migrated into that IR with conflict checks;
  the adapter reads mobility from the compiled orbit rather than directly
  from an interface edge;
- `hoyeung_drag_compat` is represented only as one optional proposal source.
  It does not define the core state model; the formal state remains one
  bounded rigid `SE(3)` pose per master constraint orbit;
- the experiment compiler no longer calls separate central-probe and
  interface-seed builders. Legacy/simple frontends first lower to one
  `AssemblySpecification`; one adapter then expands `CompiledInstanceSet`,
  consumes either `ScaffoldLinkInstance` or `TerminalExtensionInstance`, and
  emits the same native RFD3 feature contract;
- `rfd3_central_motif_probe` remains only as a diagnostic compatibility tool
  and regression fixture. It is no longer the central-motif backend used by
  `compile_experiment_assembly`;
- this third slice has passed local syntax and diff checks. After the latest
  synchronization, the complete LRZ unit suite passed **331/331 tests**.
  Native prevalidation plus one central and one interface
  GPU regression through this new compiler path are still required before
  calling the compiler migration end-to-end validated.

The fail-closed identity contract for the Git source state, every runtime
input and the RFD3 checkpoint has passed the complete LRZ unit suite. The next
local productization slice adds a compact immutable source archive to every
rendered submission. The worker will import Mosaic, RFD3 and Foundry from the
verified per-run snapshot instead of the shared mutable checkout. This source
snapshot slice requires LRZ unit validation before it is considered complete.

The earlier compiler migration milestone was:

```text
validate native AssemblySpecification lowering on LRZ
-> move all remaining initialization/finalization projection sites through
   UnifiedJointProjector
```

The current product milestone is narrower and precedes multi-orbit research:

```text
topology-neutral audit requirements and exact constraint-orbit audit
-> LRZ full unit validation
-> one public fixed_xyz render/submit/GPU/audit canary
-> first cylindrical operator end-to-end slice
```

## 2026-08-05 public design and constraint-plan slice

The product-interface refactor has started without replacing the validated
assembly compiler or exact sampler:

- `schema/design.py` defines a small immutable `UserDesignSpec` with one input
  structure, declared symmetry, topology-neutral generated regions and zero
  or more optional constraint clauses;
- the public constraints are canonical `fixed_xyz`, `cylindrical` and
  `bounded_mobile` operators rather than separate protocols or Slurm scripts;
- omission of constraints represents normal diffusion with no additional
  Mosaic motif fixing;
- `constraint_plan.py` compiles declarations into deterministic hard or
  bounded projector intent and rejects visible per-DOF conflicts;
- a backend capability gate rejects unsupported operator kinds instead of
  silently dropping them;
- `rfd3-mosaic validate/plan` recognizes the new public schema, while
  `fixed_xyz(atoms=all)` terminal/between designs can now be materialized into
  the existing experiment envelope for render/submit;
- selectors are resolved against real PDB/mmCIF atom identities before
  lowering, so differently written but overlapping ranges cannot claim the
  same constrained degree of freedom;
- the first lowerer deliberately rejects implicit endpoint fixing,
  constraints on unattached fixed regions, partial-atom fixed XYZ, and the
  not-yet-bound cylindrical/mobile operators.

This slice is locally syntax-checked. The complete LRZ unit suite and the
source-snapshot slice still require a single synchronized validation run.
The validated legacy runtime behavior is unchanged. Public fixed-XYZ execution
uses that same assembly adapter, exact projector, source snapshot and audit
worker; its LRZ unit and GPU canary gates are still pending.

The existing `OrbitRigidMotifController` already carries bounded translation
and SO(3) rotation and refreshes conditioning inside the RFD3 timestep loop.
It remains experimental because scaffold-derived guidance currently assumes
one cyclic orbit and uses a ring-specific axis/tilt objective. Generalization
requires multi-orbit simultaneous residual aggregation, optional DOF
subspaces, objective composition and Cn/Dn/T/O/I transform-registry support.

## Current project state

| Capability | Current evidence | Status |
|---|---|---|
| Interface-seed compiler | Generic fragment, topology, Cn/Dn registry, pose provenance, adapter and prevalidation paths are implemented and unit-tested | Implemented |
| Static exact-C3 RFD3 | 50/100/200-step inference preserves the complete cross-chain seed, continuous copies and declared C3 symmetry | End-to-end demonstrated |
| Central-motif exact C3 | Registry-v4 runs 5729451--5729453 passed the central-orbit and scaffold gates with no breaks or CA clashes and sub-0.00005 A symmetry maximum error | End-to-end demonstrated on the compatibility frontend; unified-compiler GPU regression pending |
| C5/C6/C7 static RFD3 | A 48-structure extracted set produced 37 strict seed+scaffold passes across all three orders | Cross-order engineering generalization demonstrated on the screened set |
| Pose exploration | Haar SO(3), joint LHS, topology-aware endpoint descriptors and quality-diversity morphology coverage are implemented | Implemented; still a GPU-budget selector, not a biological score |
| Scaffold-aware motif mobility | V100 job 5733341 produced a non-zero bounded SE(3) update and passed component, symmetry, continuity and clash audits; whole-interface-seed and axis-aware canaries remain pending | Full-SE(3) GPU canary passed; broader engineering/scientific behavior not yet validated |
| Multiple interface orbits | C3 V100 job 5733788 and D3 V100 job 5733972 jointly moved two independent orbits with differential bounded motion and passed every required audit | Dynamic C3 and six-action D3 GPU canaries passed; broader topology/order coverage pending |
| Dn inference | Static D3 job 5733912 and dynamic two-orbit D3 job 5733972 passed input, exact-orbit, mobility and scaffold-validity gates | Static and dynamic D3 GPU canaries passed; broader Dn remains below stable |
| C12/C20 | Artificial order-10 guards are removed and the local-neighbourhood kernel is unit-tested, but explicit preprocessing/output and quadratic pair-state costs remain; no explicit/local GPU equivalence run has passed | Outside the validated production architecture |
| Sequence/design validation | ProteinMPNN, multimer prediction, interface-energy ranking and experimental validation are not part of the completed evidence | Not started |

The defensible current claim is:

> RFD3 Mosaic can compile a cross-protomer interface seed and preserve it
> while generating a continuous, exact-symmetry cyclic scaffold. C3 has been
> demonstrated end to end. In the first audited C5/C6/C7 set, 37/48
> structures passed the same strict seed and scaffold gates. This supports
> engineering generalization across C5--C7, but does not yet establish
> sequence-level designability or experimental assembly.

The extracted-structure screen is:

```text
scripts/rfd3_mosaic/screen_extracted_cn_structures.sh
    -> src/rfd3_mosaic/rfd3_batch_screen.py
    -> C5/C6/C7 JSON + CSV reports
```

Its acceptance order is deliberately strict:

```text
seed integrity
-> chain continuity and compactness
-> zero hard CA clashes
-> declared-transform Cn symmetry
-> only then compare ring shape and neighbouring-chain packing
```

Ring radius, axis clearance, shape aspect and contact counts are diagnostics
for diversity and ranking. They are not calibrated folding, assembly or
experimental-success scores.

### 2026-07-31 C5/C6/C7 extracted-structure screen

The first complete batch screen used:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c5_full/extracted_cif
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c6_full/extracted_cif
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c7_full/extracted_cif
```

Results:

| Order | Screened | Seed passed | Continuous | Zero CA clash | Declared symmetry | Strict passed |
|---|---:|---:|---:|---:|---:|---:|
| C5 | 15 | 15 | 15 | 12 | 15 | 12 (80.0%) |
| C6 | 18 | 18 | 17 | 13 | 18 | 13 (72.2%) |
| C7 | 15 | 15 | 14 | 12 | 15 | 12 (80.0%) |
| Total | 48 | 48 | 46 | 37 | 48 | 37 (77.1%) |

This establishes that seed preservation and declared cyclic symmetry are no
longer C3-only behavior in the screened runs. The dominant rejection is local
CA clash rather than loss of the interface seed. Two structures also fail
continuity.

Initial candidates for downstream inspection are:

| Order | Recommended job | Reason |
|---|---:|---|
| C5 | `5722385` | Strict pass, 127 minimum neighbouring contacts and 4.055 A minimum inter-chain CA distance |
| C6 | `5722398` or `5722341` | Safer 3.940/4.223 A minimum distance than the contact-ranked jobs `5722400` and `5722401`, which lie close to the 3 A hard-clash boundary |
| C7 | `5722413` or `5722344` | Strict pass with 71/66 minimum neighbouring contacts and 3.987/4.223 A minimum distance |

Contact-ranked C6 jobs `5722400` and `5722401` remain valid strict passes, but
their 3.006 and 3.229 A minimum inter-chain CA distances make them dense
packing controls rather than automatic first choices. The screened shape
aspect ratios are broadly ring-like and relatively flat; they do not by
themselves demonstrate cage-like three-dimensional closure.

The next scientific gate is not more CIF geometry ranking. It is to take a
small, shape-diverse strict-pass shortlist through sequence design, multimer
prediction and atom-level interface/clash assessment while retaining failed
and borderline candidates as negative controls.

### Core RFD3 method-development direction (prospective)

Benchmark expansion is deferred while the tool is still being built. The
primary development objective is no longer another task-specific sampler
patch. The current hypothesis is an RFD3-native interface-orbit sampler with
two coupled runtime states:

```text
X_t = complete all-atom scaffold/assembly diffusion state
g_t = one bounded, symmetry-reduced pose state per mobile interface orbit
```

The target per-step contract is:

```text
denoise the complete assembly
-> aggregate copy-wise scaffold signals in the master frame
-> infer one bounded, copy-equivariant update for g_t
-> preserve immutable cross-protomer interface cores
-> expand every master pose through the declared group actions
-> refresh all motif and constraint conditioning
-> project X_t back to exact group closure
-> continue denoising
```

This is currently best described as **symmetry-constrained alternating
inference**, not joint diffusion. No forward noise process, reverse transition
or learned score has yet been defined for `g_t`.

The first model-context audit is complete. The symmetry sampler passes the
full length-`L` noisy assembly directly to `diffusion_module`; symmetry is not
implemented by denoising an isolated ASU and copying it only after the network.
Both full and chunked pair representations are built over that assembly.
However, the `>3`-chain sparse-attention helper incorrectly iterated over
unique chain IDs as if they were query-atom indices, so its reserved
inter-chain keys covered only a few rows. The local fix now constructs reserved
inter-chain neighbours for every query atom and keeps small attention budgets
non-negative. The new targeted tests and the full LRZ unit suite pass. A paired
C5 GPU rerun is still required before claiming any improvement in generated
inter-subunit packing.

The implementation milestones are deliberately incremental:

1. Promote the existing constraint-orbit tensors into a validated, first-class
   RFD3 `InterfaceConstraintOrbit` runtime object. **Implemented and unit-test
   validated.**
2. Make the existing mobile-pose controller consume that object while
   preserving backwards compatibility with current emitted features.
   **Implemented and unit-test validated; static restoration remains on its
   legacy-compatible path until a separate change is justified.**
3. Audit the true RFD3 model context and identify gauge freedoms before
   choosing the pose parameterization. A raw six-degree-of-freedom `SE(3)`
   state must not be adopted if axial translation or rotation only changes the
   world frame or copy labels. **Full-assembly entry is confirmed; gauge
   analysis remains open.**
4. Treat interface pose as an explicit, bounded state with a copy-equivariant
   proposal: evaluate all equivalent copies, inverse-map their signals to the
   master frame, aggregate once, and regenerate copies only by group actions.
5. Add hierarchical roles only after the pose state is sound:
   immutable `interface_core`, orbit-following `rigid_support`, and a small
   time-scheduled `flexible_boundary` next to generated scaffold.
6. Generalize from one cyclic orbit to multiple non-equivalent Cn/Dn
   interface orbits with simultaneous, order-independent conflict handling.

Moving a complete interface orbit does **not** optimize the interface core's
internal packing: its two sides retain their validated relative geometry. It
can only improve junctions to generated scaffold, seed-external contacts,
clashes and global morphology. Weak or symmetry-incompatible seeds still need
to be rejected upstream.

This is a prospective method contribution, not a validated project claim.
More Cn samples, more penalties,
or direct reproduction of RFdiffusion1 motif dragging are useful validation or
engineering work but are not by themselves the core innovation. Large
benchmarks, sequence design and refolding remain required evidence later; they
are not the immediate construction priority.

### Target symmetry scope and scalable backends

The target is not limited to C3--C7. The intended finite-group scope is
`Cn`, `Dn`, `T`, `O`, and `I`; `H` is treated here as helical/screw symmetry
and requires a finite runtime neighbourhood rather than a finite group
multiplicity.

Current limitations are distinct and must not be treated as one unchecked
constant change:

- The local branch now removes both artificial 10-transform guards: the
  Mosaic adapter accepts C12/D6, and Foundry motif-frame recovery no longer
  rejects more than 10 transforms. The C12/D6 closure, adapter and frame
  regressions have passed on LRZ.
- Foundry's named-frame parser recognizes only `Cn`, `Dn`, and
  `input_defined`.
- Mosaic's typed symmetry schema currently enumerates only cyclic and
  dihedral transform sets.
- Explicit all-copy atom state grows linearly with copy count, while token
  pair state and several audits can grow quadratically; `O` (24 copies), `I`
  (60 copies), high-order `Cn/Dn`, and long helical windows therefore cannot
  be enabled responsibly by deleting the guards.

Development should separate two execution backends behind the same
interface-orbit contract:

1. `explicit_all_copy`: exact current representation for small finite groups;
   retain as the reference implementation and numerical oracle.
2. `local_symmetry_neighbourhood`: keep one independent master ASU/orbit,
   materialize only symmetry copies that can interact with it, denoise that
   local assembly context, inverse-map copy-wise updates to the master frame,
   aggregate once, and expand the complete requested assembly only for exact
   projection/output/audit.

The first backend-independent local-neighbourhood kernel is now implemented in
`rfd3/inference/symmetry/local_neighbourhood.py`:

- Cn selects `master, -1, +1, ...` with network copy count bounded by the
  configured neighbour radius rather than group order. C200 with radius one
  therefore exposes three copies to the future network view.
- Dn can select the same local cyclic neighbourhood in both dihedral cosets.
- explicit global-to-local atom maps reject incomplete transform coverage;
- local copy predictions are inverse-transformed, averaged in canonical orbit
  coordinates, and expanded to every global copy. Omitted global copies never
  dilute the denoiser update.

The same module now also constructs a fail-closed local feature view: it keeps
whole atomized tokens, reindexes `atom_to_token_map`, crops atom/token and pair
features together, and handles Mosaic motif-constraint atom axes explicitly.
The standalone C12/C20/C200/D100 neighbourhood, feature-crop, coordinate
expansion and sequence-logit expansion tests have passed on LRZ.

An experimental integration is now wired before `TokenInitializer` and into
`SampleDiffusionWithSymmetry`. It is selected only by
`symmetry_execution_backend=local_neighbourhood`, requires low-memory mode,
exact orbit-average state, coupled noise and fixed-motif preservation, and
currently rejects dynamic motif mobility. The default remains
`explicit_all_copy`. On 2026-07-31 the complete LRZ unit suite, including the
new sampler integration and fail-closed configuration tests, passed 273/273 in
9.189 seconds. A real small-order explicit/local A/B run remains pending, so no
production script enables the backend by default. In addition, preprocessing
still constructs the complete assembly before this crop and final output still
expands every copy; this integration bounds the initializer/denoiser network
view but is not yet an end-to-end C200 memory guarantee.

The staged order is:

1. remove task-script assumptions such as `C5|C6|C7` while retaining safe
   resource guards;
2. validate parameterized high-order `Cn` and `Dn` registries, closure,
   provenance, attention neighbourhoods, and audits on CPU;
3. add proper finite `T/O/I` transform registries and generic finite-group
   prevalidation;
4. validate the experimental local-neighbourhood sampler integration and
   demonstrate agreement with explicit all-copy results on small groups before
   using it for C12+, O or I;
5. add helical screw transforms, a declared repeat window and boundary-aware
   audits as a separate infinite-symmetry mode.

The former `>10` guards have been removed locally, but production high-order
submission remains blocked by staged CPU construction, bounded GPU memory and
scientific audit gates rather than by a hard-coded order constant.

## Maintained documentation

Only the following six documents are active and should receive future
updates:

1. `DEVELOPMENT_STATUS.md` — single source of truth for current evidence,
   limitations, cluster-operation boundary and resume point.
2. `docs/rfd3_mosaic/RFD3_MULTI_INTERFACE_SEED_FINAL_PLAN.md` — stable method
   architecture, data model, invariants and success criteria.
3. `docs/rfd3_mosaic/RFD3_MOSAIC_PRODUCTIZATION_PLAN.md` — public software
   boundary, Foundry fork policy, capability ladder and inverse-assembly
   solver architecture.
4. `docs/rfd3_mosaic/C5_C6_C7_200STEP_RUNBOOK.md` — executable pose,
   inference, audit and extracted-structure screening workflow.
5. `docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md` — the only
   active experimental-method note; it must stay separate from the validated
   static claim.
6. `docs/rfd3_mosaic/USER_CLI.md` — concise public experiment configuration,
   execution-profile and submission contract.

Two additional documents are retained but not actively maintained:

- `docs/rfd3_mosaic/INTERFACE_SEED_RFD1_UPGRADE_AUDIT.md` — read-only
  historical code audit used to compare Interface-Seed 1.0 with this project.
- `docs/rfd3_mosaic/RFD3_INDEXED_INTERFACE_SEED_SHIFT_ROOT_CAUSE.md` —
  detailed, evidence-backed record of the original fixed-index/symmetry
  projector drift mechanism and its corrected runtime semantics.

Do not create another progress handoff, evolution plan or duplicate runbook.
Put new project evidence in this file, stable architectural decisions in the
final plan, executable C5/C6/C7 commands in the runbook, and mobility-only
results in the pilot note.

The superseded `RFD3_MULTI_INTERFACE_SEED_EVOLUTION_PLAN.md` was removed on
2026-07-31 after its still-relevant decisions had been incorporated into the
final plan and this status file.

## Project reading order

To understand the project without replaying the full development history, read
these files in order:

1. `DEVELOPMENT_STATUS.md` — current evidence, limitations, operational
   boundary, and exact resume point.
2. `docs/rfd3_mosaic/RFD3_MULTI_INTERFACE_SEED_FINAL_PLAN.md` — method
   architecture, data model, compiler/runtime separation, and success criteria.
3. `docs/rfd3_mosaic/USER_CLI.md` — routine validated configuration and
   submission workflow.
4. `docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md` — the current
   opt-in experiment that allows bounded scaffold-guided seed motion.
5. `docs/rfd3_mosaic/C5_C6_C7_200STEP_RUNBOOK.md` — reproducible C5/C6/C7
   pose generation, P100 inference, and audit commands.

For historical comparison with Interface-Seed 1.0, then read
`docs/rfd3_mosaic/INTERFACE_SEED_RFD1_UPGRADE_AUDIT.md`.

## Environment contract

The shared `rc-foundry` environment must not be modified with editable
installs. Activate it and point Python at this checkout:

```bash
source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"
cd /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:$PYTHONPATH"
export FOUNDRY_CHECKPOINT_DIRS=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints
```

Checkpoint used:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
```

Cluster-operation boundary:

- Codex may modify local code, scripts, tests, and documentation, but cannot
  operate the LRZ server directly.
- The user performs every local-to-LRZ synchronization, `sbatch` submission,
  job cancellation, and GitHub push.
- After every local file change, Codex must provide directly executable
  synchronization commands and the necessary server-side verification
  commands.
- Documentation-only changes do not require a standalone LRZ synchronization.
  They remain pending locally and travel with the next code/script sync batch;
  Codex must identify them as pending instead of repeatedly asking the user to
  sync documentation by itself.
- Providing a command must never be described as having synchronized,
  submitted, cancelled, pushed, or executed it.

## 2026-07-30 end-to-end milestone achieved

The static exact-C3 Interface-Seed pipeline has now completed end to end on
LRZ, including compilation, deterministic linker materialization, native RFD3
input construction, runtime prevalidation, checkpoint inference, result
serialization, seed-integrity auditing, transform-aware scaffold auditing, and
the final audit gate.

The principal 200-step result is job `5721371`:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c3_full/5721371
```

Key result:

- all three cross-protomer seed pairs passed;
- maximum all-atom seed RMSD was `0.000511 A`;
- maximum CA seed RMSD was `0.000489 A`;
- atom completeness and contact retention were both `1.0`;
- all three 146-residue chains were continuous with zero chain breaks;
- compactness passed with maximum CA radius of gyration `23.460 A`;
- declared-transform C3 symmetry passed with coordinate RMSD `0.000119 A`
  and maximum coordinate error `0.000236 A`;
- copy-internal distance-matrix RMSD was `6.18e-6 A`.

This establishes the end-to-end engineering result: the complete cross-chain
interface seed can remain fixed while RFD3 generates a continuous, compact,
exact-C3 scaffold around it. The strict scaffold report is a near-pass rather
than a clean final acceptance because one local generated-linker/right-motif
CA overlap (`2.253 A`) is reproduced in all three C3 copies, producing three
reported clashes. This remaining candidate-quality issue does not invalidate
the demonstrated end-to-end pipeline, but a clash-free pose or diffusion seed
is still required before calling a design scientifically final.

## 2026-07-29 exact orbit-state refactor

The previous hard-motif patches preserved the interface seed but did not keep
the generated scaffold in one C3-invariant state. The current local refactor
replaces that bridge with an explicit orbit-space path:

- Fixed and generated coordinates use the same row-vector convention as
  AtomArray expansion: `x_copy = x_master @ R.T + T`. The previous runtime
  projector used `R` instead of `R.T`; for C3 this silently exchanged the
  `r1/r2` direction even though the unordered copy set still looked symmetric.
- `symmetry_state_mode=orbit_average` inverse-maps every copy, averages all
  copies in the master frame, and re-expands them. Noise is sampled once per
  master atom and rotation-copied to the other transforms. Initial, noisy,
  denoised, Euler-updated, compactness-guided, and final states are all checked
  for orbit closure.
- Exact operations require atom-key-verified `sym_orbit_slot` correspondence,
  run outside bfloat16 autocast, and retain float32 state precision instead of
  quantizing a 1e-3 A closure check back to bfloat16.
- Runtime frames reconstructed from RFD3's epsilon-stabilized virtual-frame
  encoding are validated and projected to the nearest proper SO(3) rotation
  before orbit operations. Online closure retains the 1e-3 A absolute gate at
  molecular scale and adds a float32/coordinate-scale roundoff floor only at
  the very large initial EDM noise scale.
- Arbitrary per-step realignment is disabled because the stored symmetry
  operators are not conjugated into the augmented frame.
- Adapter output now embeds the actual registry matrices and explicit
  constraint-orbit actions. Prevalidation calls the real
  `DesignInputSpecification.build -> AddSymmetryFeats.forward` path, matches
  runtime matrices one-to-one to the registry, and evaluates the same
  all-copy orbit projector used during inference. Missing groups/orbits,
  malformed matrices, non-finite targets, mask non-closure, or target
  residuals above threshold fail before checkpoint loading.
- A continuous linker range is materialized once by the adapter as an exact
  `N-N` contig length. The default is the configured integer midpoint
  (`70--100 -> 85-85` for LHD101), and `RFD3_LINKER_LENGTH` can select another
  in-range value. This prevents prevalidation and the separate inference
  process from independently sampling different AtomArray lengths. The
  standalone endpoint spans are rechecked against that exact materialized
  length rather than only the configured maximum.
- Scaffold symmetry acceptance is transform-aware. Copy-internal CA distance
  matrices remain a diagnostic only; final acceptance compares each copy's CA
  coordinates with the declared C3 transform of the master chain. A translated
  or wrongly rotated but internally congruent copy therefore fails. The current
  output-chain association is validated for the one-chain-ASU C3 baseline; it
  still uses sorted output chain order and is not yet a provenance-aware
  multi-chain/Dn audit.
- Smoke and full scripts write both seed and scaffold reports, then a separate
  audit gate fails the job if either report is scientifically rejected.
- The 10-step smoke job runs the complete repository unit suite on its
  allocated compute node before adapter/prevalidation or checkpoint loading.
  This avoids the observed DSS-backed `torch` import stalls on LRZ login
  nodes and makes a unit failure stop the same job before GPU inference.
- Complete-interface `orbit_rigid` schema, atom correspondence, and bounded
  SE(3) controller are connected through an explicit, default-off experimental
  sampler hook. Formal scripts do not enable it and keep the interface static:
  moving coordinates without
  updating RFD3's precomputed motif pair-conditioning creates contradictory
  geometry, and raw fixed-atom denoiser output is not yet a validated scaffold
  pose signal.

Local `compileall`, Slurm shell parsing, and `git diff --check` pass. The local
base Python lacks `pydantic` and the RFD3 runtime, so the new suite and the real
C3 build-forward-prevalidation chain still require LRZ validation. Job
`5720844` is retained as historical evidence for the old sampler only; it
cannot validate the corrected transform direction or exact-state path.

The next gate is: targeted CPU tests -> complete CPU discovery -> adapter plus
prevalidation (including deterministic linker provenance) -> 10-step static
smoke -> 50-step static pose screen -> 200-step static validation. Dynamic
motif mobility remains blocked until its
conditioning and pose signal are designed and tested explicitly.

LRZ smoke validation update: the synchronized exact-path suite reached 216
tests; 214 passed and two regressions stopped the job before inference. One
was a real aliasing bug in the legacy atomwise projector (`Tensor.to()` can
return the input, so its in-place COM correction mutated the caller); the
projector now clones before centering. The other was a test harness error: the
full fake sampler test called an inference API that correctly requires
`torch.no_grad()` without that context. Both fixes are local and require one
fresh LRZ smoke rerun; no GPU/scaffold result from that stopped job is valid.

## 2026-07-29 group-aware constraints and symmetry diagnosis

- The adapter now compiles every concrete interface-edge instance into a
  two-sided motif constraint group and records both standalone atom indices
  and post-symmetry RFD3 source-component/transform membership.
- `AddSymmetryFeats` resolves these groups only after RFD3 has constructed and
  expanded the AtomArray. It requires both interface sides to match and every
  fixed motif atom to be covered.
- The symmetry sampler restores complete groups at every opted-in denoising
  step. Overlapping groups are order-independent when their coordinates agree
  and fail explicitly when hard constraints conflict.
- C3 entry scripts require group metadata. `rfd3_prevalidate` now resolves the
  same runtime memberships before checkpoint loading and reports group counts,
  sizes, and fixed-atom coverage.
- Symmetry projection now works out-of-place, verifies equal atom counts for
  ASU/target subunits, and transform-frame construction no longer assumes
  atoms are consecutively ordered by transform ID.
- Local compilation, shell parsing, and whitespace checks pass. LRZ
  `rc-foundry` validation subsequently passed all 168 unit tests on
  2026-07-29, including runtime cross-chain group resolution, transform-order
  independence, complete fixed-atom coverage, and conflicting-overlap
  rejection.
- LRZ adapter prevalidation then passed with three resolved C3 constraint
  groups of 496 atoms each, covering all 1488 fixed motif atoms, with no
  failures. One unguided 200-step LRZ run remains required; these CPU checks
  are not yet an end-to-end scientific result.
- The single-design entry point accepts `RFD3_NUM_TIMESTEPS` in the range
  2--200 while retaining 200 as its default. Shorter runs are diagnostic only;
  the next intermediate screen uses three geometrically distinct poses at
  50 steps before the definitive 200-step gate.
- Full 200-step job `5720844` passed all three interface-seed pairs
  (maximum CA RMSD 0.038 A, maximum all-atom RMSD 0.041 A, minimum contact
  retention 0.975, and complete atom coverage) but failed scaffold geometry.
  Chain A had only three continuity failures, whereas chains B and C each had
  94, mostly implausibly short generated-region bonds; the output also had
  3031 intra-chain and 613 inter-chain CA clashes. This chain-asymmetric
  collapse is incompatible with rigid C3 copies and points to a remaining
  ASU-to-copy atom/coordinate correspondence problem, not merely insufficient
  diffusion or a need for motif rigid-body mobility.
- Pairwise CA distance-matrix comparison confirmed the loss of rigid C3
  geometry (A--B RMSD 14.636 A; A--C RMSD 19.128 A). The concrete sampler
  cause is that upstream projects the denoised prediction, then combines it
  with independently noised coordinates in the Euler update; the actual
  state advancing to the next step is therefore not guaranteed symmetric.
  Hard-motif mode now reprojects that updated state at every step and restores
  the complete cross-chain groups afterwards, including when compactness is
  disabled. LRZ subsequently passed all 169 unit tests, including the
  compactness-disabled updated-state projection regression. A single
  pose-2131 50-step controlled rerun was the next gate at that historical
  stage; the exact orbit-state refactor above now requires CPU/preflight
  validation and a fresh 10-step smoke first.

## 2026-07-29 difficulty assessment and revised sampler architecture

The problem is not a single motif-write ordering bug. Three constraints must
hold simultaneously:

1. each cross-chain interface retains its internal all-atom geometry;
2. the full generated assembly remains exactly C3;
3. generated scaffold junctions remain continuous and physically plausible.

The earlier patches exposed the following interactions:

- A final native symmetry projection restores C3 but may reconstruct the two
  sides of an indexed cross-chain interface separately.
- Final-only motif reinsertion preserves the seed but creates a late
  motif-linker coordinate jump.
- Stepwise static motif restoration preserves the seed throughout diffusion,
  but skipping the native final projection allowed non-symmetric Euler-state
  errors to survive.
- Projecting the denoised prediction alone is insufficient because
  `X_noisy_L` contains independent atomwise noise.
- The pre-refactor C3 scripts set `allow_realignment=True`. That applied an
  arbitrary global rotation/translation every step, while the stored symmetry
  operators remain in their original frame. Unless the operators are
  conjugated by the same global transform, the scaffold projector and motif
  targets are expressed in incompatible coordinate systems. Upstream defaults
  `allow_realignment` to false; this is a local integration hazard, not
  evidence that ordinary upstream C3 generation is broken.
- The 169 passing tests cover group resolution and update-state projection,
  but do not yet prove true C3 closure with real transforms or the
  realignment/frame invariant.

The LHD101 constraints themselves are mathematically C3-compatible. With
`B=right(0)` and `C=left(1)` in the selected ASU, the runtime interface groups
must be:

```text
group(0) = B@0 + C@2
group(1) = B@1 + C@0
group(2) = B@2 + C@1
```

The adapter emits exactly this relation. The remaining risk is transform
matrix numbering/direction inferred by RFD3, which is not established merely
by observing transform IDs `[0, 1, 2]`.

The recommended formal design is orbit-space constrained diffusion:

- maintain the full-copy tensor projected onto one master-equivalent degree
  of freedom rather than three independently drifting copies;
- sample Gaussian displacement noise once on the ASU and rotate-copy it to
  the other C3 members (translations do not apply to displacement vectors);
- project predictions by inverse-transforming all copies to the master frame,
  averaging them, and re-expanding them, rather than copying Chain A alone;
- keep all diffusion states inside the C3-invariant subspace so the Euler
  update is closed by construction;
- initially disable arbitrary realignment; a future augmented frame must use
  conjugated operators `A S_k A^-1`;
- represent one cross-chain master interface with a pose `Q_t in SE(3)` and
  generate its orbit as `S_k Q_t P`;
- first hold `Q_t` static, then add bounded Kabsch-derived rigid motion so the
  interface can adapt during diffusion without changing its internal geometry
  or breaking C3.

Required new preflight/online gates:

- match Mosaic transforms to actual RFD3 matrices, allowing explicit
  permutation/direction resolution;
- fixed-target projection RMSD <= 0.01 A and maximum atom error <= 0.03 A;
- per-step orbit-equivariance residual below numerical tolerance;
- final transform-aware C3 coordinate RMSD <= 0.01 A and maximum error
  <= 0.03 A; copy-internal distance matrices are diagnostic only;
- motif rigidity, contact retention, continuity, compactness, and clash gates
  all pass independently.

Implementation should proceed in three stages:

1. static master orbit with realignment disabled and exact C3 invariants;
2. bounded master-interface SE(3) mobility with an early/middle/late schedule;
3. multiple independent motif orbits with overlap merging or explicit conflict
   rejection.

The historical post-Euler-only patch was a diagnostic bridge. It is now
superseded by the exact all-copy orbit-average implementation described above;
that implementation has completed LRZ/GPU end-to-end validation for the
LHD101 one-chain-ASU C3 baseline. It is not yet a scientifically final general
architecture for Dn, multiple independent motif orbits, or dynamic mobility.

## Historical archive boundary

All dated sections below preserve the sequence of earlier experiments. Words
such as “current”, “latest”, and “next” inside those sections describe that
historical stage and are superseded by the exact-orbit status and gate at the
top of this file.

## 2026-07-24 scaffold-generation diagnosis and guided experiment

- Job `5712555` is a positive seed-preservation result only.  Its complete
  generated protomers are structurally invalid, so it is not an end-to-end
  Interface-Seed success.
- The native RFD3 contig is parsed as intended: one continuous
  `B1-31,70-100,C1-30` ASU chain is expanded into three independent C3-related
  chains.  The observed failure is therefore not an absent contig or an
  unintended covalently closed trimer.
- A separate scaffold audit now reports chain continuity, per-chain CA radius
  of gyration, and coarse nonlocal CA clashes.  Seed integrity and scaffold
  validity are intentionally independent acceptance gates.
- `SampleDiffusionWithSymmetry` now has optional, default-off Interface-Seed
  compactness guidance.  It applies a capped, token-rigid translation to
  generated residues toward the fixed anchors of their own chain, leaves all
  fixed motif atoms untouched, fades to zero before the final denoising
  quarter, and reprojects coordinates to native symmetry after each guided
  update.
- At that stage, `lhd101_c3_full_single.sbatch` enabled the first conservative
  diagnostic
  setting with `RFD3_COMPACTNESS_WEIGHT=0.02`,
  `RFD3_COMPACTNESS_END_FRAC=0.75`, and
  `RFD3_COMPACTNESS_MAX_STEP=0.5`.  All are environment-overridable and the
  sampler defaults remain zero/off for ordinary RFD3 jobs.
- Local syntax compilation, shell parsing, and `git diff --check` pass.  The
  local base Python lacks project dependencies (`torch`, `pydantic`, and
  `pytest`), so the full unit suite must be run in the LRZ `rc-foundry`
  environment before GPU submission.

## 2026-07-24 fixed-interface result and evidence boundary

- The upstream RFD3 symmetry sampler finalized structures in the order
  `motif reinsertion -> symmetry projection -> global rigid alignment`.
  Upstream symmetry handling also distinguishes indexed motifs from entities
  treated as fully fixed.
- The conclusion that this ordering caused the observed cross-protomer
  interface displacement is a Mosaic source-code analysis, not an upstream
  Foundry statement or officially documented bug.
- The local correction finalizes in the order
  `symmetry projection -> complete-motif reinsertion -> global rigid
  alignment`, so the complete cross-protomer seed receives the final
  coordinate write.
- LRZ smoke job `5712555` is the local positive validation of that correction:
  all three recovered interface pairs passed the seed-integrity audit. These
  measurements are local experimental results and must not be presented as
  official Foundry data.
- The next scientific gate is 200-step sampling across the selected,
  geometrically diverse pose manifests, followed by independent seed-integrity
  and scaffold-validity acceptance.

## 2026-07-24 junction-failure diagnosis and stepwise motif preservation

- Guided 200-step job `5713652` preserved all three cross-protomer seed pairs,
  but failed scaffold validation with 114 CA clashes and nine chain breaks.
- Every chain broke at the same motif-linker boundaries: residues 31--32 were
  separated by about 33.9 A and residues 124--125 by about 24.1 A. This shows
  that final-only motif reinsertion restored the interface by making a large
  last-step coordinate change after the linker had already been generated.
- The sampler now has an opt-in
  `preserve_fixed_motif_during_symmetry` mode. Every symmetry projection
  restores the complete fixed motif in the current augmented frame, and the
  coordinate update keeps those atoms equal to the motif coordinates seen by
  that denoising step. Later steps can therefore adapt the generated linker to
  the true interface geometry.
- In stepwise-preservation mode, finalization no longer performs another native
  symmetry projection before motif reinsertion. This avoids recreating the
  final motif-linker coordinate jump.
- All C3 entry points enable stepwise motif preservation. The full single-GPU
  script returns compactness guidance to a default weight of zero; guidance is
  now an explicit environment-enabled experiment rather than the baseline.
- New unit coverage requires symmetry projection to modify generated atoms
  while leaving the complete multi-fragment fixed motif unchanged, and
  requires stepwise-mode finalization not to invoke a second final projection.
- This correction still requires LRZ CPU tests followed by one controlled
  200-step GPU run using pose 2131, RFD3 seed 42, and compactness disabled.
  Acceptance requires both seed integrity and junction continuity; seed
  preservation alone is insufficient.

## Completed

- Forked Foundry and established the feature branch.
- Confirmed the fork's `models/rfd3/src/rfd3` is imported through
  `PYTHONPATH`.
- Completed a native RFD3 smoke test on one P100 GPU using `dsDNA_basic`, one
  sample, ten diffusion steps, and low-memory settings.
- Defined strict Interface-Seed v2 schema objects for fragments, motion
  groups, interface ports, target geometry, symmetry orbits, interface edges,
  and directed scaffold links.
- Added cross-reference validation for ownership, ports, interfaces, orbits,
  and scaffold endpoints.
- Added and validated the LHD101 C3 configuration.
- Implemented core SE(3) operations and tests: validation, construction,
  inverse, composition, coordinate application, and axis-angle rotation.
- Implemented a generic cyclic Cn transform registry with stable transform
  IDs, arbitrary axis/center, signed orbit-offset resolution, group-element
  composition, and closure checking.
- Implemented the proper rotational Dn transform registry locally. It emits
  stable `Dn:e/rk/sk` IDs, accepts a configurable perpendicular two-fold axis,
  contains all `2n` proper rotations, preserves the requested center, and
  keeps cyclic orbit offsets inside each Dn coset.
- Added C3/C4/C5 closure, center, offset, composition, and master-copy drift
  tests.
- Server verification passed on 2026-07-21: all 38 unit tests passed in the
  shared `rc-foundry` environment (`Ran 38 tests ... OK`).
- Symmetry-orbit expansion into motion-group, fragment, and port instances was
  subsequently verified without failures in `rc-foundry`.
- The object-level `MappingRegistry` and its provenance tests were subsequently
  verified without failures in `rc-foundry`.
- Directed scaffold expansion and topology validation were subsequently
  verified without failures in `rc-foundry`.
- Deterministic PDB parsing, atom selections, and synthetic interface-frame
  tests were subsequently verified without failures in `rc-foundry`.
- The real LHD101 reference fixture, fragment selections, and port-frame
  integration were subsequently verified without failures in `rc-foundry`.
- Standalone CIF/mapping/manifest emission was verified on 2026-07-22 as part
  of the complete server suite (`Ran 80 tests in 0.429s`).
- Corrected C3 motif placement was regenerated and visually inspected on
  2026-07-22. The three motif pairs are separated C3-related copies rather
  than a collapsed, overlapping cluster.
- Audited RFD3's symmetry input path: RFD3 builds one ASU from the contig and
  then expands it with frames inferred from the full pre-symmetrized motif.
- Implemented the first static native-RFD3 adapter. It emits
  `rfd3_input.json` with an ASU contig, strict all-atom motif coordinate
  fixing, fixed motif sequence identity, and native C3 symmetry metadata.

## In progress

- Visual and quantitative inspection found that the first emitted C3 artifact
  was invalid: the unplaced seed centroid was only 2.52 A from the symmetry
  axis, giving a 0.182 A minimum inter-copy distance and 1,545 atom pairs below
  2.0 A across the three copy pairs.
- Corrected master-pose initialization (COM centering, explicit orientation,
  radial/axial placement) and a mandatory inter-group clash gate are
  implemented; the regenerated artifact passed visual separation inspection.
  The complete updated server test count still needs recording.
- Topology audit corrected an important semantic error: the preserved 7mwr
  A/B interface is same-copy (`interface orbit_offset: 0`), while each designed
  protomer connects one interface half to the geometry-selected neighboring
  copy. For the current LHD101 fixture this resolves to
  `right(k) -> left(k+1)` (`scaffold orbit_offset: +1`); the direction must not
  be hard-coded as a universal rule.
  InterfaceEdge instances and required-edge geometry diagnostics are
  implemented. The complete updated server test count still needs recording.
- Five static-adapter unit tests are implemented locally and pass syntax and
  whitespace checks. They await `rc-foundry` testing and native RFD3 input
  construction/prevalidation on LRZ.
- Added an RFD3-runtime prevalidation command that loads the emitted JSON and
  CIF, runs `DesignInputSpecification.build(return_metadata=True)`, verifies
  C3 chain/transform multiplicity, equal per-chain residue counts, recognized
  motif/fixed atoms, and ASU annotations, then writes
  `rfd3_prevalidation.json`. Four dependency-independent report-logic tests
  were added; the complete suite now contains 96 tests.
- Server verification passed on 2026-07-22: the complete updated suite ran all
  96 tests in the shared `rc-foundry` environment (`Ran 96 tests ... OK`).
- The static adapter successfully generated `presymmetrized_input.cif`,
  `mapping.json`, `manifest.json`, and `rfd3_input.json` on LRZ with the
  intended cross-copy ASU topology.
- The first runtime prevalidation exposed a residue-number namespace bug:
  AtomWorks selects mmCIF residues by `label_seq_id`, while the first adapter
  emitted original PDB `auth_seq_id` values (`B211-241` and `C165-194`). The
  adapter now correctly emits `B1-31,70-100,C1-30`; original author numbering
  remains preserved in `mapping.json` for provenance. Server revalidation is
  complete.
- Native RFD3 atom-array construction passed on LRZ on 2026-07-22. It produced
  three chains (`A`, `B`, `C`), 155 residues per chain, 1,488 recognized motif
  atoms, 732 fixed backbone atoms, and symmetry transform IDs `[0, 1, 2]`.
  The sampled ASU linker length was 94 residues (`31 + 94 + 30 = 155`).
- Re-audited the original Interface-Seed oligomer topology after questioning
  whether the entire ring should be covalently connected. The original code
  expands one contig per symmetry copy and separates those contigs as distinct
  chains; downstream notebooks explicitly design chains `A B C`. For C3 the
  intended topology is therefore three independent protomer chains:
  `B0-linker-A1`, `B1-linker-A2`, and `B2-linker-A0`. The preserved seed
  interfaces `A0:B0`, `A1:B1`, and `A2:B2` are noncovalent and assemble the
  three chains into the ring. The current native RFD3 adapter matches this
  topology; it does not create one covalently closed chain.
- Chain-colored inspection of the native smoke output confirmed this topology
  visually: cyan, magenta, and yellow form three separate protomer chains;
  each chain spans between two interface lobes, while each lobe contains a
  noncovalent motif contact between two differently colored chains. The loose
  appearance is therefore a sampling/linker-quality issue, not an accidental
  covalently closed C3 chain.
- Added a tracked Slurm script for a one-design, ten-timestep native C3 smoke
  test. The script repeats prevalidation inside the allocation before loading
  the checkpoint.
- The first smoke job (`5711261`) was submitted to the A100 partition but
  remained pending for priority. The tracked script was changed to allow the
  available V100/P100 partitions requested for faster scheduling, while
  retaining batch size one and RFD3 low-memory mode. P100 still carries an OOM
  risk for the 465-residue complex.
- The native ten-timestep C3 GPU smoke test subsequently completed and emitted
  a structure. Visual inspection shows three C3-related lobes, but the sampled
  94-residue linkers are loose and loop-rich. This confirms the execution path,
  not design quality: ten diffusion steps are far below the normal 200-step
  inference schedule. Output metadata metrics and motif preservation still
  need quantitative review before selecting full-run parameters.
- Quantitative review confirmed the ten-step output is an unconverged smoke
  structure: 209 chain breaks, 611 inter-residue clashes with sidechains, 70
  backbone clashes, 100% loop assignment, zero secondary-structure elements,
  and 13.39 A maximum CA deviation. The inference build sampled a 90-residue
  linker (`453 / 3 - 61 = 90`), independently of the earlier prevalidation
  sample. These failures justify a normal 200-step baseline before changing
  the original 70--100 linker range.
- Added a tracked, fixed-seed (`42`), one-design, 200-timestep Slurm script for
  the first full-quality baseline. It retains low-memory mode and the GPU
  partitions proven to schedule the smoke run.
- Added a second full-quality script for LRZ's `lrz-hgx-h100-94x4` partition.
  It runs one 200-step sample with seed `43`, records PyTorch/CUDA compute
  capability at startup, disables low-memory mode to use the H100's available
  memory, and writes to a separate `native_c3_full_h100` run tree. This makes
  it a useful replicate rather than duplicating the seed-42 legacy-GPU job.
- Live `sinfo` output on 2026-07-22 confirmed the current partition spelling is
  `lrz-hgx-h100-94x4` (the earlier public training material showed `92x4`), so
  the tracked H100 script was corrected before submission. At the same time,
  the seed-42 full baseline job `5711276` was running on `p100-001`.
- Reduced the H100 single-design walltime request from 12 hours to 2 hours.
  Walltime is only an upper bound, but the shorter request is more realistic
  for one 200-step H100 inference and may improve backfill scheduling.
- Keeping local, GitHub, and LRZ server copies synchronized.
- The new Dn registry, schema dispatch, D2/D3/D5 closure tests, and D3
  instance-expansion test are implemented locally. Syntax and whitespace
  validation pass; the local system Python lacks Pydantic, so the complete
  suite still needs to be rerun in the server `rc-foundry` environment.
- Implemented named group-element copy relations locally. Configuration can
  now use `copy_relation.transform: D3:s0`; relations act as
  `target = relation @ source`, allowing deterministic pairing between the
  two Dn cyclic cosets. The schema now accepts canonical transform IDs with a
  colon, and both interface and scaffold compilation resolve them.
- Extended standalone prescreen diagnostics locally. Clash reports are now
  separated into cyclic, Dn intra-coset, and Dn inter-coset pair classes;
  scaffold links report terminal-anchor distance and a conservative maximum
  contour feasibility estimate; each symmetry orbit reports central void and
  principal-axis clearance descriptors. These diagnostics are written to the
  manifest and do not add uncalibrated rejection thresholds.
- Extended the static RFD3 adapter and prevalidation logic from Cn-only to
  native Cn/Dn symmetry, with an explicit guard for RFD3's current
  10-transform symmetric-motif limit. Adapter metadata records multiplicity
  and Mosaic transform order. Added an end-to-end D2 adapter fixture.
- Audited RFD3's native dihedral frame generator and found that D3 produced
  six entries but only four unique rotations. The fork now constructs Dn from
  one fixed perpendicular two-fold generator, preserving all `2n` unique
  proper rotations. Added D3/D6 uniqueness and D2/D3/D5 closure tests. No
  model architecture or checkpoint was changed.
- Added adapter-side Cn/Dn registry preflight for multiplicity, uniqueness,
  proper rotations, and group closure. Added a real no-linker adapter mode:
  `chain_break: true` emits `/0` and records a two-chain ASU instead of
  silently creating a continuous linker. Prevalidation now supports repeated
  multi-chain asymmetric units with unequal chain lengths.
- Extended the standalone CLI summary to print symmetry/copy count, hard
  clashes, linker-span feasibility, central void radius, and axis clearance
  without requiring manual inspection of `manifest.json`.
- Implemented the first backend-independent objective/scoring layer locally.
  Configurable minimize/maximize, upper/lower bound, target-with-tolerance,
  and range terms emit per-objective diagnostics and deterministic ranking
  keys that prioritize required-constraint feasibility. Static compiler
  diagnostics are exposed through stable metric names.
- Added relaxed standalone compilation (`strict_validation=False` or CLI
  `--allow-infeasible`) so pose search can retain, diagnose, and rank invalid
  candidates. The RFD3 adapter continues to use strict validation.
- Reframed the implementation order around general software capabilities:
  objective API -> static pose search -> symmetry feasibility screening ->
  conflict diagnostics -> dynamic controller. The final plan now explicitly
  separates reusable features from C3/D2/D3 benchmark fixtures.
- The user reported that the normal-timestep native C3 RFD3 run completed and
  supplied a structure image on 2026-07-22. Visual inspection shows three
  assembly lobes and substantially formed secondary structure, but also long,
  extended inter-motif scaffold regions.
- Quantitative audit of the fixed-seed (`42`) 200-step P100 result from job
  `5711276` is complete. Relative to the ten-step smoke result, maximum CA
  deviation improved from 13.39 A to 0.876 A, internal chain breaks from 209
  to 3, sidechain-inclusive clashes from 611 to 9, and backbone clashes from
  70 to 3. The structure contains 18 secondary-structure elements with 38.7%
  helix, 29.7% sheet, and 31.5% loop instead of the smoke result's 100% loop.
  This validates normal-timestep convergence and strong motif preservation.
- The full result contains 462 residues, or 154 residues per C3 protomer. With
  61 indexed motif residues per protomer, the sampled linker is 93 residues,
  confirming that its long visual appearance follows the configured `70-100`
  linker range rather than a contig parsing failure. Compactness therefore
  remains an objective/configuration question.
- RFD3's `n_chainbreaks` metric explicitly zeroes normal inter-chain
  boundaries before counting deviations greater than 0.75 A from the standard
  3.8 A CA spacing. The remaining count of 3 therefore represents internal
  continuity defects, not the expected boundaries between chains A, B, and C.
  The run passes execution, motif-fidelity, and fold-formation checks, but
  topology continuity and sterics remain partial rather than fully accepted.
- After the complete local-to-LRZ source synchronization on 2026-07-22, the
  `rc-foundry` environment passed all 127 discovered Mosaic unit tests in
  3.677 seconds. This includes the Cn/Dn registry and RFD3 frame tests,
  adapter/prevalidation tests, standalone output tests, and objective/scoring
  tests.
- A subsequent method-level audit identified a critical acceptance gap. The
  legacy Interface-Seed implementation applies one rotation and translation
  to the complete two-fragment reference interface and only then symmetry-
  copies that intact rigid seed. Mosaic's schema expresses the same intent by
  placing `left` and `right` in one rigid `primary_seed` motion group, and its
  standalone interface-edge check validates the compiled relative pose.
  However, the current tests do not yet prove that this intact contact survives
  the full `presymmetrized_input.cif -> native RFD3 symmetric-motif build ->
  generated structure` path. In particular, supplying an already expanded
  motif together with native RFD3 symmetry may be interpreted differently
  from the legacy single-seed expansion. Until fixed-motif contact retention
  is measured at each boundary, the completed 200-step run is an execution
  and folding baseline, not a validated Interface-Seed reproduction.
- Hardened all tracked C3 Slurm entry points against stale compiler artifacts.
  Each allocation now recompiles its own adapter JSON, pre-symmetrized CIF,
  mapping, and manifest under the job-specific run directory before
  prevalidation and inference. All three scripts explicitly select
  `inference_sampler.kind=symmetry`; reusing the earlier shared
  `lhd101_c3_adapter` directory is no longer allowed. This does not replace the
  pending RFD3-built and final-model seed-contact audits.
- Compared the legacy smoke output from job `5711263` with the fresh-adapter,
  symmetry-sampler, fixed-seed (`45`) smoke output from job `5711563`. They are
  not identical: the sampled scaffold lengths are 90 and 78 residues (453 and
  417 total residues), and their coordinates and metrics differ. Nevertheless,
  both exhibit the same method-level failure. Their cyclic endpoint pairing is
  consistent (`A_start:B_end`, `B_start:C_end`, `C_start:A_end`), but the seed
  halves are catastrophically overlaid. The reference LHD101 seed has a minimum
  inter-fragment CA distance of 4.223 A and 34 CA pairs below 8 A; job `5711263`
  gives minima of 0.55--0.66 A with 130--132 pairs below 8 A, while job
  `5711563` gives 1.17--1.24 A with 112--114 pairs below 8 A. Fixed motif
  backbones must not acquire such geometry even in a ten-step smoke run. This
  disproves stale input or an unlucky random seed as the sole cause and points
  to the adapter/RFD3 symmetric-ASU coordinate interpretation. Further GPU
  sampling is blocked until RFD3-build seed geometry is audited and corrected.
- Tightened the reproduction baseline from backbone-only motif conditioning to
  strict all-atom seed freezing. Both LHD101 interface fragments now compile as
  `select_fixed_atoms: ALL`, while `redesign_motif_sidechains` remains false,
  for C3 and the D2/D3 dry-run configurations. Rigid-body initialization may
  still rotate/translate the complete two-fragment seed before a job is built,
  but no atom within that placed seed may move during RFD3 denoising. The RFD3
  prevalidator now rejects a job unless every recognized motif atom has both a
  fixed coordinate and fixed sequence identity. This is a pre-GPU hard gate,
  not merely a reported metric.
- Server verification after the strict all-atom update passed on 2026-07-22:
  the `rc-foundry` environment discovered and passed all 129 Mosaic unit tests
  in 3.244 seconds.
- Replaced the single fixed `[0, 0, 0]` LHD101 pose with reproducible
  Haar-uniform SO(3) rigid orientation sampling and a 20--30 A radial interval.
  The complete two-fragment seed remains all-atom fixed; sampling applies one
  shared SE(3) transform and therefore cannot alter its internal PPI geometry.
  Job-specific `--pose-seed` overrides are now recorded together with the
  sampled quaternion, rotation matrix, radius, axial offset, and centers.
- Added a CPU-only `rfd3_mosaic.pose_ensemble` compiler. It generates many
  deterministic pose candidates and ranks them before GPU use, rejecting hard
  clashes, unsatisfied required interfaces, infeasible continuous link spans,
  and required-objective failures. The C3 smoke/full Slurm scripts now pass
  their job seed into rigid-pose compilation instead of silently reusing one
  fixed pose for every run.
- The first 64-pose server ensemble (pose seeds 1000--1063) completed with
  64/64 candidates accepted. This validates deterministic SO(3)/radius
  sampling, but it also exposed an under-discriminating first-pass score: the
  70--100-residue contour gate is only a necessary reachability check, no
  objectives were configured, and the old final tie-break incorrectly
  preferred greater inter-group separation. The scorer now exposes minimum,
  mean, and maximum linker endpoint spans plus the maximum contour-derived
  residue requirement. The LHD101 example applies two explicitly soft
  shortlist heuristics (minimize the worst endpoint span and constrain the
  central-axis opening to a configurable soft window after hard gates), and
  the generic fallback tie-break minimizes the
  worst linker span rather than maximizing seed separation. These scores rank
  GPU candidates; they are not evidence of foldability or designability.
- The corrected 64-pose rerun produced nonzero discriminating scores. Its old
  top pose (seed 1010) combined a 25.640 A worst linker span with only 2.267 A
  axis clearance, exposing a second ranking-direction issue: unboundedly
  minimizing the central opening rewards near-axis placement. The LHD101
  example now uses an explicitly heuristic 6--14 A soft clearance window
  instead. This range is example configuration, not a universal Cn rule. Seed
  1058 (24.623 A worst span, 10.639 A clearance) is the provisional geometric
  leader before orientation-diversity selection and RFD3 validation.
- Added `rfd3_mosaic.pose_select`, which reads an existing ensemble without
  recompiling candidates. It preserves geometry-score order but suppresses
  near-duplicate orientations using the sign-invariant geodesic angle between
  sampled unit quaternions. Pool size, shortlist size, minimum SO(3) angular
  separation, and (for future multi-group inputs) the diversity group are all
  explicit CLI parameters. It never silently fills a shortlist with candidates
  that violate the requested diversity threshold.
- Visual review and the v3 ranking showed selection collapse toward the lower
  half of the 20--30 A radius interval; the top ten all lay below 25 A. This is
  not a failure of random-number generation: Haar SO(3) naturally places more
  principal axes near transverse orientations, while a single compactness
  score couples radius and orientation by preferentially retaining poses with
  short linker spans. The ensemble compiler now supports reproducible joint
  Latin-hypercube sampling of radius, axial offset, and the three Shoemake SO(3)
  unit variables. This preserves Haar orientation marginals while providing
  space-filling finite-sample coverage across all pose inputs.
- Added a coordinate-invariant visual-tilt diagnostic. Each rigid motion group
  receives a deterministic longest PCA axis from all source-seed coordinates;
  the manifest records its source/world vectors and its sign-invariant 0--90
  degree tilt relative to the symmetry axis. Degenerate PCA cases are reported
  as unavailable rather than assigned an arbitrary axis.
- Added `rfd3_mosaic.pose_stratify`. It reads an ensemble and retains the best
  accepted pose independently in each configurable radius-by-principal-tilt
  cell. The LHD101 defaults use four radius strata across 20--30 A and four
  equal tilt strata across 0--90 degrees. Empty cells and candidates outside
  configured bins remain explicit in the coverage report. This prevents one
  compact, highly tilted family from monopolizing the shortlist; the bins are
  exploration controls, not biological acceptance thresholds.
- Server verification of the joint sampler initially found one stale unit-test
  assumption rather than a compiler defect: the standalone test still required
  an exact 25 A radius even though the LHD101 configuration now samples 20--30
  A. The test now compares the emitted structure center with the provenance
  `sampled_radius` and independently checks the configured interval, preserving
  both deterministic auditability and the intended variable-radius behavior.
- The v4 256-pose joint ensemble occupied all 16 configured radius-by-tilt
  cells. Principal tilts span 1.542--74.685 degrees in the reported cell
  representatives, and radii span 20.144--29.837 A. This is the first direct
  evidence that finite-sample coverage no longer collapses to one compact,
  highly tilted pose family.
- Closed a provenance gap between CPU search and GPU inference. A Latin-
  hypercube candidate cannot be reconstructed by passing its integer pose seed
  alone because its explicit unit samples override the ordinary RNG stream.
  The RFD3 adapter now accepts a candidate manifest, validates the config hash,
  recovers the exact per-group unit samples, rebuilds the structure, and fails
  unless the rebuilt CIF SHA256 exactly matches the searched candidate. All C3
  Slurm entry points accept `RFD3_POSE_CANDIDATE_MANIFEST` and otherwise retain
  their legacy seed-based behavior.
- Historical, now superseded: the earlier C3 inference entry points explicitly
  set
  `inference_sampler.allow_realignment=True` and
  `+inference_sampler.insert_motif_at_end=True` (Hydra append syntax), in
  addition to compiling every
  motif atom as fixed-coordinate/fixed-sequence. In the RFD3 symmetry sampler,
  this is required to reinsert the ground-truth indexed motif during diffusion
  and at the final step. `allow_realignment=False` only suppresses coordinate
  noise; it does not make an indexed motif a hard positional constraint.
  The post-generation coordinate/contact audit verified the cross-chain
  interface seed directly. Current exact-orbit entry points instead disable
  realignment and do not rely on end-only motif insertion.
- Added `rfd3_mosaic.rfd3_seed_audit`, a generator-output audit that combines
  the adapter mapping with RFD3's `diffused_index_map`, recovers the two
  original source fragments, and searches for the best one-to-one cross-chain
  pairing among generated protomers. It reports per-seed all-heavy-atom RMSD,
  CA RMSD, atom completeness, reference-contact retention, and contact-distance
  RMSE. Same-chain fragment pairs are never accepted. Default acceptance
  requires CA RMSD <= 0.5 A, all-heavy-atom RMSD <= 0.75 A, at least 99% atom
  completeness, and at least 90% retention of reference contacts within
  4.5 A.
- Added a dependency-light RFD3 mmCIF/mmCIF.gz atom-site reader for this audit.
  A recovered pre-fix ten-step result was used as a negative control: all three
  inferred cross-chain seeds failed, with maximum CA RMSD 1.835 A, maximum
  all-atom RMSD 3.246 A, and minimum contact retention 0.473. This confirms the
  audit detects the original fixed-motif failure instead of passing it through
  symmetry alone.
- All three C3 Slurm entry points run seed and transform-aware scaffold audits
  after inference. Both JSON reports are written before a separate audit gate
  marks the job failed when either scientific check is rejected.
- Corrected seed-2153 job 5712416 reached the first realignment step but P100
  rejected `torch.linalg.svd` on a bfloat16 covariance
  (`svd_cuda_gesvdjBatched not implemented for BFloat16`). The shared Kabsch
  utility now promotes only the alignment solve to float32 for float16/bfloat16
  callers, retains float64 when requested, and casts aligned coordinates back
  to the caller dtype. This preserves hard motif reinsertion instead of
  disabling realignment. CPU bfloat16 regression coverage was added both to
  Foundry's alignment tests and to the Mosaic unittest discovery suite.
- Smoke job 5712530 showed that dtype promotion alone was insufficient:
  RFD3's outer bfloat16 autocast converted the float32 covariance-producing
  `einsum` back to bfloat16. The complete fix now disables autocast only around
  the small Kabsch covariance/SVD/rotation block; regression tests execute the
  bfloat16 call from inside an autocast context to reproduce the actual sampler
  call path.
- On 2026-07-23 the latest sampler finalization, BF16 alignment, seed audit,
  and associated local changes were synchronized to the LRZ working tree.
  Server-side unittest discovery completed successfully: 151 tests ran in
  3.719 seconds and all passed. This cleared the CPU test gate. The subsequent
  10-step GPU smoke job `5712555` passed all three cross-chain seed-integrity
  checks; full 200-step scaffold-quality validation remains outstanding.

### Why the fixed-interface finalization patch is necessary

The previous symmetry-sampler finalization order was:

```text
reinsert ground-truth fixed motif
-> apply native symmetry projection
-> globally rigid-align the result to the motif
```

An indexed interface motif can contain fragments on different protomers.
Native symmetry projection can therefore apply different transforms to the
two fragments. Each fragment remains internally correct, but their relative
cross-chain pose—and consequently the original interface contacts—is
destroyed. A final global rigid alignment cannot repair this: one global
rotation and translation cannot simultaneously invert two different
per-protomer transforms.

`SampleDiffusionWithSymmetry._finalize_with_fixed_motif()` changes the order
to:

```text
apply native symmetry projection to the generated scaffold
-> reinsert the complete ground-truth fixed motif as one coordinate set
-> globally rigid-align using all fixed motif atoms
-> return without another symmetry projection
```

This gives the complete fixed interface the final coordinate-write
precedence. The patch does not alter the contig, chain topology, checkpoint,
network weights, or C3 transform definitions.

The accompanying Kabsch change in `src/foundry/utils/alignment.py` performs
only the covariance/SVD/rotation solve in float32 (or preserves float64),
explicitly outside the outer bfloat16 autocast context, and casts aligned
coordinates back to the caller dtype. This avoids the P100 bfloat16 SVD
failure without converting the full inference calculation to float32.

Passing CPU tests proves the intended ordering and mixed-precision code path,
not the scientific result. GPU acceptance still requires all three recovered
cross-chain interface copies in `seed_integrity_audit.json` to satisfy atom
completeness, RMSD, and reference-contact-retention thresholds. Linker
junction geometry and final C3 consistency must also be checked after motif
reinsertion.

## Canonical pose-to-200-step workflow

Canonical ensemble:
`.../runs/rfd3-mosaic/lhd101_c3_joint_lhs_v4` (256 joint Latin-hypercube
poses; 16/16 radius-by-tilt cells occupied).

The first 200-step multi-pose batch is fixed to six v4 stratified candidates:

| pose seed | rank | radius (A) | tilt (deg) |
| ---: | ---: | ---: | ---: |
| 2131 | 1 | 20.499 | 41.153 |
| 2153 | 2 | 20.144 | 65.476 |
| 2003 | 15 | 22.428 | 6.144 |
| 2248 | 25 | 25.380 | 33.776 |
| 2213 | 31 | 25.618 | 57.702 |
| 2200 | 35 | 27.782 | 74.685 |

Key rules:

- all pairwise SO(3) distances are >=30 degrees (minimum 31.241 degrees);
- pass each candidate's `manifest.json`, not its integer pose seed;
- hold `RFD3_SEED=42` constant for the first comparison;
- each job must pass `seed_integrity_audit.json`;
- after seed preservation, evaluate junctions, chain breaks, clashes, fold
  quality, and C3 consistency.

- GPU smoke job 5712555 is the first positive seed-preservation validation of the
  corrected finalization order. Its `seed_integrity_audit.json` passed all
  three one-to-one cross-chain interface pairs (`A:B`, `B:C`, and `C:A`).
  All pairs had 496/496 matched heavy atoms (1.0 completeness), maximum CA
  RMSD 0.053180 A, maximum all-heavy-atom RMSD 0.048469 A, and minimum
  4.5 A reference-contact retention 0.978799. Contact-distance RMSE was at
  most 0.035735 A. This is a decisive positive control against the previous
  approximately 12.1 A combined CA RMSD and zero-contact-retention failure.
  The complete cross-chain interface is therefore preserved for this local
  10-step test pose. This does not establish full scaffold quality or an
  upstream RFD3 bug fix. The next phase is independent 200-step validation across
  multiple geometrically and orientationally diverse candidate manifests;
  each run must pass its own seed audit.

- Direct audit of the newly downloaded generated CIF
  `rfd3_input_lhd101_c3_interface_seed_0_model_0.cif.gz` on 2026-07-23 proved
  that fixed-atom annotations and end-of-run reinsertion were still
  insufficient. Each fragment was internally preserved (approximately
  0.05--0.09 A CA/all-heavy-atom RMSD), but no cross-chain fragment pairing
  retained the reference interface: the best cyclic pairs were approximately
  12.1 A CA RMSD with zero retained 4.5 A reference contacts. Thus RFD3 had
  fixed two isolated fragment shapes, not the complete two-fragment interface
  seed. The output is a scientific failure even though inference completed.
- Root cause was found in the symmetry sampler's final operation order. It
  reinserted the ground-truth fixed motif and then applied the native symmetry
  projection, allowing that projection to move the two protomer-spanning
  fragments independently. The local sampler now projects the generated
  scaffold into symmetry first and reinserts/aligned the complete fixed motif
  last. A regression test deliberately separates two motif fragments during a
  mock symmetry projection and requires finalization to recover their original
  4 A cross-fragment separation. The correction is present on LRZ and passed
  all 151 server-side CPU tests. GPU smoke job 5712555 subsequently passed the
  local interface-preservation gate.
- Three P100 submissions (5711682--5711684) exited before Python startup even
  though their stderr files were empty. Their stdout terminated immediately
  after `nvidia-smi` reported a corrupted infoROM, and `set -e` interpreted its
  nonzero diagnostic exit as a fatal job error. GPU inventory logging is now a
  nonfatal conditional in every C3 Slurm entry point; PyTorch/CUDA loading and
  inference remain fatal, so real compute failures are still surfaced.
- Each C3 Slurm entry point now redirects stdout and stderr, after creating its
  job directory, to `$RUN_ROOT/$SLURM_JOB_ID/slurm-$SLURM_JOB_NAME-$SLURM_JOB_ID.{out,err}`.
  Adapter files, RFD3 outputs, validation reports, and logs therefore stay
  together; Slurm's bootstrap streams are sent to `/dev/null`.
- Upstream Foundry's RFD3 symmetry documentation supports pre-symmetrized
  C/D motifs via `inference_sampler.kind=symmetry`, `diffusion_batch_size=1`,
  and `symmetry.is_symmetric_motif=true`; it does **not** provide an
  Interface-Seed / `asy_motif` / `motif_drag` example. Mosaic therefore uses
  the upstream symmetry entry point while supplying the missing
  Interface-Seed-specific pre-expansion, cross-copy contig topology, and
  fixed-motif reinsertion explicitly. This is an adapter layer, not a claim
  that RFD3 natively implements the earlier RFdiffusion extension.

## Not completed yet

Current priority order:

1. Exact-sampler targeted and complete LRZ CPU validation passed on
   2026-07-29: all 216 tests passed. Intermediate and final symmetry checks
   call the production scale-aware orbit-closure gate, while the independent
   fixed-motif coordinate comparison retains its strict `1e-5 A` regression
   check.
2. Rebuild the LHD101 adapter input and pass real
   `DesignInputSpecification.build -> AddSymmetryFeats.forward` prevalidation.
   This passed on LRZ: the linker materialized to 85 residues, three 496-atom
   constraint groups covered all 1488 fixed atoms, maximum fixed-target orbit
   error was `6.93e-5 A`, RMSD was `3.28e-5 A`, and both transform and orbit
   audits reported no failures.
3. Run one 10-step static exact-C3 smoke. Both seed and transform-aware
   scaffold audits must pass; the smoke is a wiring gate, not a fold-quality
   claim. Job `5721328` stopped before denoising because Lightning transported
   C3 feature matrices as bfloat16: the resulting ~`2e-3` raw orthogonality
   error was rejected before bounded polar normalization. The duplicate raw
   gate is removed locally; strict prevalidation of the original frame,
   maximum `1e-3` polar correction, and strict normalized-SO(3) checks remain.
   A bfloat16 C3 transport regression test was added before rerunning.
   Job `5721335` then reached fixed-target validation and exposed the same
   Fabric conversion on coordinates (RMS `0.040 A`, max `0.148 A`). The local
   engine now retains the pre-transfer geometry and restores only exact-orbit
   coordinates, noise, transforms, and constraint targets as float32 on the
   accelerator; the neural network remains bf16 mixed precision and no
   scientific threshold is relaxed.
   Job `5721339` reproduced the identical residual because the first engine
   implementation retained only a shallow alias to the nested batch. The
   correction now takes detached tensor clones before Fabric transfer and has
   a regression test that replaces tensors inside the same nested object.
   Job `5721344` showed the same residual and no precision-restoration log:
   the engine-side Hydra override copy was not a reliable runtime-mode
   detector. Exact geometry is now detected from the verified batch contract
   (`sym_transform`, `sym_orbit_slot`, and `sym_orbit_slot_verified=true`),
   which is symmetry-family independent and leaves ordinary batches unchanged.
   The synchronized correction passed all 220 LRZ unit tests.
   Job `5721348` still lacked the restoration log because the full orbit
   contract is not yet present at the engine precision boundary. Geometry
   preservation is therefore now unconditional for RFD3 inference:
   coordinates/noise and any present transforms/targets are restored as
   float32 after Fabric transfer, while model operations remain under bf16
   autocast.
   Job `5721355` proved from the stack that Lightning `_FabricModule.forward`
   still sits after the engine restoration point and reapplies the trainer
   precision policy to model arguments. Exact-orbit inference now overrides
   Fabric trainer precision to `32-true` at engine construction; non-exact
   samplers retain checkpoint precision. This is keyed by orbit-average mode,
   not by a C3 symmetry ID. A constructor-wiring regression test now checks
   the actual `RFD3InferenceEngine -> BaseInferenceEngine` trainer override,
   rather than testing only the mode predicate.
   Job `5721362` then crossed the runtime compatibility gate and preserved the
   complete interface seed. Declared-transform symmetry passed with maximum
   coordinate RMSD `1.21e-4 A` and maximum error `2.37e-4 A`; copy internal
   distance-matrix RMSD was `6.57e-6 A`. Compactness also passed and CA clashes
   fell to 9. The 10-step scaffold itself remained chemically under-denoised:
   every symmetry-identical 146-residue chain had 73 continuity failures
   (219 total), so the audit correctly failed. This result validates exact
   C3 state propagation but is not a valid final scaffold.
   Convergence jobs `5721369`, `5721370`, and `5721371` then tested the same
   seed-45 pose at 50, 100, and 200 steps. All three preserved the complete
   interface seed with 100% contact retention, had zero chain breaks, passed
   compactness, and retained declared-transform C3 coordinate RMSD near
   `1.2e-4 A`. Each failed only one intra-protomer CA clash copied through the
   exact C3 orbit: a generated-linker residue contacted the terminal residue
   of the right fixed motif. The 200-step case was best (`2.253 A`) but still
   below the hard `3.0 A` cutoff. The threshold must not be relaxed; the next
   scientific task is pose/diffusion-seed screening for a clash-free scaffold.
4. Screen selected pose manifests and diffusion seeds at 50 steps for a
   clash-free scaffold, then promote the best candidate to 200 steps.
   The next screening set should use the experimental morphology-aware
   `rfd3_mosaic.pose_qd` shortlist. It preserves the validated Haar SO(3) and
   joint Latin-hypercube generator, keeps the existing ensemble rank as the
   quality order, and distributes GPU candidates across axis-clearance and
   axial/radial-aspect cells with a global SO(3) separation gate. The standalone
   manifest now records axial span, radial thickness, aspect ratio, covariance
   eigenvalues, and shape sphericity for each symmetry orbit. These descriptors
   are exploration coordinates, not designability thresholds.
   The first 512-pose trial accepted 506 candidates and covered 13 morphology
   cells, but unconstrained cell filling admitted ensemble ranks 480 and 492.
   QD eligibility is therefore now restricted by default to the top 25% of
   accepted ensemble-ranked poses before morphology and SO(3) diversification.
   This top-quarter rule remains a compute-priority heuristic, not a claim that
   shorter generated-scaffold endpoint spans are universally better. These
   spans connect fixed fragments belonging to one protomer across adjacent
   interface positions; they are not flexible linkers between assembled units.
   The preserved cross-protomer interface seeds mediate unit self-assembly.
   Without an explicit target assembly size, morphology cells are parallel
   experimental conditions.
   Position quality must be estimated with a paired 50-step screen that uses
   the same set of at least three diffusion seeds for every pose; a replicate
   succeeds only if seed integrity, declared-transform symmetry, continuity,
   and hard-clash audits all pass. Promote positions by replicate success rate
   and declared morphology goals, not by endpoint span alone.
CPU pre-screening now goes beyond span/contour: every generated-protomer
   boundary reports C/N terminal-tangent-to-chord angles, tangent and peptide
   plane relative angles, chord axial fraction/out-of-plane angle, minimum
   chord-to-axis clearance, and an interior straight-chord clearance from the
   other fixed motif atoms. These are configurable boundary-condition and path
   risk descriptors, not claims that the generated 70--100-residue scaffold
   will follow a straight line or fold successfully.
   A C5/C6/C7 capability suite is now prepared. Each order has an explicit
   config, the same Haar-SO(3) plus Latin-hypercube pose generator, the same
   QD selection policy, and one generic P100 200-step entry point. The radial
   distributions are not copied from C3: they use
   `R_n = R_3 sin(pi/3) / sin(pi/n)` so the sampled adjacent-copy chord range
   is preserved when the cyclic order changes. Absolute cavity objective
   windows are scaled by the same factor, while QD uses the dimensionless
   `minimum_axis_clearance / sampled_radius` descriptor. This prevents larger
   cyclic orders from being penalized or collapsed into one morphology bin
   merely because their ring radii are larger. The C5/C6/C7 runs remain
   capability experiments until their adapter prevalidation, full inference,
   seed, continuity, clash, compactness, and declared-transform symmetry
   audits all pass.
   A tracked H100 robustness-screen entry point now submits the controlled
   matrix C5/C6/C7 x top three QD poses x five diffusion seeds x 50 steps
   (45 jobs). It records every job ID and exact pose manifest in a timestamped
   TSV. This is the first large-scale estimate of pose- and diffusion-seed
   robustness; it does not replace the existing convergence controls or
   downstream sequence/structure validation.
5. Replace sorted-chain output association with provenance-aware copy mapping
   before claiming general multi-chain or Dn scaffold auditing.
6. Validate D2/D3 through the real build/prevalidation and GPU paths.
7. Design dynamic motif pair-conditioning and a scaffold-derived pose signal
   before enabling the experimental orbit-rigid hook in formal scripts.
8. Only after those gates, extend to multiple independent motif orbits,
   soft-rigid motion, ligand/metal constraints, negative design, and additional
   symmetry families.

## Current limitations

- Cyclic Cn and the proper rotational Dn registry are implemented and have
  passed the complete 127-test server suite. Dn has not yet passed native RFD3
  generation validation.
- Polyhedral T/O/I, helical screw symmetry, and user-supplied explicit
  transform sets are not implemented yet.
- `schema/states.py` and `topology/pose_graph.py` remain placeholders for the
  later dynamic-guidance phase.
- Standalone CIF output contains motif coordinates only. The three configured
  70--100 residue scaffold links are recorded in the manifest but do not yet
  have generated coordinates.
- Standalone atom/residue indices are not claimed to be RFD3 indices until the
  adapter reads the CIF and verifies its own mapping.
- Radial placement removes catastrophic overlap but does not by itself prove
  that every requested cross-copy interface pose is optimal; explicit
  interface-edge geometry validation remains required before RFD3 inference.
- No RFD3 model architecture or checkpoint has been modified or retrained.
- The exact static C3 sampler path has passed LRZ runtime and GPU end-to-end
  validation for the one-chain-ASU LHD101 C3 baseline.
- Native C5/C6/C7 configs and run scripts exist, but no C5/C6/C7 GPU result
  has yet been validated. The requested P100 entry point uses low-memory
  mode; memory feasibility, especially for C7, remains an explicit runtime
  gate. These orders must not be described as established capability before
  the full audit gate passes.
- The schema, symmetry registry, and instance compiler can express higher
  orders such as C12 and C20. The local branch has removed the adapter and
  Foundry frame-recovery limit of 10 transforms, with C12/D6 CPU regressions;
  this removes the artificial input boundary but does not yet establish GPU
  inference support. Dense token-pair memory remains quadratic in assembly
  size, the checkpoint's relative-chain encoding saturates beyond nearby
  copies, high-order chain-ID paths are unvalidated, and the current
  seed-integrity audit has factorial pairing cost. C12/C20 must not yet be
  submitted as production native P100 diffusion jobs until the new CPU
  construction tests pass on LRZ and a bounded GPU probe is defined.
- Transform-aware output auditing currently assumes transform-major sorted
  chain IDs for the one-chain-ASU C3 baseline.
- Orbit-rigid mobility is an unvalidated, explicit opt-in experiment and is
  disabled in every formal Slurm entry point.
- Native C3 input construction and 10/50/100/200-step inference have
  succeeded. The 200-step candidate preserves the complete two-fragment
  interface, exact C3, continuity, and compactness, but retains one local
  linker/motif CA overlap copied threefold. A clash-free candidate is still
  required before claiming a scientifically final design, robustness, or
  generalization to Dn and multi-interface cases.

## 2026-07-30 scaffold-aware mobility pilot

- A separate, default-off experiment now closes the missing dynamic
  conditioning loop: a moved interface target also refreshes RFD3
  `motif_pos` and group target coordinates before the next denoising step.
- The pilot fails closed unless it uses one design, the low-memory/chunked
  pair path, exact orbit-average state, coupled noise, and fixed-motif
  preservation. Input mobility declarations and sampler opt-in must agree.
- The proposed scaffold-derived controller treats the complete cross-chain
  seed as one master SE(3) object, expands its copies through the declared Cn
  actions, and scores generated/fixed junctions, coarse CA clashes, excessive
  axis tilt, and displacement from the sampled pose.
- The first C5 configuration permits at most `1 A / 5 deg` cumulative motion.
  Proposal-only is the default; applying motion requires an explicit flag.
  Formal static C3/C5/C6/C7 entry points remain unchanged.
- This is local refinement, not a high-tilt rescue mechanism and not a
  retrained RFD3 model. Targeted LRZ tests, a full unit run, and paired
  static/mobile GPU validation are still required before treating the
  experiment as successful.
- The first real C5 proposal-only attempt, job `5722585`, passed all 247
  repository tests but stopped before its first denoising step because the
  boundary finder assumed ordinary peptide neighbours were present in
  `token_bonds`. Foundry runtime features do not require those polymer edges.
  The local fix now combines explicit token bonds with same-chain consecutive
  `residue_index` CA tokens. Regression tests cover an empty-token-bond C5
  contig and reject false adjacency across chains or residue gaps. This fix is
  syntax-checked locally and remains pending LRZ unit and real-feature
  validation; job `5722585` is not a mobility result.

The concise design and validation boundary is recorded in
`docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md`.

### 2026-08-04 orbit-owned SE(3) control migration

- Motif mobility is now represented on `SymmetryOrbitSpec` and lowered to a
  topology-neutral `ConstraintOrbitInstance`; it is no longer conceptually
  owned by an interface edge. Existing edge-level mobility remains a
  compatibility input and is migrated fail-closed into the owning orbit.
- The assembly IR now carries constraint orbits plus generated scaffold links
  and N/C terminal extensions. This is the common representation needed to
  compile an interface-spanning motif and a central motif without separate
  sampler algorithms.
- Orbit mobility metadata now survives the complete compiler-to-RFD3 runtime
  path: allowed subspace, proposal source, cumulative translation/rotation
  bounds, timestep window, response, per-step trust region and objective IDs
  are encoded as RFD3 features and parsed by `ConstraintOrbitLayout`.
- `OrbitRigidMotifController` now applies the schedule belonging to each orbit
  rather than silently using only global sampler defaults. Its diagnostics
  report the effective subspace, proposal, objectives and trust region for
  every orbit. Legacy inputs that do not explicitly declare an orbit schedule
  retain their sampler-level schedule, so existing pilot commands remain
  reproducible during the migration.
- Native runtime execution currently accepts `bounded_se3` with
  `denoiser_fit` or `scaffold_objectives`. `radial`, `radial_axial`,
  `tilt_only` and `hoyeung_drag_compat` are valid IR contracts but fail closed
  until a topology-defined reference frame and proposal backend are wired;
  they are not falsely treated as arbitrary Cartesian SE(3).
- Ho-Yeung's original per-step COM dragging remains the compatibility design
  reference. It is not the native algorithm: the formal controller moves one
  master rigid pose and regenerates every copy through declared group actions,
  preserving the exact motif and symmetry orbit.
- Local `py_compile` and whitespace validation pass. After synchronization,
  the complete LRZ `rc-foundry` unit suite passed **313/313 tests in 9.804
  seconds**. This validates the orbit-owned mobility schema/IR, feature
  transport, runtime parsing, per-orbit scheduling, legacy schedule fallback
  and all preceding exact-symmetry regressions at the unit-test level. It is
  not yet a GPU validation of dynamic motif movement.

### Extracted C5/C6/C7 batch screening

`python -m rfd3_mosaic.rfd3_batch_screen` now audits an extracted structure
directory and resolves each `<job-id>__*.cif` back to its sibling run. The
strict rank requires the seed report plus continuity, compactness, zero CA
clashes, and declared-transform symmetry. Ring-plane and coarse inter-chain CA
packing descriptors are reported only for ranking, not as calibrated
acceptance thresholds. A local diagnostic over 18 copied C6 structures found
13 with zero chain breaks and zero CA clashes, four with replicated clashes,
and one severely discontinuous result; seed provenance and declared-transform
strict status must be recomputed on LRZ where the original job directories are
available.
`scripts/rfd3_mosaic/screen_extracted_cn_structures.sh` runs the C5/C6/C7
screens together and prints one compact comparison of the hard-gate counts and
top-ranked candidates.

### Selected low-tilt P100 comparison

The first retained C5 mobility candidate is pose seed `3419`:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/lhd101_c5_mobile_lhs_v1/candidate_0419_seed_3419/manifest.json
```

`scripts/rfd3_mosaic/submit_lhd101_c5_mobile_pair_p100.sh` submits a controlled
comparison on only the P100 partitions:

```text
same candidate + same diffusion seed 42
-> proposal-only 50 steps
-> applied-mobility 50 steps
-> matching 200-step jobs, each held by afterok on its own 50-step result
```

The wrapper records every job and dependency in
`c5_mobile_seed3419_p100_v1.tsv` under the run base, rejects incompatible
resume files, and can resume safely after the Slurm QOS submission limit. Its
experiment fingerprint includes the manifest, config and pilot-script SHA256,
diffusion seed, mobility interval, target tilt, and linker length, so a changed
shell environment cannot silently mix conditions. These jobs are prepared but
are not recorded as executed or validated until their result audit files are
inspected.

### 2026-08-03 C5/C6/C7 low-tilt production campaign

- Added `submit_lhd101_cn_low_tilt_8_3.sh` for one strictly accepted
  0--15-degree interface-seed pose per C5/C6/C7 order and 100 independent
  200-step diffusion seeds per pose (300 logical jobs).
- The full pose ensembles, rather than the QD shortlists, are searched. The
  three selected manifests are frozen with SHA256 provenance under the `8.3`
  campaign root before submission.
- Submissions are balanced across orders and capped at 36 per invocation for
  the LRZ QOS limit. Resume is state-aware: active/completed/audited tasks are
  skipped, while infrastructure-incomplete failures can receive one retry.
- The shared Cn batch script now accepts an isolated campaign root and no
  longer treats a non-zero diagnostic `nvidia-smi` exit as a model failure;
  the following PyTorch CUDA probe remains fatal.
- This campaign is prepared locally but is not recorded as synchronized,
  submitted, or scientifically validated until the LRZ selection table, job
  accounting, and both result audits are inspected.

### 2026-08-03 multi-interface D3 engineering checkpoint

- Removed the adapter's one-copy-zero-link restriction. It now compiles one
  or more disjoint scaffold segments, emits a deterministic chain per segment,
  and preserves legacy single-link JSON fields unchanged.
- Runtime motif constraint groups are now assembled from every selected ASU
  link. Multiple configured interfaces therefore become separate constraint
  orbits under one native symmetry registry.
- Seed-integrity mapping is no longer restricted to exactly two indexed
  fragments. The audit CLI derives all unambiguous fragment pairs from
  `interfaces -> ports -> fragments`, evaluates them independently, and
  retains the old report schema for a single interface. Chain association now
  uses minimum-cost bipartite matching instead of factorial permutation
  enumeration, so D3 multi-chain output does not make the audit intractable.
- Added `lhd101_d3_two_orbit_engineering.yaml` as an explicit plumbing test:
  two duplicated LHD101 interface orbits, two disjoint ASU chains and D3
  expansion. This is not yet a connected protein cage or a biological design.
- Added `validate_lhd101_d3_two_orbit.sh` as the fail-closed LRZ CPU gate. It
  runs adapter and seed-audit regressions, compiles the exact D3 input, and
  invokes the real Foundry input builder/prevalidation. GPU inference must not
  be added until this gate passes in `rc-foundry`.
- Local syntax compilation, shell parsing and whitespace checks pass. The
  local system Python lacks `pydantic`, so behavioral tests remain pending on
  LRZ; no server execution is claimed here.
- LRZ adapter and seed-audit regressions subsequently passed (19 adapter tests
  and 5 seed-integrity tests), but the first real prevalidation exposed an
  upstream frame-recovery ambiguity: two duplicated, sequence-identical motif
  occurrences per ASU were grouped into 12 entity instances, while D3 has six
  unique transforms. Mosaic already owns and validates the exact transform
  registry, so multi-chain Mosaic inputs now explicitly request declared
  frames instead of re-inferring them from entity multiplicity. The default
  Foundry behavior remains unchanged for ordinary inputs. This correction is
  syntax-checked locally and awaits LRZ regression/prevalidation.
- A separate 200-step run reached the network on a 16 GB P100 but failed in
  token initialization with CUDA OOM (13.07 GiB allocated and a further
  2.93 GiB requested). P100 is therefore excluded from this two-orbit D3
  experiment; subsequent runtime probes should use H100 or 80 GB A100 only.

### 2026-08-04 central-motif bidirectional-growth diagnosis

The product target now explicitly contains two symmetric fixed-motif
topologies, both of which must be supported rather than conflated:

1. a cross-subunit interface seed at the ends of an ASU scaffold segment,
   with the intervening segment generated; and
2. one fixed motif in the middle of each protomer, with new N- and C-terminal
   regions generated around it.

The second topology is not represented as a fake left/right interface.
Runtime constraint metadata now accepts an explicit `fixed_motif` group with
one `motif` role while retaining the strict two-role schema for interface
groups. `rfd3_central_motif_probe` derives a controlled central-motif input
from an existing C3 adapter input, and
`lhd101_c3_central_motif_probe_p100.sbatch` provides four paired arms:

```text
A  true original RFD3 state, realignment enabled, no complete-orbit restore
B  exact orbit-average/coupled-noise Mosaic state with complete restore
C  same original RFD3 state as A, but realignment disabled
D  legacy state, realignment disabled, complete motif-orbit restore
```

This matrix separates configuration/realignment effects from symmetry
projector overwrite and exact-state effects. A central-orbit audit compares
the complete output motif orbit to the source registry using a single joint
rigid alignment and an invariant all-pairs distance-matrix residual. The
experiment is prepared locally but no LRZ result is claimed yet. Unlike the
larger D3 two-orbit case, this C3 single-orbit probe is intentionally small
enough to test on P100 first.

The implementation target is arm D (`exact_mosaic`), not attribution of an
older external run. Arms A--C remain optional regression controls and should
not consume routine GPU budget. Production acceptance requires the central
motif orbit audit plus scaffold continuity, clash and symmetry audits; passing
only the motif audit is not sufficient evidence of a usable design.

### 2026-08-04 unified user-facing experiment CLI

Routine use no longer requires writing a long Slurm script. A strict,
versioned experiment YAML selects either `interface_seed` or `central_motif`,
while a separate execution profile owns partitions, resources, environment
activation and checkpoint paths. The public `exact_mosaic` preset expands to
the validated correctness contract: no realignment, motif-precedence restore,
orbit-average state, coupled noise and required complete motif groups. These
invariants are not exposed as casual user toggles.

The new `rfd3-mosaic validate/render/submit` flow freezes resolved paths and
hash provenance, emits a short generated sbatch, delegates execution to one
Python worker and applies the topology-specific motif audit plus the common
scaffold audit gate. Built-in P100, H100 and A100-80G profiles and two example
experiment files are present. Local syntax/render checks are required before
handoff; LRZ environment tests and GPU submission remain pending until the
changes are synchronized. User instructions live in
`docs/rfd3_mosaic/USER_CLI.md`.

For routine interactive use, the still simpler `central` and `interface`
commands generate that versioned experiment YAML internally. A central-motif
user supplies only the validated input JSON, motif selector, optional N/C
lengths, execution profile and output root; exact-symmetry settings remain
software-owned. The explicit YAML commands remain the auditable advanced and
batch interface rather than a prerequisite for ordinary use.

## Verification commands

```bash
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:$PYTHONPATH"
python -m unittest discover -s tests/rfd3_mosaic/unit -p 'test_*.py' -v
python -c "from rfd3_mosaic.compile import load_interface_seed_config; load_interface_seed_config('configs/rfd3_mosaic/single_interface/lhd101_c3.yaml'); print('LHD101 config OK')"
git status --short --branch
```

### 2026-08-06 scaffold-driven relative seed mobility canary

The public fixed-component pose contract now distinguishes the source of a
bounded rigid-pose proposal:

```yaml
pose:
  mode: bounded_mobile
  proposal: denoiser_fit | scaffold_objectives
```

`denoiser_fit` remains the compatibility default.  The first public mobility
canary proved runtime transport, exact symmetry, conditioning refresh and
pose bounds, but its observed motion was effectively zero because the hard
fixed-coordinate conditioning caused the denoiser proposal to coincide with
the current target.  It is therefore a safety regression rather than a
positive mobility result.

`scaffold_objectives` exposes the existing scaffold-boundary proposal through
the public compiler.  The new
`lrz_public_relative_seed_mobility_v100_canary.yaml` keeps the right seed orbit
fixed as a gauge anchor and permits the complete left seed orbit to translate
and rotate within explicit bounds.  Each seed remains internally rigid and
all copies remain exact C3 images; only their relative rigid pose may change.
GPU job `5733341` completed on `gpu-001` in 6 minutes 56 seconds and provided
the first positive scaffold-driven relative-pose result.  The mobile left
component translated by 0.160231 A and rotated by 0.653719 degrees, while the
right component remained the fixed reference.  The mobile component's
maximum per-copy internal RMSD was 0.000012 A and the fixed component's joint
orbit RMSD was 0.000013 A.  The output retained exact C3 geometry
(maximum symmetry coordinate RMSD 0.000034 A), zero chain breaks and zero CA
clashes.  The constraint-orbit, component-mobility and scaffold-validity
audits all passed.

This is evidence that `scaffold_objectives` can produce a non-zero bounded
relative SE(3) update without corrupting either component or the assembly.
It is not yet evidence for the newly added `radial`/`radial_axial` execution
subspaces or for an adaptive pose planner: the canary deliberately used full
`bounded_se3`.  Those capabilities require their own unit and GPU gates before
promotion.

Job `5733341` must also not be confused with whole-interface-seed orbit
mobility.  It deliberately represented the two halves of an interface as two
independent components, moved one half and anchored the other.  The biological
target contains three C3-related complete interface seeds; each complete seed
must retain the joint geometry of both halves while the pose of the complete
seed orbit changes relative to the symmetry axis.  The corrected
`lrz_public_whole_interface_orbit_mobility_v100_canary.yaml` therefore assigns
both fragment selectors to one `complete_interface_seed` coupling group and
applies one bounded rigid pose to that joint component before rebuilding all
three C3 copies.  That separate canary is the acceptance gate for the intended
whole-seed behavior.

V100 job `5733680` passed that whole-seed gate.  The compiled design contained
one 579-heavy-atom component: both interface fragments were coupled into one
rigid seed and expanded into three C3-related copies.  The complete seed orbit
translated by 0.146656 A and rotated by 0.588174 degrees.  Maximum per-copy
internal RMSD was 0.000015 A; output C3 coordinate RMSD was 0.000033 A; chain
break and CA-clash counts were both zero.  The constraint-orbit,
component-mobility and scaffold-validity audits all passed.  Together with the
split-component canary and the full public compiler/audit path, this promotes
general bounded-SE(3) orbit mobility from `gpu_canary` to `engineering`.
At that stage, axis-aware radial/radial-axial mobility remained CPU-validated
until its separate GPU gates completed.

Those axis-aware gates have now completed.  V100 job `5733718` executed the
complete two-fragment seed orbit in `radial` mode: seven updates produced a
0.074646 A translation with exactly zero axial component and zero rotation.
V100 job `5733719` executed the same seed in `radial_axial` mode: seven updates
produced a 0.161358 A translation, including -0.157698 A along the declared C3
axis, with zero rotation.  For both jobs the constraint-orbit,
component-mobility and scaffold-validity audits passed.  These runs establish
GPU execution and subspace enforcement; their small displacement remains an
engineering signal rather than evidence of scientifically optimal pose
refinement.

The observed whole-seed displacement (0.146656 A and 0.588174 degrees) is an
engineering execution signal, not evidence of scientifically meaningful pose
optimization.  The current pilot objective contains junction, clash, tilt and
initial-pose prior terms.  This design already had zero clashes and continuous
junctions; its seven active update calls therefore had little reason to move.
The 3 A / 10 degree declaration is a safety ceiling, not a requested target.
Before claiming adaptive assembly refinement, Mosaic must add explicit
assembly-level feasibility objectives, objective/trajectory audits and a
controlled recovery challenge with a known unfavorable starting pose.  Do not
increase weights merely to manufacture a larger displacement.

After this single-mobile-orbit gate, development should proceed in this
order: expose axis-aware radial/axial/azimuth/tilt/twist subspaces; add an
adaptive pose planner that derives feasible intervals from connectivity,
closure and clash geometry rather than asking routine users to guess them;
record
per-objective energy and pose trajectories; aggregate proposals jointly for
multiple mobile orbits without update-order dependence; then validate Dn and
polyhedral cage groups.  Do not add more topology-specific sbatch scripts;
all new behavior must pass through the public design schema, assembly IR,
sampler lifecycle, provenance and semantic audits.

### 2026-08-06 simultaneous multi-interface orbit implementation

The successful whole-interface canary is still a **single mobile orbit**.  Its
two selectors form one rigid component and its three C3 copies are one group
orbit; it does not demonstrate several chemically or topologically distinct
interfaces adapting together.  Static Dn compilation is CPU-validated, while
native Dn GPU execution, finite T/O/I groups and finite-window helical
execution remain incomplete.

The next runtime slice therefore removes the scaffold-guidance single-orbit
restriction.  Every mobile constraint orbit now derives a proposal from the
same immutable timestep snapshot.  Proposals are materialized together,
evaluated against one assembly-level junction/clash objective plus one
tilt/prior term per orbit, and accepted or rejected atomically.  No orbit sees
a partially committed update from an earlier YAML declaration, so the result
is designed to be independent of declaration order.  The legacy
`update_from_scaffold` entry point remains a single-orbit wrapper; the sampler
uses the plural `update_orbits_from_scaffold` lifecycle.

The LRZ full suite passed **423 tests**, including snapshot-synchronous
two-orbit proposals, atomic joint acceptance, declaration-order independence,
sampler admission of more than one scaffold-driven orbit and compiler
lowering of two independently mobile components.  Simultaneous multi-orbit
control is therefore `cpu_validated`, not merely `schema_only`.  It still
requires a GPU canary with at least two disjoint interface orbits, both showing
nonzero bounded motion, exact per-orbit reconstruction, atomic diagnostics,
zero chain breaks and valid complete-assembly symmetry.  Only then should Dn
GPU closure be used as the next group-level gate.  T/O/I require a finite-group
registry and local-neighbourhood execution; H requires an explicit repeat
window and boundary semantics rather than pretending that an infinite screw
group is a finite point group.

The first two-orbit V100 execution, job `5733773`, is a **partial kernel
result**, not an end-to-end pass.  The runtime recognized both declared mobile
orbits, applied one joint update, refreshed conditioning, retained all
579/579 fixed heavy atoms and preserved exact C3 geometry (maximum symmetry
coordinate RMSD 0.000033 A).  Maximum per-copy internal RMSD was 0.000012 A
for the first component and 0.000014 A for the second.  Both components moved
by approximately 0.014153 A and rotated by 0.055953 degrees.  However, the
10-step output contained 177 chain breaks and 12 CA clashes, so the scaffold
gate failed.  Ten-step under-denoising is the leading explanation, but it is
not yet proven: a matched 50-step execution must distinguish an intentionally
short smoke artifact from a controller/compiler interaction.  Because the two
motions were also nearly identical, this canary proves multi-orbit execution
plumbing but does not yet prove useful differential adaptation of independent
interfaces.  Maturity therefore remains `cpu_validated`.

The mobility audit now requires explicit `atomic_joint_acceptance` trajectory
evidence whenever more than one scaffold-driven orbit is declared; merely
reporting two bounded component poses is no longer sufficient.  The
replacement 50-step V100 configuration is
`experiments/lrz_public_two_orbit_joint_mobility_v100_50step.yaml`.  Promotion
requires all three audits to pass, both orbits to show bounded non-zero motion,
atomic joint diagnostics, zero chain breaks and zero CA clashes.

The runtime observability layer now transports stable constraint-orbit and
coupling-component identifiers into the sampler.  Every scaffold-driven step
records per-orbit junction, clash, tilt and pose-prior values before and after
the proposal, their deltas, the proposed SE(3) increment, the assembly-level
joint energy change and whether that orbit was actually committed.  Joint
rejection records an all-orbit rollback.  The worker extracts these diagnostics
into strict JSON as `mobility_trajectory.json`, while the mobility audit rejects
missing identifiers, non-finite objective terms, incomplete orbit records or
non-atomic commit decisions.  A differential-response regression gives the
two mobile orbits different boundary environments and requires different pose
proposals; a separate forced joint-rejection regression requires both states
to remain unchanged.  These changes improve explainability and fail-closed
behavior but do not promote the capability until LRZ tests and the 50-step GPU
gate pass.

The replacement 50-step V100 execution, job `5733788`, **passed the complete
GPU canary gate** on 2026-08-06.  The worker finished with exit code 0 in
5 minutes 56 seconds.  Both independently declared components retained
complete heavy-atom coverage (579/579 total) and rigid per-copy geometry:
maximum internal RMSD was 0.000012 A for `fixed_component_001` and 0.000015 A
for `fixed_component_002`.  The two orbits responded differently to the same
diffusion trajectory: the first translated 0.160616 A and rotated 0.652521
degrees, while the second translated 0.074847 A and rotated 0.323851 degrees.
The mobility audit reported two declared and two runtime components,
`atomic_joint_runtime=true`, ten scheduled controller calls and seven active
window calls.  Ten controller calls are correct for a 50-step trajectory
because the current update interval is five diffusion steps.  The final
scaffold contained three continuous chains, zero chain breaks, zero CA clashes
and exact C3 closure (maximum symmetry coordinate RMSD 0.000032 A).  Constraint,
mobility and scaffold audits all passed.  Relative to the failed 10-step job
`5733773`, this result supports under-denoising as the leading explanation for
the earlier broken scaffold, although the changed diffusion seed prevents a
strict causal attribution.  It also demonstrates genuinely differential rather
than duplicated orbit motion.  `multi_orbit_joint_control` is therefore
promoted from `cpu_validated` to `gpu_canary`.

This run predates automatic extraction of the embedded sampler trajectory, so
the worker did not initially write a standalone `mobility_trajectory.json`.
The artifact was backfilled successfully from the immutable result JSON after
completion using the current strict-JSON exporter; new submissions using the
current worker write it automatically.  The next maturity gate is not another
identical single run: require multiple diffusion seeds (including at least one
longer trajectory), consistent atomic trajectory records, and then a native
Dn two-orbit GPU closure before promotion to `engineering`.

## 2026-08-06 constraint-runtime lifecycle extraction

The first sampler productization slice now introduces
`rfd3.inference.symmetry.constraint_runtime.MosaicConstraintRuntime`.  Exact
Mosaic sampling uses one owned target and one `UnifiedJointProjector` across
initialization, every denoiser prediction, every Euler-updated state, optional
post-guidance repair and finalization.  Optional mobility enters through one
scheduled proposal transaction: scaffold-boundary proposals see a restored
immutable snapshot, accepted targets refresh conditioning before projection,
and rejected targets cannot change the advancing state.  The RFD3 network
invocation, noise schedule, coupled noise and Euler formula are unchanged.
Legacy symmetry mode retains its existing compatibility path.

The runtime reports phase counts and conditioning refreshes in result metadata.
New focused tests cover phase order, interval scheduling, transactional target
commit, rejected-proposal rollback and non-finite target rejection.  Existing
exact-sampler and scaffold-mobility integration tests now require lifecycle
diagnostics.  Local static compilation and `git diff --check` pass; the local
Python environment lacks PyTorch, so LRZ must run the focused tests and the
complete suite before this lifecycle becomes the accepted default.  After CPU
validation, replay one static exact C3 golden and one mobile multi-orbit C3
canary before beginning native D3 GPU work.

## Resume point

Sync the lifecycle module, sampler integration and focused tests to LRZ.  Run
`test_constraint_runtime.py`, `test_symmetry_motif_finalization.py` and then the
complete unit suite.  If they pass, replay one static exact C3 golden and one
50-step two-orbit C3 mobile canary using the new runtime diagnostics.  Do not
begin D3 GPU work or add another constraint type until those two equivalence
gates pass.  After equivalence, move controller construction behind a runtime
factory, then use that same boundary for native D3 two-orbit execution.

## 2026-08-06 product run shell

The first user-facing operations layer is now implemented locally rather than
as another campaign-specific script. The canonical execution verb is `run`;
`submit` remains an alias. `status` accepts a run directory, submission receipt
or numeric Slurm JobID and aggregates Slurm, worker, audit and artifact state
into one versioned JSON contract. `report` emits a self-contained HTML summary
and the same canonical JSON beside it. Numeric JobID discovery uses `--root` or
`RFD3_MOSAIC_RUN_ROOT`, eliminating the repeated hand-written `find`, `sacct`
and JSON snippets used during development.

The product layer deliberately distinguishes execution from scientific
acceptance. A queued/running job has `passed=null`; a worker or scheduler
failure has `passed=false`; and `passed=true` requires a completed worker
summary plus every declared audit. A scheduler-only `COMPLETED` record without
the worker summary is fail-closed as `completed_without_worker_summary`.
Focused tests cover completed and failed runs, missing/pending run directories,
numeric JobID discovery, artifact collection and escaped portable HTML. Local
static compilation and `git diff --check` pass. The local interpreter still
lacks `pydantic`, so the new reporting tests and full suite must run in the LRZ
`rc-foundry` environment before this slice is accepted.

The next product slice after LRZ validation is a persistent campaign index and
executor abstraction. That will make JobID lookup constant-time, allow arrays
and resume, and remove direct `sbatch` knowledge from the CLI. It should reuse
this status/report contract rather than introduce another output format.

That second slice is now implemented locally. `SlurmExecutor` owns parsable
submission and validated JobID extraction; the CLI no longer contains direct
`subprocess`/`sbatch` logic. Every new submission creates one atomic record at
`OUTPUT_ROOT/.rfd3-mosaic/jobs/JOB_ID.json`, and the immutable allocated worker
updates the same record to running, completed or failed. Worker-side indexing
is deliberately best-effort so an operational index write cannot invalidate a
scientifically successful structure and its audit reports. The `runs` command
lists indexed jobs, while numeric `status` uses the index before falling back
to recursive discovery for historical runs. Tests cover parsable Slurm output,
unknown-executor rejection, atomic index lifecycle, legacy worker upsert and
constant-time status resolution. This executor/index slice must accompany the
reporting files in the next LRZ validation batch.

Historical results can be migrated explicitly with `runs --rebuild`. The
rebuild scans worker summaries, imports only numeric run identities, preserves
existing executor/submission fields, reconstructs historical ordering from
summary modification time and reports every malformed summary. It is
idempotent and leaves the compatibility discovery fallback in place.

## 2026-08-07 public assembly-graph compiler

The next product-core slice replaces the implicit two-seed mental model with
a public assembly graph. `UserDesignSpec` now accepts any number of named
`components`, component-owned interface `ports`, geometric `interfaces`
between ports and directed generated-chain `connections`. Single-fragment
components use `geometry: rigid`; several
selectors that must preserve one common relative pose must state
`geometry: joint_rigid`. Compact connection endpoints use `component.C` and
`component.N`, while multi-selector endpoints fail closed until the exact
selector is named.

The compiler does not create a parallel cage pipeline. Each component becomes
one fixed-geometry constraint component and motion group; each named public
port selects one or more complete fragments on its owning component; each
interface binds two reusable ports through an explicit copy relation; and each
connection becomes a generated segment. The previous component-to-component
interface spelling remains a compatibility shorthand that generates private
ports per edge. All components are
expanded by the same declared finite symmetry registry and enter the existing
joint projector, mobility controller and RFD3 adapter. Component rigidity is
covered by the existing constraint-orbit audit. Interface relations are now
also compiled into a frozen, topology-neutral runtime audit plan containing
the concrete symmetry copy pairing, canonical RFD3 residue selectors and
declared target geometry for every edge instance. After inference,
`rfd3_interface_relation_audit` reconstructs those exact atom correspondences
from `diffused_index_map`: `preserve_input` is checked by independent rigid
fits of both ports followed by relative translation/rotation comparison, while
`contact` is checked by output COM distance and declared heavy-atom contact
count. Required failed edges enter the same fail-closed audit gate as motif,
mobility and scaffold validity and are reported as
`assembly_interface_relation_audit.json`. Initial poses remain independently
keyed by the public component identifiers and are mapped to internal
motion-group identifiers during lowering.

An explicit AssemblySpecification `constraint_group_strategy` now separates
node rigidity from edge semantics. Public graphs use `motion_groups`: fixed
runtime constraint orbits are generated from component nodes even when
interface edges exist. Legacy interface-seed specifications retain `auto`,
which preserves their historical interface-edge constraint groups. This is
necessary because a contact relation must not silently weld two independently
mobile nodes into one exact rigid seed. The first graph backend also requires
every component selector to be named by a generated connection endpoint;
unattached protein/ligand nodes await explicit ownership and multichain
semantics rather than being guessed.

The first relation vocabulary is deliberately small and executable:
`preserve_input` retains a reference relative transform, while `contact`
expresses a COM-distance interval, a heavy-atom contact minimum, or both.
Schema validation rejects unknown nodes, duplicate edge IDs, identity
self-interfaces, port/component selector ownership conflicts, ambiguous
multi-selector connection endpoints, invalid N/C direction and mixing the
graph frontend with legacy parallel fields. A port may bind to the same named
port on a non-identity symmetry copy, which is required for homotypic cage
faces. Static preflight still expands the complete assembly and rejects severe
clashes before a scheduler slot is consumed.

This is not yet multi-stabilizer cage semantics. Phase 1 uses one global
symmetry action because the current interface/link expansion requires both
nodes to have compatible copy indices. Vertex-, edge- and face-orbit
stabilizers will require explicit coset/orbit identities in the assembly IR;
they are not represented by fake extra scripts. The capability ledger now
marks the static `public_assembly_graph` path as `gpu_canary`: T job `5735772`
completed the graph-authored 50-step V100 gate with relation, constraint and
scaffold audits passing. This does not yet promote the new output-stage
interface-design controller or multi-stabilizer cage semantics.

The first T multi-face graph preflight exposed an important reference-frame
bug rather than an invalid user interface edge.  Static interface validation
was comparing a cross-copy relation observed after component initialization
and symmetry expansion against the two untransformed source-file port frames.
That comparison is valid only for ports on the same copy.  `preserve_input`
now freezes the relation in the fully initialized, symmetry-expanded compiled
assembly, which is also the presymmetrized structure used by the
post-diffusion relation audit.  Explicit `target_transform` relations remain
independent declared targets and retain their original strict validation.
Regression coverage addresses a nonidentity named C3 neighbour under a
sampled master pose; the T graph configuration uses the canonical registry ID
`T:g01`.

## 2026-08-07 tetrahedral two-orbit GPU canary

LRZ job `5734641` completed the first valid native tetrahedral execution with
two independently declared fixed motif components. The runtime reconstructed
both components through all 12 proper T rotations, retained 2316/2316 expected
heavy atoms and passed both the constraint-orbit and scaffold-validity audits.
The maximum per-copy internal RMSD was 0.000008 A, the maximum joint-orbit
error was 0.000150 A and the maximum complete-assembly symmetry-coordinate
RMSD was 0.000106 A. The 12-chain output had zero chain breaks and zero CA
clashes.

Independent repeats `5734679` and `5734680` subsequently completed on
different V100 nodes and passed prevalidation, exact constraint-orbit and
scaffold-validity audits. Static T two-orbit execution therefore has three
successful GPU realizations; this strengthens engineering reproducibility but
does not replace the missing dynamic-T and O/I validation gates.

This closes the **static T, two-independent-orbit GPU canary**. It does not
validate tetrahedral dynamic mobility, O or I GPU execution, mixed stabilizers
or a general cage-design workflow. The aggregate `polyhedral_groups`
capability therefore remains `cpu_validated` until O and I receive their own
runtime gates; the narrower static-T execution state is now `gpu_canary`.

The manual PyMOL verification for this run required an explicit 24-fragment to
12-output-chain atom correspondence and produced 0.000624 A RMSD over all 2316
fixed heavy atoms. That procedure is now encoded in
`scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py`. The
`mosaic_align_fixed` PyMOL command reads the run's compiled residue map and
symmetry registry, creates a separately aligned output object and leaves the
original visualization state untouched. Users no longer need to reproduce the
T-chain mapping by hand or rely on PyMOL sequence heuristics. The higher-level
`mosaic_load_run` command accepts only a run directory, discovers and loads the
reference and result structures, preserves the raw result, performs the same
alignment and applies a consistent comparison style. For already loaded
structures, the zero-argument `mosaic_align` command detects the reference and
result objects, locates their local run metadata and applies the same audited
mapping. It fails closed when either inference is ambiguous.

## 2026-08-07 graph-interface packing guidance v2

The unified output-stage interface field now distinguishes broad interface
formation from a degenerate point contact. In addition to nearest-pair
attraction, every required edge derives a conservative per-side residue
coverage target from its requested contact count and penalizes missing
coverage independently on both chains. Clash energy is normalized by the
number of participating residues rather than all pairwise distances, so one
severe overlap cannot disappear numerically as cage size grows.
Coverage is accompanied by a contiguous-patch term based on adjacent token
runs; spatially scattered residues therefore cannot satisfy a broad-interface
request merely by being individually close. Concrete symmetry copies are
averaged within each source interface before different declared interfaces
are averaged, so orbit multiplicity cannot silently determine objective
weight.

Guidance gradients are still applied only to generated tokens. Before the
bounded step, adjacent selected tokens on the same output chain are smoothed;
the operation never crosses a fixed-motif gap or a chain boundary. Exact
symmetry and fixed-orbit projection remain authoritative after guidance.
Runtime diagnostics and the guidance audit now record attraction, coverage,
continuity, clash, distance, per-side covered and contiguous residue counts,
per-edge energy and mean/maximum token steps.
This controller remains scientifically unvalidated until the designed-
interface GPU canary and multi-seed repeats pass their final generated-heavy-
atom relation audits.

## 2026-08-10 ordinary intent boundary and terminal packing revision

The ordinary-user layer is now a real input-analysis surface, but it is not
yet an automatic cage solver. `rfd3-mosaic inspect` detects pairwise
chain-contact patches in PDB/mmCIF input and writes a replayable
`simple_cage_intent`. The intent records ring/cage/auto architecture,
homomer/heteromer/auto composition, optional symmetry and size ranges, any
number of detected interface seeds, and physical use as `auto`, `exact` or a
range. Compatibility planning filters full-group Cn/Dn/T/O/I hypotheses and
reports unresolved ownership, scaffold order, neighbour relations and
continuous pose variables. It deliberately remains non-executable until a
resolver freezes those variables into the same expert `UserDesignSpec` and
`AssemblySpecification` path.

The executable expert graph accepts any number of named components and ports,
and every public interface may contain two or more participants; no three- or
four-face limit exists. Multi-participant supplied relations lower to a
contact-supported binary execution tree without changing their atomic public
identity or physical multiplicity. Native variadic sampler tensors,
polymer-unit ownership inferred across ambiguous many-seed inputs,
stabilizer/coset orbits and mixed physical multiplicities remain the next
architecture-resolver work rather than hidden special cases.

Job `5741076` proved that packing diagnostics v4 could form transient contacts
but did not retain a broad final interface: all three C3 edges finished with
adequate heavy-atom pair counts but only 7/6 contacted residues and 2/4
contiguous residues versus required 9/9 and 6/6. The root cause was a proxy and
lifecycle mismatch, not a reversed gradient: the old sinusoidal field was
fully disabled for the last 20% of diffusion, and orientation/shape selected
scattered nearest residues rather than the same continuous patch audited at
the output.

Packing diagnostics v5 therefore makes one coherent runtime contract:

- guidance retains a bounded terminal weight instead of switching off;
- a deterministic final-polish phase runs through the same exact joint
  projector before finalization;
- coverage/continuity failure activates a configurable fraction of the token
  trust region instead of accepting vanishing gradients;
- continuity selects a genuine token-adjacent window with soft occupancy;
- orientation and contact-depth shape reuse that same contiguous patch;
- the shape term contains a shallow packing-distance well rather than only a
  variance term;
- default backbone regularization is reduced and CA clash pressure increased;
- automatic continuity is capped by the longest physically available
  generated run in both runtime and final heavy-atom audit, while explicit
  user targets remain strict;
- schema-v5 diagnostics distinguish controller execution from
  `final_proxy_targets_satisfied` and record final coverage/continuity plus
  the number of terminal polish steps.

This slice is locally syntax-checked and has focused regression coverage for
late-time hold, disconnected short runs, trust-region activation, final proxy
evidence and explicit-versus-automatic audit targets. It requires the complete
LRZ unit suite and a newly rendered 50-step V100/P100 canary before the packing
capability can be promoted.

### 2026-08-10 first executable ordinary resolver

The missing CLI bridge is now implemented for the smallest architecture that
can be frozen without inventing topology: one binary `preserve_exact`
supplied-interface seed in a full-orbit Cn ring. `rfd3-mosaic resolve`
enumerates both chain directions and adjacent-copy directions, preserves the
input interface as one joint-rigid component, creates the cross-copy polymer
connection, and sends every hypothesis through the common static compiler,
ranking and strict YAML replay/hash gate. It emits ordinary expert-compatible
`UserDesignSpec` files under `selected/`; it does not introduce an ordinary
sampler or automatically run rank 1.

The boundary remains explicit. Three-or-more participants, unknown relative
seed poses, heteromer ownership, homomer-equivalence claims, Dn/T/O/I
connection transforms, diameter/cavity objectives and stabilizer/coset
multiplicities remain rejected.

The generic deterministic directed polymer path-cover primitive enumerates
rotation/reversal-unique interleaved cycles for disjoint binary seeds and
proves that every seed side is used exactly once. Its hypotheses remain
`executable: false`: that object contains no input-contact, backbone-anchor,
symmetry-winding, linker, clash or replay evidence.

A separate experimental bridge,
`prepositioned_multi_binary_cn_v1`, now makes only one subset executable. It
requires several disjoint binary preserve-exact seeds already co-positioned
in one input frame, complete boundary `N/CA/C` anchors, `composition: auto`
and a full-orbit Cn ring. It enumerates path cover, chemical direction,
closing seam and winding; lowers each candidate to the normal public graph;
validates its expanded interface/unit topology; and passes it to the common
static compiler/ranker and strict replay. It does not optimize seed pose or
automatically select rank 1.

This bridge remains `schema_only`. Calling it 70% engineering-complete
requires, from one frozen snapshot: the complete LRZ suite; a real two-seed
`inspect -> plan -> resolve` with deterministic manifest and zero advertised-
candidate replay failures; one selected YAML passing public/runtime
prevalidation plus linker/clash/group-closure checks; a newly rendered
50-step V100/P100 run passing all fixed-seed, symmetry, continuity and
scaffold audits; and a second input or Cn order without source-specific code.
Even then, general pose discovery, hyperedges, component equivalence,
stabilizer/coset and T/O/I remain separate gates.

The replay gate now includes the native RFD3 adapter for this multi-seed
frontend, not only standalone structure hashing. Cross-copy scaffold seams
retain their physical target copy in the contig. Fixed constraint groups bind
the selectors that actually enter that contig and convert compiler physical
transforms into native RFD3 actions relative to the materialized ASU. For
example, a C3 seam containing `A@0` and `F@2` reconstructs the original three
supplied interface copies as `A@0+F@1`, `A@1+F@2` and `A@2+F@0`; it must not
rewrite `F` to an absent copy-zero `B` selector or lock `A@0+F@2` as the
functional seed. The conversion uses group composition rather than cyclic
index arithmetic, so the contract remains valid for non-commutative finite
registries.

A real 7mwr A/B contact was split into two disjoint contact patches for an
engineering regression: all 16 C3 path/direction/seam/winding candidates
compiled without hard clashes, two selected candidates passed strict YAML
replay, expanded alternating-topology replay and native-adapter replay with
zero replay failures. The first LRZ run then correctly exposed the missing
runtime-token coverage in the old copy-zero encoding even though 616 unit
tests had passed. The adapter correction and a full
`prevalidate_rfd3_input` regression are now implemented locally; the complete
LRZ suite and selected-YAML validation must be rerun from the corrected
snapshot. This is engineering evidence, not yet a GPU pass and not a claim
that the two patches are independent biological interface types.

## 2026-08-10 three-day demo evidence snapshot

Product progress is now reported with explicit evidence gates rather than a
single completion percentage. The two ordinary-user tasks are distinct but
lower through the same runtime: `preserve_supplied_geometry` restores an
interface already present in the input, whereas `create_symmetric_interface`
keeps a motif exact and asks generated regions to create a new neighbour
interface. Exact motif recovery is a necessary gate for both and a sufficient
packing result for neither.

Run `5741271` used
`examples/rfd3_mosaic/inputs/Prism_C3_G2_fixed_motif.pdb`, fixed `A12-20`
and two 35-residue terminal generations. It passed complete fixed-orbit,
symmetry, continuity, CA-clash and final heavy-atom interface-relation audits
(270/270 fixed heavy atoms; joint RMSD 0.0000185 A), but its runtime
`final_proxy_targets_satisfied` remained false and retrospective morphology
measurement found about 17.9 A central radial clearance. It is therefore the
partial generated-interface baseline, not the 7mwr supplied-seed run and not
a compact-pore success.

Run `5741324` used a strictly replayed selected candidate derived from
`examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb`. Its two
engineering seeds were disjoint patches of the same A/B contact:
`A186-189/B238-240` and `A191-192/B234-235`. RFD3 inference completed. The
generalized multi-chain audit now recovers 273/273 fixed heavy atoms and all
six supplied interface instances through compiler-declared cross-seam
provenance. The run nevertheless fails the independent scaffold gate because
six real CA clashes repeat under C3 (minimum 0.896 A). Its status is therefore
**exact-seed runtime proven, complete design failed**, not a passing cage.

The immediate demo gates are: correct the two-seed endpoint/linker clash, use
the new morphology report, repeat both ordinary tasks from frozen P100/V100
submissions, and retain only results whose task-specific audit set passes.
O/I/H, unknown-pose general cage solving and downstream
sequence/refolding remain outside the three-day claim.

### Fixed arrangement and generated guidance are orthogonal (2026-08-11)

`task: create_symmetric_interface` no longer implicitly authorizes motif
motion. Its new safe default is `fixed_arrangement: locked`: the complete
compiled fixed arrangement remains authoritative while graph guidance acts on
generated atoms. `fixed_arrangement: optimize_components` explicitly enables
the existing bounded SE(3) component controller. Exact geometry inside every
coupling group remains hard in both modes. This separates Hubert-style fixed
C4/C3 layouts with newly generated packing from multi-seed designs whose
rigid interfaces may change their mutual radius, angle and distance.

### Packing-aware mobile-orbit implementation (2026-08-11)

The core sampler now has one lifecycle transaction for generated-interface
patch motion and bounded SE(3) motion of all declared mobile motif orbits. It
replaces the previous split ordering (motif motion before Euler, graph packing
after Euler). A proposal may return a new fixed target and matching scaffold
coordinates together; the constraint runtime validates, projects and commits
them atomically. Rejection, proposal-only execution and proposal exceptions
restore both motif pose and patch-selection state.

Evidence level: **CPU implementation and integration tests passed**. The
packing, constraint-runtime and motif-mobility focused tests and the complete
794-test suite pass in the temporary Python 3.12 development environment. The
remaining promotion evidence is one frozen V100/P100 50-step canary with
non-zero bounded rotation, lower final packing energy, exact motif and
symmetry recovery, continuous chains and no new CA clashes.

### Adaptive multi-interface packing and output evidence (2026-08-17)

The generated-interface controller now selects capture, expansion and polish
from observed reciprocal-patch quality instead of advancing solely by
diffusion time. Joint proposals must lower total packing energy while
preserving both the worst source-interface objective and a bounded per-source
regression. The same contract is used by packing-coupled motif mobility.

An immutable capacity preflight rejects explicit coverage/contact requests
that exceed generated token capacity and rejects overlapping physical
interfaces that cannot receive exclusive residue patches. Geometry that can
still change under diffusion is not incorrectly rejected.

Final interface audit now measures reciprocal heavy-atom residue pairs,
contact density, depth uniformity, contact islands, burial, local void and
hydrophobic-composition proxies on the written structure. Guidance audit
schema v8 requires capacity and adaptive-phase provenance. The complete CPU
suite passes 794 tests on the temporary development server. GPU packing
quality remains the independent promotion gate.
