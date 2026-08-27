# Generated-interface packing guidance

## Scope

Mosaic supports two different scientific inputs on one compiler and sampler:

- `preserve_supplied_geometry`: every supplied interface seed is an atomic,
  joint-rigid relation. Mosaic scaffolds between user-declared participants
  and does not redesign that interface.
- `create_symmetric_interface`: the input is a motif without the requested
  neighbour interface. Generated residues receive packing guidance while the
  motif arrangement is either locked or explicitly allowed to move.

`fixed_arrangement` controls only relative motif pose. Exact geometry inside
each fixed coupling group is a hard invariant in every mode:

- `locked`: generated patches move; the supplied complete motif arrangement
  does not translate or rotate.
- `optimize_components` with `component_motion: guided`: generated patches
  and complete motif orbits use one atomic `radial_axial_rotation` packing
  transaction.
- `component_motion: free`: the same transaction uses bounded full SE(3).

## Runtime packing contract

Generated-interface guidance is a two-resolution objective. The first term is
the broad coordination-number contact prior used by RFdiffusion's public
`olig_contacts` potential:

```text
x_ij = (d_ij - d0) / r0
C_ij = 1 / (1 + x_ij^6)
E_broad = - w_inter * s(t) * sum(C_ij) / min(N_left, N_right)
s(t) = guide_scale * (1 - progress)^decay_power
```

It sees every generated CA pair belonging to a compiler-declared physical
interface edge. Its purpose is early capture: it supplies a smooth gradient
when no narrow contiguous patch has yet been selected. Mosaic minimizes
energy, hence the negative sign. The default switch (`r0=8 A`, `d0=2 A`),
guide scale (`2`) and quadratic decay follow the public RFdiffusion convention.

The second term is Mosaic's selective interface refinement. Each physical
interface is one reciprocal pair of sequence-contiguous generated patches.
Patch residues move through a blended local SE(3) translation and rotation,
not independent CA attraction. All symmetry copies are reconstructed through
the declared group action.

The joint objective includes:

- contact attraction and requested residue coverage;
- sequence continuity on both sides;
- patch tangent/interface-normal orientation;
- nearest-contact depth uniformity (shape proxy);
- local backbone and generated/fixed junction geometry;
- global CA exclusion and target-edge clash rejection;
- patch exclusivity across different physical interfaces;
- equal weighting by declared interface identity, independent of orbit size;
- a worst-interface **proposal-acceptance** rule that prevents one controller
  update from improving one interface by making another materially worse.
  This governs an optimization transaction; it is not an output-quality
  verdict.

The schedule is state adaptive:

1. the RF-style all-pair prior is strongest early and decays toward zero;
2. `capture`: a distant reciprocal patch receives a broad attraction basin;
3. `expand`: a captured but narrow/scattered patch emphasizes coverage and
   continuity;
4. `polish`: only a sufficiently broad contiguous patch receives smaller
   orientation/shape-focused SE(3) updates.

Diffusion timestep supplies an annealing envelope, but it cannot force an
unsatisfied patch into polish. Patch identity locks only after final-radius
coverage and continuity are met. Every proposal is symmetry projected, then
accepted atomically or rolled back.

The RF-style prior does not weaken fixed geometry. With a locked motif only
generated coordinates receive gradients. With guided component motion, the
existing atomic transaction may additionally update one complete joint-rigid
motif orbit, then regenerate all symmetry copies. A globally fixed motif is
restored by the hard projector after every denoising/guidance step.

Ordinary users select calibrated `packing: loose|balanced|tight` presets.
Expert `guidance.inter_chain_weight` changes only the broad inter-chain
contact prior; it no longer scales Mosaic's continuity, orientation, shape or
safety objectives. This preserves the familiar RFdiffusion intra/inter
control while keeping Mosaic's higher-resolution terms independently auditable.

## Pre-diffusion capacity checks

Before sampling, Mosaic proves immutable capacity facts:

- both sides contain enough generated residues for an explicit coverage
  request;
- explicit contiguous targets fit real sequence-contiguous token runs;
- requested contact count does not exceed the available residue-pair space;
- overlapping interface participant pools contain enough residues to assign
  exclusive patches.

Coordinate distance is not used as a hard preflight because the denoiser may
change it. Impossible token/sequence capacity fails closed; merely poor
initial geometry remains visible to the adaptive capture phase.

## Written-structure evidence

The authoritative `assembly_interface_relation_audit.json` evaluates heavy
atoms in the final CIF, excluding mapped fixed residues for generated
interfaces. In addition to clashes, raw atom contacts, residue coverage and
continuity, it records:

- reciprocal contacting residue-pair count and density;
- a smooth heavy-atom burial proxy;
- mean and standard deviation of residue contact depth;
- sequence contact-island counts;
- a local contact-void fraction proxy;
- hydrophobic contact and unpaired-hydrophobe fraction proxies.

Names ending in `proxy` are deliberately not reported as SASA, Rosetta shape
complementarity or energetic designability. Those require a sequence/folding
stage and are outside the current backbone-generation claim.

`graph_interface_guidance_audit.json` independently proves that
runtime edge identities match the compiler, immutable capacity preflight was
present, patch identity did not hop after locking, adaptive phases were
recorded, the scheduled broad contact prior was finite and attached to the
declared physical edges, and a finite post-finalization proxy was recorded.
Its top-level runtime result reports this executable contract only. Whether
the final coverage, continuity, orientation and shape controller references
were reached is recorded separately as advisory diagnostic evidence.

Runtime continuity and post-hoc continuity are intentionally distinct:

- runtime guidance optimizes reciprocal sequence-contiguous **CA windows** at
  its scheduled differentiable distance;
- the relation audit observes contacts between available backbone heavy atoms
  at the declared output cutoff.

The two measurements must be shown side by side. A campaign report must not
replace one with the other, take the best edge when every physical edge
matters, or combine them into a universal `accepted` verdict. The maintained
collector therefore reports the worst reciprocal side/edge for each measure,
keeps all generated coordinates, and labels unmet proxy targets for review.

The implementation is based on the published RFdiffusion contact-potential
form and schedule, not on an invented pass threshold. Mosaic's additional
coverage/continuity/orientation/shape scores remain advisory generation and
screening proxies until sequence/refolding evidence exists.

When motif pose and generated-patch motion are optimized together, the exact
SE(3) gradient, line-search and atomic transaction inequalities are defined in
the [rigid-mobility mathematical contract](RIGID_MOBILITY_MATHEMATICAL_CONTRACT.md).

Primary references:

- RFdiffusion paper: <https://doi.org/10.1038/s41586-023-06415-8>
- public `olig_contacts` implementation:
  <https://github.com/RosettaCommons/RFdiffusion/blob/main/rfdiffusion/potentials/potentials.py>
- public potential schedule/configuration:
  <https://github.com/RosettaCommons/RFdiffusion/blob/main/rfdiffusion/potentials/manager.py>

## Evidence boundary

CPU implementation and regression tests can close software semantics. Several
frozen 50-step GPU seeds are still required to characterize exact motif,
symmetry, chain continuity, clash burden and interface metric distributions.
Those observations support comparison and calibration; they are not a claim
that Mosaic can infer whether a user likes a backbone or whether it will work
experimentally. CPU success must not be described as stable
interface-generation quality.
