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

Each physical interface is one reciprocal pair of sequence-contiguous
generated patches. Patch residues move through a blended local SE(3)
translation and rotation, not independent CA attraction. All symmetry copies
are reconstructed through the declared group action.

The joint objective includes:

- contact attraction and requested residue coverage;
- sequence continuity on both sides;
- patch tangent/interface-normal orientation;
- nearest-contact depth uniformity (shape proxy);
- local backbone and generated/fixed junction geometry;
- global CA exclusion and target-edge clash rejection;
- patch exclusivity across different physical interfaces;
- equal weighting by declared interface identity, independent of orbit size;
- a worst-interface acceptance contract that prevents one interface from
  paying for a materially worse one.

The schedule is state adaptive:

1. `capture`: a distant reciprocal patch receives a broad attraction basin;
2. `expand`: a captured but narrow/scattered patch emphasizes coverage and
   continuity;
3. `polish`: only a sufficiently broad contiguous patch receives smaller
   orientation/shape-focused SE(3) updates.

Diffusion timestep supplies an annealing envelope, but it cannot force an
unsatisfied patch into polish. Patch identity locks only after final-radius
coverage and continuity are met. Every proposal is symmetry projected, then
accepted atomically or rolled back.

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

`graph_interface_guidance_audit.json` schema v8 independently proves that
runtime edge identities match the compiler, immutable capacity preflight was
present, patch identity did not hop after locking, adaptive phases were
recorded, and the post-finalization proxy contract passed.

## Evidence boundary

CPU implementation and regression tests can close software semantics. A new
packing capability is scientifically promoted only after several frozen
50-step GPU seeds pass exact motif, symmetry, chain continuity, global clash,
runtime guidance and final heavy-atom interface audits. CPU success must not
be described as stable interface-generation quality.
