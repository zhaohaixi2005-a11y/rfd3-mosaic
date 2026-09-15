# Fixed-layout LHD101 pilot

This pilot isolates whole-seed motion in the two-anchor interface-seeded task.
It is the first fixed-layout baseline, not an implementation of a new spatial
routing algorithm and not a claim of zero crossings.

`lhd101_locked.yaml` fixes both fragments, A165-194 and B211-241, as the same
`supplied_interface_seed` rigid group. Initial placement may rotate/translate
the whole group; diffusion must not move it or distort its internal geometry.
The movable control is generated from the same configuration by changing only
the experiment name, `fixed_arrangement` to `optimize_components`, and
`preferences.component_motion` to `free`. This enables the existing bounded
SE(3) controller and its automatic capture; the comparison measures this
combined policy, not separate effects of rotation, translation and capture.

The paired pilot uses pose seed 10063 with diffusion seeds 15091500/15091501,
and pose seed 10039 with diffusion seeds 15091502/15091503. Each pose's locked
and movable arms run on the same server/environment. Two outputs share each
pose (`designs: 2`, `replicates_per_pose: 2`); each output has its own diffusion
seed. The worker's existing compiler feasibility gate remains enabled. A
rejected pose is recorded as a pre-generation failure, not silently replaced.

The generation path is B(copy i) -> 70-100 generated residues -> A(copy i +/- 1).
The current compiler chooses +/- 1 once from the initial fragment geometry.
This pilot does not change that COM-based choice into an endpoint-path search.
Full C3 context, exact symmetry, local-fixed-anchor initialization and current
core/routing guidance remain active. No artificial central exclusion volume or
new curved-path force is introduced.

The mobile control now uses the corrected capture implementation: in two-anchor
tasks, neighbours are bound from actual fixed polymer endpoints and cannot
switch according to instantaneous COM distances. This rule is shared across
symmetry groups and seed types. For the declared `preserve_supplied_geometry`
task, the compiler rejects an invalid supplied interface/polymer incidence
graph before emitting an inference input. Generic fragment-scaffolding inputs
retain the diagnostic because they need not describe interface-seeded assemblies. These
are common safeguards in both arms, not additional experimental variables.

Before interpreting an arm comparison, compare compiled contigs, resolved copy
relations, materialized lengths and initial coordinates. Equal random seeds
alone do not prove matched initial inputs across policy changes. Record source
commit, checkpoint hash, environment and every pre-generation failure.

For every result, inspect:

- Complete seed A+B geometry, including cross-fragment distances, and the
  constraint-orbit audit. Locked-arm coordinate errors must also remain within
  the existing numerical tolerances against the initialized target.
- Actual endpoint identities, full-chain continuity and peptide geometry.
- Cross-chain CA segment collision count and implicated residue pairs, CA
  clashes, and separate inspection for noncolliding interlacing. A small pore
  is not by itself a crossing. The existing segment threshold is not a proof
  of all-atom validity or of absence of global entanglement.
- Final continuity-projection diagnostics, monomer support and compactness.
  The existing final projection has no collision rollback; this remains a
  known limitation. Final crossings must fail the contract even if the
  projection or all other checks succeeded.

Keep all outputs. Report generation completion separately from seed preservation,
connectivity, crossing, and overall contract status. Eight outputs can expose
mechanisms; they cannot establish a reliable success-rate improvement.
