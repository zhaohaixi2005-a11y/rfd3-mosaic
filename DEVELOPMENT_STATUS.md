# RFD3-Mosaic development status

Last updated: 2026-08-24

RFD3-Mosaic is an actively developed research preview. This page describes
the public capability boundary; it does not depend on a particular validation
server or deployment site.

## Release-ready foundation

- installable `rfd3-mosaic` Python distribution;
- packaged public examples and execution profiles;
- guided `init`, portable `examples` and discoverable `profiles` onboarding;
- `doctor`, `capabilities`, `plan`, `validate`, `resolve`, `render`, `run`,
  `status`, `report` and `audit` lifecycle commands;
- direct synchronous and Slurm execution adapters;
- immutable execution configuration and software provenance;
- strict result verdicts based on required scientific audits;
- automated CPU regression and wheel-installation checks.

## Supported scientific target

The current release target is:

- Cn and Dn exact symmetry;
- fixed central motifs;
- complete supplied interface seeds;
- several rigid motif/component orbits;
- generated-interface packing guidance;
- locked, radial, radial/axial and bounded rotational component motion;
- ordered generated polymer connections;
- motif, symmetry, interface, mobility and scaffold audits.

The regular-diffusion initializer is symmetry-local for Mosaic-compiled
generated regions: each generated path starts from fixed anchors in its own
physical copy instead of placing every copy at one shared global origin. This
contract applies uniformly to Cn, Dn, T, O and I and leaves native RFD3 inputs
on their historical default.

Supported means that the path has an executable compiler/runtime contract and
fails closed when the contract cannot be satisfied. Scientific usefulness of
generated candidates still requires input-specific evaluation.

## Research capabilities

The following paths have meaningful implementation and CPU coverage, but are
not yet stable public guarantees:

- T/O/I finite-group compilation;
- component stabilizers, cosets and quotient orbits;
- multiple supplied interface identities with independently solved poses;
- multi-participant interface relations;
- mixed component multiplicities;
- automatic finite-group topology and pose candidate enumeration.

These features are available for controlled research evaluation. Their
remaining work is dominated by broader executable replay, GPU evidence and
schema stabilization rather than by server access requirements.

## Active gaps

1. Demonstrate repeatable generated-interface packing quality across diverse
   inputs and diffusion seeds.
2. Generalize interface-edge stabilizers and mixed physical multiplicities.
3. Remove remaining special cases from native polymer-path lowering.
4. Expand ordinary-user error messages and maintained examples.
5. Define schema migration and stable-version compatibility policy.
6. Separate historical engineering evidence from maintained public manuals.

Sequence design and refolding are intentionally deferred and are not part of
the current release target.

## Validation policy

CPU tests establish schema, compiler, geometry, replay, feature construction
and audit behavior. GPU tests establish sampler execution, resource behavior
and scientific output evidence. Neither is a substitute for the other.

Hardware names in validation records describe tested environments only. They
do not restrict where RFD3-Mosaic may run.

Detailed historical notes are retained under `docs/internal/` for provenance.
