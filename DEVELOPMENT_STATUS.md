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
- result reporting that separates generated artifacts, declared software/
  geometry contracts and non-destructive advisory screening;
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
- topology and pose candidate enumeration inside a user-declared finite group
  and architecture.

These features are available for controlled research evaluation. Their
remaining work is dominated by broader executable replay, GPU evidence and
schema stabilization rather than by server access requirements.

## Active gaps

1. Characterize generated-interface controller behavior across diverse inputs
   and diffusion seeds without inventing a universal raw-backbone threshold.
2. Expand ordinary-user explanations and maintained examples.
3. Define schema migration and stable-version compatibility policy.

Automatic discovery of symmetry/order, connectivity, interface multiplicity
or a supposedly best cage architecture is outside the current product plan.
Unrestricted quotient/path generalization is likewise not a v0.1 completion
requirement. Existing implementations remain available for validated research
cases; they are not being deleted or rewritten.

Sequence design and refolding are intentionally deferred and are not part of
the current release target.

## Validation policy

CPU tests establish schema, compiler, geometry, replay, feature construction
and audit behavior. GPU tests establish sampler execution, resource behavior
and scientific output evidence. Neither is a substitute for the other.

Hardware names in validation records describe tested environments only. They
do not restrict where RFD3-Mosaic may run.

Detailed historical notes are retained under `docs/internal/` for provenance.
