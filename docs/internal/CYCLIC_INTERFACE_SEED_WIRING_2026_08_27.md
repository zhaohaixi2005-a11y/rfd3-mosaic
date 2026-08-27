# Cyclic interface-seed polymer wiring

## Scope

This note records the compiler rule used when a supplied two-sided interface
seed is expanded under cyclic symmetry and the generated scaffold must form
the protein chain between neighbouring interface instances.

It does not change supplied-interface geometry, fixed-atom restoration,
symmetry projection, mobility bounds, diffusion guidance or result screening.

## Two relations that must not be confused

For copy index `k`, let the supplied non-covalent interface be

```text
A(k) ... B(k)
```

where `...` denotes the preserved intermolecular contact. `A(k)` and `B(k)`
must not be connected by a generated peptide linker.

The designed protein chain instead uses one of the two adjacent-copy paths:

```text
L+(k) = B(k) -- generated scaffold -- A(k+1)
L-(k) = B(k) -- generated scaffold -- A(k-1)
```

Indices are taken modulo the cyclic order. Offset zero is excluded by
construction.

## Pose-dependent selection

A global SO(3) sample changes which adjacent interface half is geometrically
nearer. A universal hard-coded `+1` therefore produces the right chain path
for some poses and a long, visually crossing path for others.

For each independently instantiated design, Mosaic evaluates

```text
d(s) = || COM_CA(B(0)) - COM_CA(A(s)) ||_2,  s in {+1, -1}
s*   = argmin_s d(s)
```

The C-terminal-to-N-terminal endpoint distance is recorded and used only as a
deterministic tie-breaker. The primary fragment-COM rule follows the published
RFdiffusion Interface-Seed implementation, which chose between the second and
last cyclic neighbours after sampling the motif pose.

After choosing `s*`, the public value `nearest_adjacent` is lowered to an
ordinary explicit `orbit_offset: +1` or `orbit_offset: -1`. Downstream Assembly
IR expansion, RFD3 input construction, frozen replay and audits therefore do
not repeat or reinterpret the choice.

## Evidence for the former mixed outcomes

The maintained LHD101 template formerly hard-coded `orbit_offset: 1`, despite
an earlier development note already stating that the direction was
geometry-dependent. Direct compilation of the same template shows that pose
seed `10001` selects `+1`, while pose seed `10002` selects `-1`. This explains
why one campaign could contain both plausible and visibly cross-connected
chains without any change in diffusion code.

## Provenance and invariants

Each compiled design records:

- the policy (`nearest_adjacent`);
- both evaluated offsets;
- fragment CA-COM and endpoint distances;
- the selected explicit offset.

Regression tests require:

- different poses can select opposite directions;
- offset zero is never evaluated;
- every generated link connects different physical copies;
- an explicit numeric offset remains unchanged;
- non-cyclic symmetry fails closed instead of inventing a cyclic neighbour.

The supplied A/B fragments remain in one joint-rigid component, so their
internal coordinates and non-covalent interface geometry are unaffected.
