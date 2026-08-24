# Symmetry-local generated-coordinate initialization

Date: 2026-08-24

## Root cause

Regular RFD3 input parsing historically centers the fixed motif and then sets
every non-fixed atom coordinate to the single global origin. This is a valid
initialization for one compact conditioning frame. It is not valid for a
Mosaic assembly whose fixed motif has already been expanded into spatially
separated symmetry copies.

The failure was exposed by the I-symmetry continuity canary, job `5760519`.
Its first generated structure had 60 chains and exactly five backbone breaks
per chain. The first break began at the fixed-to-generated junction and was
46.63 A in CA distance. The fixed motifs were arranged on a large I orbit,
while every generated extension was initialized at the group origin. The
result was the observed radial-spoke structure; it was not a random-seed or
visualization artifact.

## Corrected contract

Mosaic-generated RFD3 specifications now explicitly declare:

```json
"generated_coordinate_initialization": "local_fixed_anchor"
```

For regular diffusion, detailed generated coordinates are still discarded.
The difference is the frame in which they are discarded:

- a terminal generated region starts at the nearest fixed residue anchor in
  its own chain;
- a generated region between two fixed anchors is initialized by linear
  interpolation between those anchors;
- a generated-only chain without a fixed anchor retains native RFD3's global
  origin initialization;
- fixed coordinates are never changed by this operation.

Because symmetry-related fixed anchors are related by the declared group
transform, their generated initial coordinates are related by the same
transform. Exact coupled noise and the existing symmetry projection remain in
force during diffusion.

## Compatibility boundary

The setting is emitted only by the Mosaic adapter. Native RFD3 inputs that do
not request it retain `global_origin`, preserving historical behavior.
Partial diffusion is unchanged because it already retains supplied
coordinates. Fully fixed designs contain no generated atoms and are therefore
unchanged.

The correction is group-agnostic:

- Cn and Dn are affected when their motif copies are appreciably displaced
  from the group origin;
- T and O receive the same correction for every physical copy;
- I is the most visibly affected in the current canary because it has 60
  copies at a large radius;
- severity scales with orbit radius and generated path geometry, not merely
  with group order.

## Validation boundary

CPU regression tests establish that native global-origin behavior is retained
by default, local terminal and interpolated anchors are correct, and local
initialization commutes with a proper group rotation. Adapter regression tests
establish that compiled Mosaic designs record the new provenance.

GPU validation must rerun the I continuity canary from a frozen revision. The
expected first-order signal is removal of the repeated large
fixed-to-generated radial junction. A generated coordinate file remains a
usable output even when downstream advisory screens flag it; continuity and
scientific quality must be reported separately.
