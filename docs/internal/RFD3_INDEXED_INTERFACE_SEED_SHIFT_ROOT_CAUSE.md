# Archived RFD3 indexed cross-subunit interface-seed shift analysis

> Internal debugging record. Not a public usage requirement.

## Source-level root-cause analysis

This document records why an RFD3 motif can be declared coordinate-fixed yet
still move in the native symmetry sampler, why an apparently preserved
monomer motif does not prove preservation of a cross-subunit interface seed,
and exactly where the coordinate overwrite occurs.

The source audit is pinned to the official RosettaCommons Foundry commit:

```text
3b739a2318398f71f5d446cef6bc8d3ca3bd1295
```

All GitHub links below point to that immutable commit rather than the moving
`production` branch.

## Executive conclusion

The earlier diagnosis is directionally correct, but the precise distinction
is not simply “monomer motif versus interface motif” or “same chain versus
different chains.” RFD3 does not branch on those biological categories.
Instead, the native symmetry path contains two different notions of
fixedness:

1. **Model-level coordinate fixedness**, represented by
   `is_motif_atom_with_fixed_coord` and populated from
   `select_fixed_atoms`.
2. **Projector-level fixedness**, represented by
   `sym_entity_id == FIXED_ENTITY_ID`.

These masks are not equivalent for an indexed motif. The diffusion model
honors the first mask: fixed atoms receive zero atomwise diffusion/churn noise
and the standard EDM output parameterization becomes an identity map at those
atoms. With realignment enabled they can still undergo a shared stochastic
global rigid augmentation, which does not alter their joint distance matrix.
The symmetry projector honors only the second mask. Indexed motifs remain
ordinary symmetry entities and their non-ASU copies are reconstructed from
the ASU after denoising.

The observed displacement is therefore best described as:

> A symmetry-specific constraint-representation gap between atom-level fixed
> coordinate conditioning and entity-level symmetry projection. An indexed
> cross-subunit interface seed is not represented as one joint protected
> constraint in the symmetry projector, so the post-denoising projector can
> overwrite target-copy fixed coordinates even though the denoiser itself
> preserved them.

This is not evidence that the checkpoint or denoising network is defective.
It is also not a failure to parse `select_fixed_atoms`. It is a precedence and
representation conflict in the symmetry execution path.

## Evidence classification

The statements in this document are separated into three evidence classes:

- **Direct source fact**: follows immediately from the pinned official code.
- **Mechanistic consequence**: follows mathematically from the documented
  assignment and update order.
- **Local experimental evidence**: comes from RFD3-Mosaic jobs and must not be
  presented as official Foundry data.

The source proves that the projector can overwrite an indexed fixed motif.
For a strict input-specific causal demonstration, coordinates should also be
captured immediately before and after the projector, as described near the
end of this document.

## The concrete RFD3-Mosaic input

The representative input has the following semantics:

```yaml
contig: B1-31,70-100,C1-30
select_fixed_atoms:
  B1-31: ALL
  C1-30: ALL
symmetry:
  id: C3
  is_symmetric_motif: true
```

The two selected fragments are fixed coordinate motifs, while the central
segment is generated. Native symmetry constructs one ASU contig and expands
it into three C3-related output chains. The complete preserved PPI is a
noncovalent relationship between motif halves belonging to different
symmetry-related output chains. Consequently, preserving each fragment's
internal structure is necessary but not sufficient: the cross-chain relative
rotation, translation, distance matrix, and contact network must also remain
unchanged.

## Important correction: the denoiser sees the expanded assembly

Native RFD3 symmetry inference is not simply:

```text
denoise one isolated ASU -> copy it only at the end
```

The actual high-level sequence is:

```text
build one ASU from the contig
-> expand it into the complete Cn/Dn atom array
-> pass the expanded noisy assembly to the denoiser
-> apply an ASU-to-copy symmetry projection after denoising
```

`DesignInputSpecification.build()` invokes symmetry expansion before origin
initialization and feature construction:

- [input build order, `input_parsing.py` L510-L521](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/input_parsing.py#L510-L521)
- [symmetry expansion call, `input_parsing.py` L743-L751](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/input_parsing.py#L743-L751)
- [copy construction and concatenation, `symmetry_utils.py` L132-L152](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L132-L152)

The complete `X_noisy_L` is then passed into `diffusion_module`. Thus the
network can, in principle, see inter-subunit geometry. The ASU-only operation
that matters here is the **post-denoising symmetry projector**, which chooses
the ASU as the coordinate source and overwrites the other copies.

## Four official concepts that must not be conflated

| Concept | Official role | Consequence in this case |
| --- | --- | --- |
| `select_fixed_atoms` | Selects atoms whose coordinates are fixed for motif conditioning | Produces `is_motif_atom_with_fixed_coord`; controls noise and atom-level diffusion time |
| indexed motif / `contig` | Places motif residues at explicit output sequence positions, potentially continuous with generated residues | Remains an ordinary symmetry entity and is symmetrized at runtime |
| unindexed motif / `unindex` | Uses a motif whose final sequence position is not indexed | The non-indexed motif can receive `FIXED_ENTITY_ID` and be skipped by the projector |
| `is_symmetric_motif: true` | States that the source motif is pre-symmetrized and should be used to infer symmetry frames | Does not create a cross-chain motif group and does not change the projector's fixed mask |

The official documentation explicitly identifies its two unindexed
symmetric-enzyme examples as ASU-local with no inter-subunit motifs; its
indexed example is likewise a single active-site residue rather than a joint
cross-subunit motif:

- [official symmetry example scope, `symmetry.md` L69-L80](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/docs/examples/symmetry.md?plain=1#L69-L80)

Therefore, those examples do not establish support for an indexed joint motif
whose defining geometry spans two symmetry copies.

## Phase 1: input parsing and fixed-mask construction

### 1. `select_fixed_atoms` is parsed correctly

The input validator maps `select_fixed_atoms` to the atom-array annotation
`is_motif_atom_with_fixed_coord` and applies the selection residue by residue:

- [selection mapping and application, `input_parsing.py` L423-L502](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/input_parsing.py#L423-L502)

The official user-facing contract says that selected input atoms are fixed in
three-dimensional space:

- [official `select_fixed_atoms` documentation, `input.md` L279-L285](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/docs/input.md?plain=1#L279-L285)

Consequently, the root cause is not a missing selection or an adapter failure
to emit the fixed-atom dictionary.

### 2. The selected contig fragments are indexed motifs

RFD3 defines a motif atom as an atom carrying any relevant conditioning. It
then classifies a motif as indexed when it is neither a small molecule nor an
unindexed motif:

```text
is_indexed_motif = not small_molecule
                   and not unindexed_motif
                   and motif
```

- [motif classification, `symmetry_utils.py` L253-L292](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L253-L292)

Because `B1-31` and `C1-30` appear in the indexed `contig`, they are indexed
motifs even though all of their atoms are coordinate-fixed.

## Phase 2: symmetry expansion and the semantic split

### 1. `is_symmetric_motif` does not change projector-level fixedness

In the accepted pre-symmetric path, RFD3 derives the symmetry frames from the
source atom array rather than using only canonical frames:

- [source-frame recovery, `symmetry_utils.py` L94-L115](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L94-L115)

It then applies those frames to construct and concatenate every symmetry
unit. The option participates in validation and frame-recovery branching, but
it neither changes `FIXED_ENTITY_ID` assignment nor creates a protected joint
cross-subunit motif group in the projector.

### 2. Only non-indexed motifs become fixed symmetry entities

`fix_3D_sym_motif_annotations()` is the root classification point. The active
mask is equivalent to:

```text
fixed symmetry motif = motif and not indexed motif
```

The code comments state that indexed motifs are connected to generated
residues and should be symmetrized at each step. Only the non-indexed set is
assigned the special fixed entity and transform identifiers:

- [indexed-motif exclusion, `atom_array.py` L98-L110](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/atom_array.py#L98-L110)
- [assignment of `FIXED_ENTITY_ID`, `atom_array.py` L32-L56](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/atom_array.py#L32-L56)

An indexed selected atom can therefore simultaneously satisfy:

```text
is_motif_atom_with_fixed_coord = true
sym_entity_id != FIXED_ENTITY_ID
```

This is the exact point at which model-level fixedness and projector-level
fixedness diverge.

## Phase 3: initial diffusion state

The symmetry sampler creates the initial state as reference coordinates plus
scaled Gaussian noise. Before addition, the noise at every fixed atom is
set to zero:

```text
noise[fixed atoms] = 0
X_initial = noise + reference coordinates
```

- [initial structure construction, `inference_sampler.py` L136-L147](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L136-L147)

Thus, at initialization,

\[
X^0_m = F_m, \qquad m \in M,
\]

where \(M\) is the selected fixed-atom set and \(F\) is the **post-build
sampler reference** `coord_atom_lvl_to_be_noised`. This distinction matters:
\(F\) is produced after input parsing and symmetry expansion; it is not
necessarily the untouched raw coordinates from the source structure. Any
input-specific build-time mapping error must be audited separately from the
runtime projection analyzed below.

## Phase 4: one complete symmetry diffusion step

For the RFD3-Mosaic command using
`allow_realignment=True` and `insert_motif_at_end=True`, one reverse-diffusion
step has the following order.

### Step 4.1: reinsert and globally augment the motif

At the beginning of every step, `allow_realignment=True` calls
`centre_random_augment_around_motif()`. Its default is
`reinsert_motif=True`. The function therefore:

1. performs one Kabsch alignment of the complete post-build sampler-reference
   fixed coordinate set to the current fixed-atom pose;
2. writes the aligned original coordinates into all fixed positions;
3. centers the structure;
4. applies one global random rotation and translation to the whole assembly.

The important nuance is that this is not a raw copy into the original
absolute coordinate frame. It restores the original **joint internal
geometry** in a globally aligned current frame, preserving SE(3) invariance.

- [per-step realignment call, `inference_sampler.py` L451-L457](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L451-L457)
- [reinsertion and augmentation implementation, `inference_sampler.py` L625-L677](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L625-L677)

Immediately after reinsertion, the cross-subunit motif can be geometrically
correct as one joint coordinate set.

### Step 4.2: add churn noise only to non-fixed atoms

The sampler generates scaled Gaussian noise and explicitly zeros it at fixed
positions:

```text
epsilon[fixed atoms] = 0
X_noisy = X_current + epsilon
```

- [per-step noise handling, `inference_sampler.py` L466-L477](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L466-L477)

Therefore,

\[
X^{\mathrm{noisy}}_m = X_m, \qquad m \in M.
\]

Random diffusion noise is not the cause of fixed-seed displacement.

### Step 4.3: denoise the expanded assembly

The diffusion module assigns an atom-level time:

\[
t_m = t\,(1-\mathbf{1}_{m\in M}).
\]

Every fixed atom therefore receives \(t_m=0\):

- [fixed-atom time mask, `RFD3_diffusion_module.py` L203-L213](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/RFD3_diffusion_module.py#L203-L213)

The official model configuration selects `f_pred: edm`:

- [EDM model setting, `rfd3_net.yaml` L55-L60](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/configs/model/components/rfd3_net.yaml#L55-L60)

Under that parameterization, the coordinate output is:

\[
D_\theta(x,t)=
\frac{\sigma_d^2}{\sigma_d^2+t^2}x+
\frac{\sigma_d t}{\sqrt{\sigma_d^2+t^2}}R_\theta(x,t).
\]

At a fixed atom, \(t=0\), so:

\[
D_\theta(x,0)=x.
\]

The learned update has an exact zero coefficient at that atom. Regardless of
the neural-network value of \(R_\theta\), the fixed-atom denoiser output equals
its noisy input:

- [EDM output formula, `RFD3_diffusion_module.py` L143-L159](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/RFD3_diffusion_module.py#L143-L159)
- [output call using atomwise `t_L`, `RFD3_diffusion_module.py` L372-L380](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/RFD3_diffusion_module.py#L372-L380)

This proves that, for the standard official EDM configuration, the denoiser
itself is not the operation that moves fixed atoms.

### Step 4.4: project the denoised output into symmetry

After the denoiser returns, the symmetry sampler calls the projector while the
noise schedule is above the symmetry cutoff:

- [post-denoising projection call, `inference_sampler.py` L515-L518](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L515-L518)

With the default `sym_step_frac=0.9`, this applies through approximately the
first 90% of reverse-diffusion steps.

The projector constructs its protected set as:

```text
projector fixed mask = sym_entity_id == FIXED_ENTITY_ID
```

It does not consult `is_motif_atom_with_fixed_coord`. For each ordinary
symmetry entity it takes the ASU coordinates and reconstructs every transform
copy:

\[
P(X)_{e,k}=R_kX_{e,\mathrm{ASU}}+T_k.
\]

The target copy's pre-existing coordinates do not enter this expression.
The direct coordinate overwrite occurs at:

- [ASU-to-copy projector, especially L344 and L364-L378,
  `symmetry_utils.py`](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L332-L380)

For an indexed target-copy fixed atom, the denoiser may produce

\[
D_m=X^{\mathrm{noisy}}_m,
\]

but the projector can immediately replace it with

\[
D'_m=P(D)_m \ne X^{\mathrm{noisy}}_m.
\]

This assignment is the first operation in the audited path that can replace
non-ASU fixed coordinates independently of their pre-projection values and
thereby change cross-copy relative geometry. The projector's earlier COM
recentring can also change indexed fixed atoms' absolute coordinates, but it
acts as a common translation and does not itself change pairwise geometry.

The sampler contains a held-motif path that may set
`partial_diffusion=True` when `allow_realignment=False`. In the projector this
only skips center-of-mass correction. It does not skip the ASU-to-copy
overwrite loop, so it is not equivalent to protecting indexed fixed atoms.

### Step 4.5: Euler update writes the projected displacement into state

The sampler computes:

\[
\Delta=
\frac{X^{\mathrm{noisy}}-D'}{t_{\hat{}}},
\]

followed by:

\[
X^{\mathrm{next}}=
X^{\mathrm{noisy}}+
s(c_t-t_{\hat{}})\Delta.
\]

Because \(c_t<t_{\hat{}}\), this can be written as:

\[
X^{\mathrm{next}}_m=
X^{\mathrm{noisy}}_m+
\alpha\left(P(D)_m-X^{\mathrm{noisy}}_m\right),
\qquad
\alpha=s\frac{t_{\hat{}}-c_t}{t_{\hat{}}}>0.
\]

Whenever projection is active and
\(P(D)_m\ne X^{\mathrm{noisy}}_m\), the projector residual enters the next
sampling state. Overshoot occurs only when
\(\alpha=s(t_{\hat{}}-c_t)/t_{\hat{}}>1\); a configured step scale greater
than one alone does not guarantee it. There is no fixed-mask overwrite after
this Euler update.

- [Euler delta and coordinate update, `inference_sampler.py` L520-L542](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L520-L542)

The complete repeated step is therefore:

```text
restore the joint reference motif in the current pose
-> globally augment the assembly
-> add zero noise to fixed atoms and noise to generated atoms
-> denoise the complete expanded assembly
-> overwrite ordinary target copies from the ASU
-> write the projected displacement into the next Euler state
```

## Additional conflict with `allow_realignment=True`

The following is a mechanistic consequence of the pinned call sequence and
rigid-transform algebra. Its contribution to a particular observed shift
should be quantified with the stage snapshots described below.

The realignment step applies a global rigid transform \(H\) to the coordinates.
If the original symmetry operator is \(S_k\), then after changing the global
coordinate frame the corresponding operator should be conjugated:

\[
S'_k=H S_k H^{-1}.
\]

The pinned sampler returns the augmentation rotation but does not use it to
conjugate the stored `sym_transform` features before calling the projector.
The projector continues to use the original \(S_k\). In general,

\[
S_kHF \ne HS_kF,
\]

unless \(H\) happens to commute with the symmetry group operation.

This has an important consequence. A seed may be perfectly orbit-closed in
the original input frame. After a random global realignment it is orbit-closed
under \(HS_kH^{-1}\), not under the unchanged \(S_k\). Reapplying the old
operator can therefore move target copies even when the initial copy mapping
was correct.

The relevant source sequence is:

- [global augmentation, `inference_sampler.py` L451-L457](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L451-L457)
- [unchanged symmetry features passed to the projector,
  `inference_sampler.py` L377-L397](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L377-L397)

This frame mismatch is especially relevant to the RFD3-Mosaic command because
`allow_realignment=True` was enabled to obtain motif reinsertion.

## Phase 5: finalization and final coordinate precedence

When fixed atoms exist, `allow_realignment=True`, and
`insert_motif_at_end=True` as in the RFD3-Mosaic command audited here, the
official finalization order is:

```text
reinsert the complete post-build sampler-reference fixed motif
-> center and globally augment the assembly
-> apply native symmetry projection using the stored operators
-> globally rigid-align the prediction to the sampler reference
```

The corresponding source ranges are:

1. motif reinsertion and global augmentation:
   [`inference_sampler.py` L552-L559](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L552-L559);
2. symmetry projection:
   [`inference_sampler.py` L561-L562](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L561-L562);
3. global Kabsch alignment:
   [`inference_sampler.py` L564-L569](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L564-L569).

The first operation can restore the complete cross-subunit motif exactly as
one coordinate set. The immediately following projector is nevertheless
allowed to overwrite its indexed target-copy atoms.

The final Kabsch operation cannot repair a changed interface relative pose. A
single global rigid transform \((Q,c)\) preserves every pairwise distance:

\[
\left\|Qx_i+c-(Qx_j+c)\right\|
=\left\|x_i-x_j\right\|.
\]

If projection has changed the left-right cross-fragment distance matrix, no
single global transform can restore it. Global alignment can only choose the
best overall reference frame and leave a residual error distributed across
the two fragments.

## Why a monomer motif can appear perfectly preserved

The statement “the monomer motif remains fixed” can describe two different
cases.

### Unindexed monomer motif

An unindexed motif can be assigned `FIXED_ENTITY_ID`. The projector skips it,
so it is protected at both the model and projector levels.

### Indexed ASU-local monomer motif

An indexed motif within one symmetry unit may still be reconstructed by the
projector. If all of its atoms receive the same rigid transform,

\[
x_i'=Rx_i+t,
\]

then its internal distances remain unchanged:

\[
\|x_i'-x_j'\|=\|x_i-x_j\|.
\]

After Kabsch alignment, its internal RMSD can therefore be nearly zero even
though its absolute pose was replaced.

### Cross-subunit interface motif

A cross-subunit interface is a joint constraint:

\[
M=M_{\mathrm{left}}^{(g)}\cup
M_{\mathrm{right}}^{(g+1)}.
\]

It requires preservation of:

- the internal geometry of each fragment;
- their relative rotation;
- their relative translation;
- their cross-fragment distance matrix;
- their interface contact network.

The projector has no representation of this union as one rigid constraint.
It works by `(entity, ASU, transform)` and reconstructs the copies separately.
The two individual fragments can consequently retain excellent internal RMSD
while their mutual interface geometry is destroyed.

This explains the characteristic observation:

```text
left fragment internally correct
right fragment internally correct
cross-chain pair incorrectly positioned
```

## Orbit-closure condition for a fixed motif

A fixed coordinate set \(F\) survives the official projection only when it is
an exact fixed point of the projector. For every entity \(e\), transform copy
\(g\), and corresponding atom slot \(i\), it must satisfy:

\[
F_{e,g,\pi_g(i)}=R_gF_{e,0,i}+T_g,
\]

where \(\pi_g\) is the correct atom and fragment-role permutation between the
ASU and copy \(g\).

Equivalently,

\[
P(F)_M=F_M.
\]

The condition can fail through any mismatch in:

- transform order;
- copy phase;
- previous-versus-next neighbor direction;
- left/right fragment role assignment;
- chain/entity mapping;
- atom-slot permutation;
- incomplete fixed-orbit coverage;
- use of static symmetry operators after global realignment.

“Cross-chain” is therefore not, by itself, a sufficient cause. The precise
statement is:

> A cross-subunit fixed motif survives only when its complete coordinate set,
> role permutation, and copy ordering are exactly closed under the sampler's
> symmetry action. Native RFD3 does not represent that joint motif group as an
> independent protected object.

## Exact source locations to report

There is no single arithmetic typo that fully describes the failure. The
source chain has three levels:

| Level | Official location | Meaning |
| --- | --- | --- |
| Root classification | [`atom_array.py` L98-L110](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/atom_array.py#L98-L110) | Only non-indexed motifs receive fixed symmetry-entity treatment |
| Direct coordinate overwrite | [`symmetry_utils.py` L376-L378](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L376-L378) | Target-copy coordinates are replaced by transformed ASU coordinates |
| Per-step trigger | [`inference_sampler.py` L515-L518](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L515-L518) | Projector runs after denoising |
| Final trigger | [`inference_sampler.py` L552-L569](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L552-L569) | Projector runs after final motif reinsertion and before global alignment |

The projector's COM correction at
[`symmetry_utils.py` L351-L354](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L351-L354)
can change indexed fixed atoms' absolute coordinates earlier in the function,
but it applies a common translation. The assignment at L376-L378 is the
critical operation that replaces target copies independently of their prior
coordinates and can change cross-copy relative geometry.

If one direct mutation statement must be named, it is the assignment at
`symmetry_utils.py:376-378`. If the deeper reason the atom is eligible for
that assignment must be named, it is the indexed-motif classification at
`atom_array.py:107`. If the final-output precedence error must be named, it is
the call at `inference_sampler.py:562` after reinsertion at L554-L559.

## What is not the cause

The pinned source rules out the following explanations:

- `select_fixed_atoms` was not parsed;
- fixed atoms received ordinary Gaussian diffusion noise;
- the EDM denoiser directly predicted a new fixed-atom coordinate;
- the contig was absent;
- the model necessarily saw only an isolated ASU;
- a final global alignment should have been able to repair arbitrary
  cross-fragment displacement.

The source does not by itself prove that every shifted output in every input
must arise from the same projector mismatch. Incorrect adapter mapping,
incorrect output pairing, or a non-orbit-closed input can produce similar
symptoms. Stage-level coordinate snapshots are the strict way to disambiguate
them.

## Local RFD3-Mosaic evidence

The local results are consistent with the source mechanism:

1. Before the finalization-order correction, individual motif fragments were
   internally close to reference, while the best cross-chain pairing was
   approximately 12.1 Å CA RMSD and retained no reference interface contacts.
   This is the predicted signature of independent rigid fragment preservation
   with incorrect relative placement.
2. After changing final write precedence so that the complete interface seed
   was reinserted after scaffold projection, smoke job `5712555` passed all
   three cross-chain pairs. It reported 496/496 matched heavy atoms per pair,
   maximum CA RMSD 0.053180 Å, maximum all-heavy-atom RMSD 0.048469 Å, and
   minimum reference-contact retention 0.978799.

These are local intervention results, not official Foundry benchmarks. They
support the coordinate-precedence diagnosis but do not prove complete
scaffold quality. The final-only correction later exposed motif-linker
junction failures consistent with the scaffold having evolved around the
pre-reinsertion motif pose before the motif was restored after diffusion.

## Implications for a correct extension

A final-only ordering change is a useful causal diagnostic:

```text
project generated scaffold
-> reinsert complete cross-subunit motif
-> globally align
-> do not project again
```

It gives the motif the final coordinate write and can restore interface
geometry. It is not a complete sampling solution because a large last-step
motif correction can break the motif-scaffold junction.

A more complete extension requires the following invariants during sampling:

1. Treat the complete cross-subunit interface as a joint constraint group or
   a symmetry orbit of joint groups, rather than as independent fixed atoms.
2. After each symmetry-active operation, restore or jointly project the
   complete motif group before the next Euler state is accepted.
3. Preserve exact atom-slot correspondence across all copies.
4. Couple symmetry-related noise or otherwise guarantee that the full runtime
   state remains orbit-closed.
5. If global realignment is retained, conjugate the symmetry operators into
   the augmented coordinate frame; otherwise disable realignment in exact
   orbit mode.
6. Validate both motif integrity and scaffold continuity. Passing one must not
   substitute for passing the other.

These changes extend the constraint semantics. They do not require changing
the checkpoint weights to explain or eliminate the coordinate-overwrite
mechanism.

## Minimal Baker-grade causal trace

For an unambiguous runtime demonstration, record four coordinate snapshots:

1. immediately before the per-step projection call at
   `inference_sampler.py:515`;
2. immediately after the call at `inference_sampler.py:518`;
3. after final motif reinsertion and before `inference_sampler.py:562`;
4. immediately after `inference_sampler.py:562`.

For each snapshot, report:

- complete joint fixed-motif RMSD;
- each fragment's internal distance-matrix error;
- cross-fragment distance-matrix error;
- cross-interface contact retention;
- orbit-closure residual

  \[
  r_{\mathrm{orbit}}=
  \max_{e,g,i}\left\|
  F_{e,g,\pi_g(i)}-(R_gF_{e,0,i}+T_g)
  \right\|.
  \]

The predicted signature is:

```text
before projector:
  joint motif correct

after projector:
  individual fragment internal errors remain small
  cross-fragment error and joint RMSD increase
```

If the first change occurs exactly across `apply_symmetry_to_X_L()`, the
input-specific causal attribution is direct. If the joint distance matrix does
not change there, the investigation must return to atom mapping, output
pairing, or adapter topology rather than blaming the projector.

## Recommended scientific wording

Avoid:

> RFD3 fixed motif functionality is broken.

Prefer:

> At Foundry commit `3b739a2`, fixed-coordinate conditioning and symmetry
> projection use different masks. The diffusion module honors
> `is_motif_atom_with_fixed_coord`, whereas the projector exempts only
> `FIXED_ENTITY_ID`. Indexed motifs remain ordinary symmetry entities and are
> reconstructed from the ASU. This is compatible with ASU-local or exactly
> orbit-closed motifs, but it does not represent an indexed inter-subunit
> interface seed as one joint protected constraint in the symmetry projector.

This wording acknowledges that the indexed-motif behavior is intentional in
the official implementation while identifying the unsupported constraint
semantics exposed by the RFD3-Mosaic use case.

## Baker-ready English summary

> At Foundry commit `3b739a2`, `select_fixed_atoms` is parsed correctly,
> fixed atoms receive zero atomwise diffusion/churn noise, and the EDM
> denoiser acts as an identity map on those atoms because their atom-level
> diffusion time is set to zero. They can still undergo a shared stochastic
> global rigid augmentation when realignment is enabled. The non-rigid
> displacement analyzed here occurs after denoising, in the symmetry
> projection. The denoiser uses `is_motif_atom_with_fixed_coord`, whereas the
> projector exempts only atoms assigned `sym_entity_id == FIXED_ENTITY_ID`.
> `fix_3D_sym_motif_annotations()` assigns that entity only to non-indexed
> motifs; indexed motifs are intentionally kept in ordinary symmetry entities
> and reconstructed from the ASU. The direct coordinate overwrite is at
> `symmetry_utils.py:376-378`.
>
> With `allow_realignment=True`, the complete motif is reinserted before each
> denoising step and again at finalization. During symmetry-active diffusion
> steps, projection follows the per-step reinsertion; in the audited final
> branch it also follows the final reinsertion. For an ASU-local motif, this
> may amount to a common rigid transform and preserve its internal geometry.
> A cross-subunit interface motif is a joint constraint across different
> symmetry copies; rebuilding those copies separately can change their
> relative transform and contact geometry. The final global Kabsch alignment
> cannot repair that change. We therefore interpret the behavior as a
> representation gap for indexed inter-subunit motif constraints, rather than
> a parsing failure or a failure of the learned denoiser.

## Official source index

- [`input.md`: input fields and `select_fixed_atoms`](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/docs/input.md?plain=1#L279-L285)
- [`symmetry.md`: scope of official motif examples](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/docs/examples/symmetry.md?plain=1#L69-L80)
- [`input_parsing.py`: fixed-mask parsing and build order](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/input_parsing.py#L423-L521)
- [`symmetry_utils.py`: expansion, classification, and projection](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py#L94-L152)
- [`atom_array.py`: fixed-entity annotation policy](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/inference/symmetry/atom_array.py#L98-L110)
- [`RFD3_diffusion_module.py`: fixed-atom EDM behavior](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/RFD3_diffusion_module.py#L143-L159)
- [`rfd3_net.yaml`: official EDM output setting](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/configs/model/components/rfd3_net.yaml#L55-L60)
- [`inference_sampler.py`: complete symmetry sampling and finalization order](https://github.com/RosettaCommons/foundry/blob/3b739a2318398f71f5d446cef6bc8d3ca3bd1295/models/rfd3/src/rfd3/model/inference_sampler.py#L401-L569)
