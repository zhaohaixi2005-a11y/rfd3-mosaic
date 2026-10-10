# Rigid-mobility mathematical contract

## Scope and claim boundary

This document is the normative description of how RFD3-Mosaic translates and
rotates a compiler-declared mobile rigid component during backbone generation.
It records the equations, search constraints, proposal acceptance rule and
parameter provenance. The implementation is in
`rfd3/inference/symmetry/motif_mobility.py` and
`rfd3/inference/symmetry/scaffold_guidance.py`.

The controller minimizes an explicit inference-time geometry objective. Its
energy is **not** a physical free energy, an RFD3 confidence, a folding score,
or evidence that a final backbone is designable. For scaffold-objective
proposals, one accepted update proves only this narrower statement:

> On one fixed current denoised-backbone snapshot, replacing the current
> rigid-component pose by the candidate pose lowers the declared local
> geometry objective and satisfies the protected runtime constraints.

The comparison holds the scaffold snapshot fixed. It therefore does not
confound a pose change with a different diffusion-noise realization. Later
RFD3 steps may change the scaffold again; final continuity, clash, symmetry,
motif and interface observations are reported by separate audits.

For complete-scaffold transport, this frozen-scaffold objective is a proposal
surrogate. The final geometric validator evaluates the jointly transported
seed, reference and generated scaffold. It does not make the surrogate a
differentiable objective of that transported state; useful motion and design
quality still require matched generation experiments.

The optional denoiser-fit proposal estimates a rigid pose from the network
prediction instead of minimizing that scaffold objective. Its updates enforce
step and cumulative bounds and publish all orbit fits atomically; they do not
by themselves certify a decrease in scaffold energy. Numerically unchanged
fits are recorded as no motion.

## Coordinate state and exact invariants

Let the supplied master component contain coordinates

\[
X^0 = (x^0_1,\ldots,x^0_m), \qquad x^0_i\in\mathbb{R}^3,
\]

with center \(c=m^{-1}\sum_i x^0_i\). Its current pose is one element of
\(SE(3)\), \(Q=(R,t)\), and is materialized as

\[
x_i(Q)=(x^0_i-c)R^\top+c+t,
\qquad R\in SO(3),\;t\in\mathbb{R}^3.
\]

Every symmetry copy is generated from this one master pose by the
compiler-declared group action \(g=(R_g,t_g)\):

\[
x_{g,i}=x_i(Q)R_g^\top+t_g.
\]

Consequently, pairwise distances inside the supplied component are invariant:

\[
\|x_i(Q)-x_j(Q)\|_2=\|x^0_i-x^0_j\|_2.
\]

Copies never move independently. A `fixed` component has \(R=I,t=0\) at all
steps. Only an explicitly `bounded_mobile` complete master component can be
updated.

## Pre-diffusion pose diversity

Large assembly-level exploration belongs primarily to per-design
initialization, before RFD3. Radius and axial variables are stratified with a
randomized Latin hypercube. Orientation is sampled Haar-uniformly on \(SO(3)\)
with the Shoemake unit-quaternion construction. For independent
\(u_1,u_2,u_3\in[0,1)\), Mosaic uses

\[
q=(
\sqrt{1-u_1}\sin 2\pi u_2,
\sqrt{1-u_1}\cos 2\pi u_2,
\sqrt{u_1}\sin 2\pi u_3,
\sqrt{u_1}\cos 2\pi u_3
).
\]

This avoids the orientation bias of independently uniform Euler angles. The
random seed and the resulting quaternion, rotation matrix, radius and axial
offset are written to pose provenance. Hard linker, clash, topology and user
constraint checks reject infeasible initializations.

Pre-diffusion feasibility rejects fixed-copy atom clashes and insufficient
linker contour length. Cyclic wedge occupancy, tangent alignment, axis
clearance (3.8 A), fixed-atom chord clearance (2.0 A) and terminal back-turn
(120 degrees) are advisory routing descriptors. They are not necessary
conditions for a flexible generated chain and no longer independently reject
a candidate. Explicit required interfaces and user objectives remain hard.

One task realizes one input pose and shares it across every diffusion sample.
`prepare-poses` ranks feasible candidates and freezes different starting poses
into separate tasks. The same runtime mobility policy remains attached to each
task; movable trajectories may end at different poses or converge across
tasks. See [task pose semantics and selection rules](TASK_POSES.zh-CN.md).

`uniform_so3` remains the ordinary default for every symmetry family.  When a
workflow has an independently justified preferred axis, it may explicitly use
`principal_axis_cone`.  This is an optional orientation prior, not a
replacement for (SO(3)) or (SE(3)), and no maintained C, D, T, O or I
workflow enables it implicitly.  For cone half-angle
\(\theta_{\max}\), Mosaic samples solid angle rather than sampling tilt angle
uniformly:

\[
\cos\theta=1-u_1(1-\cos\theta_{\max}),\qquad
\phi=2\pi u_2,\qquad \psi=2\pi u_3.
\]

Here \(\phi\) is azimuth around the declared symmetry axis and \(\psi\) is a
complete roll around the sampled target direction.  A future mixed prior can
be expressed as

\[
p(R)=\alpha\,p_{\mathrm{Haar}}(R)+(1-\alpha)\,p_{\mathrm{cone}}(R),
\]

but Mosaic does not invent \(\alpha\) without a workflow-specific benchmark.
The Ho-Yeung interface-seeded reference uses broad three-angle orientation
sampling, so Mosaic's maintained LHD examples retain `uniform_so3`.

### Hard endpoint reachability

For a generated segment containing \(N\) residues between two fixed CA
anchors separated by \(d\), there are \(N+1\) anchor-to-anchor CA intervals.
Using \(\ell_{CA}=3.8\) A as the contour-length reference, the minimum residue
count required by that endpoint placement is

\[
N_{\min}=\max\left(0,\left\lceil\frac{d}{\ell_{CA}}\right\rceil-1\right).
\]

If \(N_{\min}>N_{\max}\) for any non-break scaffold link, strict standalone
compilation rejects the pose before RFD3.  This is a necessary reachability
condition, not a sufficient foldability test and not a quality ranking.  A
relaxed diagnostic compilation may still write the report, but cannot be used
as an executable run input.

## Optional generated-chain route ownership

New ordinary tasks compile this straight-chord prior as advisory
(enabled:false), because valid folded reference backbones can violate it.
Explicit native enabled:true retains the opt-in required policy. A complete
scaffold uses its separate reference contract, which remains required.
For every generated run
\(c\) bounded by compiler-declared fixed anchors \(a_c,b_c\), define its chord
\(L_c=[a_c,b_c]\).  For generated CA coordinate \(x\), Mosaic evaluates

\[
d_c(x)=\operatorname{dist}(x,L_c),\qquad
d_k(x)=\operatorname{dist}(x,L_k),
\]

and the relative routing excess

\[
r_{c,k}(x)=\operatorname{ReLU}\left(
\frac{m(s)+d_c(x)-d_k(x)}{\ell_{CA}}
\right).
\]

The differentiable route-ownership term is

\[
E_{\mathrm{route}}=
\frac{1}{|\mathcal G|}\sum_{(c,k,x)\in\mathcal G}r_{c,k}(x)^2,
\]

where \(\mathcal G\) contains generated CA coordinates and adjacent bond
midpoints in two-fixed-anchor runs, paired with every competing run on other
chains. For a run of \(n\) residues, the margin is
\(m(s)=c\min(1,s/\tau,(n+1-s)/\tau)\), with defaults \(c=3.2\) Å and
\(\tau=2\) residues. Thus ties also incur a positive penalty away from anchors.
It does **not** pull the chain onto a
straight line, choose an inward or outward ring curvature, repel all
inter-chain contacts, or move fixed atoms.  Literal cross-chain CA-segment
collision remains a separate barrier, and continuity projection remains a
separate peptide-path constraint.

This is a topology-aware routing prior, not a proof of ambient isotopy or
foldability.  Open chains do not have a closure-independent linking number;
therefore Mosaic reports route violations and segment proximity explicitly
rather than claiming a mathematical no-entanglement theorem.

## Scaffold-pose objective

For a current scaffold coordinate snapshot \(S\) and candidate rigid pose
\(Q\), the base objective is

\[
E_{\mathrm{scaffold}}(Q;S)=
w_j E_{\mathrm{junction}}
+w_c E_{\mathrm{clash}}
+w_t E_{\mathrm{tilt}}
+w_p E_{\mathrm{prior}}.
\]

### Junction term

For compiler-derived fixed/generated boundary pairs \(\mathcal J\), with CA
distance \(d_k(Q,S)\), target \(d_0\), and pseudo-Huber scale \(\delta\),

\[
E_{\mathrm{junction}}=
\frac{1}{|\mathcal J|}\sum_{k\in\mathcal J}
\delta^2\left(
\sqrt{1+\left(\frac{d_k-d_0}{\delta}\right)^2}-1
\right).
\]

The current defaults are \(d_0=3.8\) A and \(\delta=0.25\) A. The former is a
coarse CA-neighbour geometry reference, not a final-backbone pass threshold.

### Clash term

For non-bonded mobile-fixed/generated-CA pairs \(\mathcal C\), with distance
\(d_{ij}\) and exclusion radius \(d_c\),

\[
E_{\mathrm{clash}}=
\frac{1}{|\mathcal C|}\sum_{(i,j)\in\mathcal C}
\max(0,d_c-d_{ij})^2.
\]

The current differentiable controller uses \(d_c=3.0\) A. Final structure
audits remain authoritative for written-coordinate clashes.

### Tilt interval term

Let \(a\) be the component principal axis, \(z\) the relevant symmetry axis,
and \(\theta=\arccos(|(Ra)\cdot z|)\). With maximum target tilt
\(\theta_{\max}\),

\[
E_{\mathrm{tilt}}=
\max(0,\cos\theta_{\max}-|(Ra)\cdot z|)^2.
\]

This is an interval penalty. It does not pull all accepted structures to one
exact tilt. Axis-independent T/O/I mobility omits this cyclic-axis term rather
than inventing a preferred axis.

### Soft pose prior

With translation scale \(\sigma_t\) and rotation scale
\(s_R=2\sin(\theta_R/2)\),

\[
E_{\mathrm{prior}}=
\left\|\frac{t}{\sigma_t}\right\|_2^2
+\frac{\|R-I\|_F^2}{2s_R^2}.
\]

This discourages gratuitous motion but is not a hard lock. The effective prior
scales expand with the declared search envelope:

\[
\sigma_t=\max(\sigma_{t,0},T_{\max}/3),\qquad
\theta_R=\max(\theta_{R,0},\Theta_{\max}/3).
\]

### Optional generated-interface term

For `create_symmetric_interface`, the runtime may add the graph-interface
packing objective:

\[
E(Q;S)=E_{\mathrm{scaffold}}(Q;S)+E_{\mathrm{interface}}(Q;S).
\]

Its published-contact prior, patch terms and safety boundary are specified in
[Generated-interface packing guidance](PACKING_GUIDANCE.md). Supplied-interface
scaffolding does not invent this second interface objective by default, but it
may request it explicitly with
`sampling.scaffold_packing: symmetric_generated`; the supplied joint-rigid
interface remains an independent hard invariant.

### Generated-scaffold core term

For ordinary `create_symmetric_interface` designs and explicit cross-chain
supplied-interface designs, Mosaic enables the intra-chain scaffold-core
objective by default. This is independent of the inter-chain interface
objective and may be disabled explicitly with
`guidance.intra_chain_weight: 0` for an ablation. For generated residue \(i\),
let \(s_i\) be its differentiable count of sequence-distant CA contacts and
\(s_0\) the configured support target. The mean support deficit is

\[
E_{\mathrm{support}}=
\frac{1}{N}\sum_i \max(0,s_0-s_i)^2.
\]

A mean alone can hide one long unsupported arm. Define
\(d_i=\max(0,s_0-s_i)^2\), average \(d_i\) in sequence-local windows of width
equal to the controller's tertiary-contact separation, and apply a normalized
smooth maximum:

\[
\bar d_j=\frac{1}{W}\sum_{i=j}^{j+W-1}d_i,
\qquad
E_{\mathrm{worst}}=
\tau\left[\log\sum_j \exp(\bar d_j/\tau)-\log M\right].
\]

The normalization avoids changing the scale merely because a chain has more
windows. This is an inference-time controller term, not a universal backbone
acceptance threshold. It is combined with long-range contacts and
length-normalized radius of gyration; exact fixed coordinates remain owned by
the constraint projector.

### Robust finite-assembly capture

For a movable joint-rigid component under supported finite symmetry, early
capture adds a center objective without altering any coordinate inside the
rigid group. For each
generated residue \(i\), let \(s_i\) be its smooth sequence-distant contact
support and define

\[
w_i=w_{\min}+\operatorname{clip}\left(\frac{s_i}{s_0},0,1\right),
\qquad w_{\min}=0.05.
\]

For generated chain \(k\), its ordinary and support-weighted centers are

\[
C_k^{\mathrm{COM}}=\frac{1}{N_k}\sum_i x_i,
\qquad
C_k^{\mathrm{core}}=\frac{\sum_i w_i x_i}{\sum_i w_i}.
\]

For each symmetry copy of the rigid group, the two nearest generated-chain
identities are selected from detached geometry. Their centers remain
differentiable. If \(u_c\in[0,1]\) is progress through the capture phase, the
target midpoint is

\[
M_k(u_c)=(1-u_c)\frac{C_a^{\mathrm{COM}}+C_b^{\mathrm{COM}}}{2}
+u_c\frac{C_a^{\mathrm{core}}+C_b^{\mathrm{core}}}{2},
\]

and the capture term for rigid-copy center \(G_k(Q)\) is

\[
E_{\mathrm{capture}}=
\frac{1}{K}\sum_k
\left\|\frac{G_k(Q)-M_k(u_c)}{d_{\mathrm{contact}}}\right\|_2^2.
\]

Thus the early Ho-Yeung-style midpoint supplies a broad capture signal before
a core exists, while later capture is less influenced by long unsupported
arms. Neighbours are local finite-group neighbours: Cn/Dn can use their
physical symmetry-aligned frame, whereas T/O/I use bounded Cartesian SE(3)
without a fabricated global axis. The term is active only during `capture`,
only for compiler-recognized movable rigid groups with generated scaffold, and
never for locked components.

## Deterministic SE(3) proposal

At the current pose, Mosaic differentiates the scalar objective with respect
to an infinitesimal rotation vector \(\omega\in\mathbb{R}^3\) and translation
increment \(v\in\mathbb{R}^3\):

\[
g_R=\nabla_{\omega}E,\qquad g_t=\nabla_vE.
\]

The negative gradients are projected into the compiler-declared motion
subspaces \(P_R,P_t\) and normalized:

\[
\hat d_R=\operatorname{normalize}(-P_Rg_R),\qquad
\hat d_t=\operatorname{normalize}(-P_tg_t).
\]

Examples include full bounded \(SE(3)\), radial translation only,
radial-plus-axial translation, rotation only, and radial/axial translation
with rotation. Ordinary cyclic interface creation suppresses arbitrary
tangential translation. A zero projected gradient produces no local-gradient
motion. During the early `capture` phase only, Mosaic also evaluates
deterministic signed probes along every compiler-permitted translation and
rotation basis direction, plus a bounded set of coupled translation/rotation
probes. This gives a small reproducible multi-start search around the current
pose without relaxing either trust-region or cumulative bounds. `settle` and
`polish` use the projected gradient alone. Locked components never enter this
code.

For full bounded \(SE(3)\) with a cyclic axis, the proposal basis is the local
orthonormal frame

\[
B=(e_r,e_\phi,e_z),\qquad e_\phi=e_z\times e_r.
\]

Signed translation and rotation probes are evaluated along all three axes;
their coupled probes include radial, tangential and axial combinations.
Because \(B\) spans \(\mathbb{R}^3\), this changes the finite multi-start
neighbourhood, not the declared degrees of freedom. Axis-free full \(SE(3)\)
uses the Cartesian basis.

For scheduled rotation and translation step sizes \(h_R,h_t\), raw increments
are

\[
\Delta\omega=h_R\hat d_R,\qquad \Delta t=h_t\hat d_t.
\]

They are clipped first by per-update trust regions and then by cumulative
bounds relative to the compiled initial pose:

\[
\|\Delta t\|_2\le d_{\mathrm{step}},\quad
\angle(\Delta R)\le\theta_{\mathrm{step}},\quad
\|t\|_2\le T_{\max},\quad
\angle(R)\le\Theta_{\max}.
\]

`T_max` and `Theta_max` are user/task search-envelope parameters, not
literature-defined measures of backbone quality.

### Current ordinary-task defaults

These defaults are compiled only when the user permits component-pose
optimization. They are measured relative to the compiled initial pose. An
expert declaration may override them explicitly.

| Public task and arrangement | Motion subspace | Active window | Base response | Per-update limits | Cumulative limits |
| --- | --- | ---: | ---: | ---: | ---: |
| any task, `locked` | none | none | 0 | 0 A / 0 deg | 0 A / 0 deg |
| ordinary movable finite-symmetry task, `optimize_components` | Cn/Dn: radial + axial translation and rotation; T/O/I or explicit free motion: bounded SE(3) | 2%-88% | 0.55 | 2.5 A / 6 deg | 60 A / 90 deg |

The `60 A / 90 deg` envelope is deliberately a loose anti-divergence guard,
not a preferred displacement and not a quality score. All atoms inside a
declared rigid component remain rigid while its complete orbit may cross a
large distance to reach a locally packed state. Movement stops because no
tested safe proposal lowers the declared objective, not because a prescribed
distance was consumed. The generated polymer is therefore not intended to
compensate for a bad movable pose by becoming a long loop. Pre-RFD3 hard
feasibility, objective descent, line search and transaction safety can accept
a smaller update or no update.

The heuristic defaults used by the ordinary scaffold controller are:

| Parameter | Current default |
| --- | ---: |
| junction weight \(w_j\) | 1.0 |
| clash weight \(w_c\) | 1.0 |
| tilt-interval weight \(w_t\) | 0.25 |
| soft-prior weight \(w_p\) | 0.05 |
| junction CA reference \(d_0\) | 3.8 A |
| pseudo-Huber scale \(\delta\) | 0.25 A |
| soft clash distance \(d_c\) | 3.0 A |

Every resolved run records the effective values. Changing a default requires
both a code change and a provenance-visible configuration change; old
snapshots retain the values actually used.

These coefficients have not been established as optimal by a matched
generation benchmark. Junction and clash terms are in squared distance units;
tilt and the normalized pose prior are dimensionless. Equal coefficients do
not imply equal influence. Use the recorded weighted term values and allowed
pose gradients to identify dominant forces before testing coefficient changes.

## Time-normalized capture, settle and polish schedule

For diffusion progress \(p\) and a component active on \((p_s,p_e)\), define

\[
u=\frac{p-p_s}{p_e-p_s}.
\]

Outside the open interval, the component is frozen. Inside it:

| Local interval | Phase | Relative response |
| --- | --- | ---: |
| \(0\le u<0.4\) | capture | 1.0 |
| \(0.4\le u<0.8\) | settle | 0.5 |
| \(0.8\le u<1.0\) | polish | 0.2 |

The implementation retains the declared base sensitivity \(r\). With default
scale triplet \((s_c,s_s,s_p)=(5,2.5,1)\), it first computes

\[
a_c=\min(1,r s_c),
\]

then uses

\[
a_{\mathrm{phase}}=a_c\frac{s_{\mathrm{phase}}}{s_c}.
\]

Therefore settle and polish remain exactly 50% and 20% of capture even when a
larger base response would otherwise saturate more than one phase. The schedule
uses percentages, not absolute timestep numbers, so 50-, 100- and 200-step
runs share the same semantics.

## Backtracking line search and local acceptance

Mosaic first tests scales \(\alpha\in(1,0.5,0.25)\). If no feasible
improving candidate exists in this coarse round, it tests six further halvings
from \(1/8\) to \(1/256\). An explicitly supplied scale sequence is used
exactly, without automatic extension. At each scale it evaluates

\[
Q_\alpha=(\exp(\alpha\Delta\omega)R,\;t+\alpha\Delta t)
\]

followed by projection onto cumulative bounds, on the same scaffold snapshot.
The actual increment after projection is checked against the step bounds:
SO(3) projection across the pi branch can increase that increment. A current
pose already outside its declared cumulative bound is rejected at entry.
The first finite candidate satisfying

\[
E(Q_\alpha;S)<E(Q;S)
\]

and passing the geometry validator is eligible. During `capture`, this comparison is applied to the gradient
direction and all deterministic multi-start probes. Let
\(\Delta E_*=E_0-E_*\) be the gain of the best improving candidate. The
eligible near-optimal pool is

\[
\mathcal P=\{q_j:E_0-E_j\ge0.75\,\Delta E_*\}.
\]

One member of \(\mathcal P\) is selected reproducibly from the design's
diffusion seed, controller update count and stable orbit identity (SHA-256,
mixed modulo \(2^{64}\)). Reordering named orbits does not reassign their
random streams. A rejected complete-scaffold transaction rolls back its copied
controller counter, so a later attempt can reuse that counter; it is not
guaranteed to be a distinct substream at every diffusion step. Independent
designs need not collapse to the same local minimum, while no non-improving
candidate can be accepted. Without a selection seed, the minimum-energy member
is retained for backward compatibility. During `settle` and `polish`, the
first improving gradient-line-search candidate is retained. If none lowers the
objective, the pose is unchanged.

For several mobile orbits, all proposals are computed from one immutable
snapshot. The complete candidate is committed atomically only when at least
one local proposal was accepted and

\[
E_{\mathrm{joint}}(Q'_1,\ldots,Q'_n;S)
<E_{\mathrm{joint}}(Q_1,\ldots,Q_n;S)-10^{-12}.
\]

If the combined proposal fails, all proposed increments are scaled together
by successive halves, down to \(1/256\), on the same snapshot. Rotations use
SO(3) interpolation. Every trial rechecks step/total bounds, geometry and the
joint objective; only one complete candidate is committed. Inter-orbit clash
energy participates in both local gradients and the joint decision.

The \(10^{-12}\) margin is numerical tie-breaking, not a scientific quality
threshold. Exhausting this bounded search rolls every orbit back; no ordered
subset is committed. Failure does not prove that no feasible pose exists.

## Packing-coupled transaction acceptance

When generated-patch motion and rigid-component motion are coupled, Mosaic
requires all of the following:

1. some coordinate proposal exists;
2. graph-interface energy decreases by more than \(10^{-10}\);
3. combined scaffold-plus-interface energy decreases by more than
   \(10^{-10}\);
4. graph-interface proposal safety is satisfied;
5. protected edge and global minimum safety distances do not worsen below
   their baseline-safe floors;
6. junction loss does not worsen beyond its baseline or declared protection
   limit.

The \(10^{-10}\) values are numerical comparison tolerances. Coverage,
orientation and shape controller references are optimization diagnostics, not
universal final-backbone pass criteria. Failure of any condition restores the
complete pose and patch snapshot.

## What is recorded for every attempted update

The runtime result diagnostics retain a `trajectory` entry for each controller
call, including calls outside the active window. For every active orbit the
record includes:

- diffusion progress, active-window fraction and `capture`/`settle`/`polish`
  phase;
- declared subspace, base response, temporal response and effective response;
- unprojected and motion-subspace-projected rotation/translation gradient
  norms;
- objective weights and every initial/proposed/delta term (`total`,
  `junction`, `clash`, `tilt`, `prior`, plus optional pose/packing energy);
- proposed translation vector, translation norm and rotation angle;
- every backtracking trial scale, candidate energy, finiteness and improvement
  decision;
- weighted term values, allowed translation/rotation derivatives and their
  first-order step sensitivities (an optimizer diagnostic, not a quality score);
- joint backtracking trials and actual committed increments;
- local `accepted`, joint `accepted`, transaction `committed` and rollback
  outcome;
- current cumulative translation/rotation and the configured per-update and
  cumulative bounds.

This distinguishes three events that must not be conflated: a local proposal
can lower its own orbit objective, the complete multi-orbit assembly can still
reject it, and a packing-coupled transaction can roll the complete candidate
back for violating a protected condition. The written trajectory, rather than
the mere existence of a generated CIF, answers why a particular update moved,
was shortened, or remained unchanged.

After outer packing rollback, `applied`, per-orbit `committed`, committed
increments and saved orbit poses describe the restored state. The provisional
inner decision remains separately available as `pose_proposal_accepted`.
`scaffold_pose_search_summary` counts local rejection reasons and committed
movement. Its path length is a sum of increments across orbits, not final
displacement; geometry failure counts can overlap.

## Native RFD3 mobile proposal compatibility

Use `pose.proposal: scaffold_objectives` for native RFD3 mobile components.
New `bounded_mobile` declarations that omit `proposal` select this backend.
An explicit legacy `denoiser_fit` request is retained during parsing and
rejected by native sampler construction, before weights or GPU generation.
Fixed components retain their existing behavior.

The generic controller still implements inverse orbit averaging and a proper
Kabsch fit for denoisers that provide a pose signal. Native RFD3 EDM does not
provide that signal on fixed atoms: their effective noise level is zero, so
the output scaling gives `X_out_fixed = X_input_fixed` independently of the
network update. Fitting these fixed atoms back to their template can accumulate
numerical drift but cannot optimize their pose from a learned prediction.
The model's fixed mask and pretrained output scaling have not been altered.

## Parameter provenance

| Quantity | Meaning | Provenance |
| --- | --- | --- |
| Haar SO(3), group expansion, Kabsch fit | geometric construction | mathematical identity/algorithm |
| motion subspace, fixed/mobile state, cumulative bounds | allowed user task | compiler/user contract |
| 40/40/20 and 1.0/0.5/0.2 | coarse-to-fine runtime policy | Mosaic engineering default; configurable, not a published biological law |
| 3.8 A junction reference | local CA-neighbour geometry target | differentiable peptide-geometry reference; not a final pass cutoff |
| 3.0 A controller clash radius | soft proposal exclusion | Mosaic safety default; final audits are separate |
| objective weights and tilt interval | heuristic optimization settings | Mosaic defaults; not empirically optimal; recorded in runtime provenance |
| RF-style broad contact prior | early generated-interface capture | public RFdiffusion potential and schedule |
| RFD3 denoiser prediction | learned backbone signal | loaded RFD3 checkpoint |
| line-search and atomic rollback tolerances | numerical stability | software contract, not scientific screening |

There is no universal backbone-generator standard for a maximum rigid-seed
translation or rotation. Public RFdiffusion symmetric motif scaffolding fixes
the motif pose relative to canonical symmetry axes. The Ho-Yeung
interface-seeded extension instead samples broad initial orientation/distance
and applies COM dragging without a cumulative physical-unit cap. FrameDiff and
FrameFlow diffuse residue frames over SE(3), which is not the same contract as
preserving a supplied multi-chain interface as one rigid body. Mosaic therefore
keeps large diversity in the outer pose ensemble and treats runtime cumulative
bounds as an explicit, auditable search envelope.

Primary context:

- RFdiffusion: <https://doi.org/10.1038/s41586-023-06415-8>
- RFdiffusion implementation and symmetric motif documentation:
  <https://github.com/RosettaCommons/RFdiffusion>
- FrameDiff: <https://proceedings.mlr.press/v202/yim23a.html>
- FrameFlow motif scaffolding:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10802670/>
