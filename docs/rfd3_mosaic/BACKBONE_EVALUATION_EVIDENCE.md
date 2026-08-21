# Evidence-backed evaluation of generated backbones

This document defines what RFD3-Mosaic may call **generated**, **screened**,
or **designable**.  It deliberately separates a raw backbone generator from
sequence design, structure prediction and experimental validation.  A
parseable coordinate file is not retroactively called a failed generation
because it misses a later scientific preference.

## The four stages

1. **Generation result.**  Did the program emit a parseable coordinate file
   with finite coordinates?  If yes, that artifact is `GENERATED`; later
   checks do not delete this fact.
2. **Conditioning contract.**  Were declared motif, rigid-body, symmetry,
   topology and output-count requirements preserved?  These are required
   software invariants and are reported independently as `CONTRACT_MET` or
   `CONTRACT_FLAGGED`.
3. **Backbone-only description.**  Geometry, secondary structure, radius of
   gyration, clashes, structural diversity and interface morphology are
   measurements or cohort rankings.  They are not foldability proofs.
4. **Sequence-conditioned designability.**  One or more sequences are designed
   for each backbone, refolded independently, and compared with the design.
   RFdiffusion, RFD3, Scaffold-Lab, PXDesign and BoltzGen place their main
   computational success criteria here.

## What the major systems actually do

| System | Raw-backbone hard quality gate? | Published/official evaluation |
|---|---|---|
| RFdiffusion 1 | No universal one. Auxiliary potentials guide sampling and are task-tunable. | For general designs the Nature work used ProteinMPNN and AF2, with mean pAE `<5 A`, design-to-AF2 backbone RMSD `<2 A`, and motif RMSD `<1 A` for the reported benchmark. For symmetric oligomers it designed 16 sequences per backbone and used AF2 mean pLDDT `>80` plus whole-oligomer backbone RMSD `<2 A`. |
| RFdiffusion 3 | No universal raw-CIF pass. | Official Foundry documentation explicitly defines its plotted pass rate as a **refolding pass after four MPNN-based sequence attempts**. Cluster pass is passing Foldseek clusters divided by all generated backbones. `step_scale` and `gamma_0` change the designability/diversity trade-off; they are sampler controls, not acceptance formulas. |
| Ho-Yeung/Chim et al. | Yes, but only a cohort-relative backbone prefilter. | The author notebook sequentially keeps structures strictly below the cohort median for chain-A PyMOL loop fraction, longest contiguous PyMOL loop, and chain-A carbonyl-C radius of gyration. It then uses sequence/refolding and later interface/experimental filters. There are no universal absolute loop or Rg cutoffs. |
| Scaffold-Lab | Raw geometry is described, but designability is downstream. | 100 backbones per condition, 10 ProteinMPNN sequences per backbone, then ESMFold. Unconditional designability is scTM `>0.5`; short motif-scaffolding uses scRMSD `<2 A`. Diversity is Foldseek clusters/backbones at TM threshold `0.5`. |
| PXDesign + Protenix | Generation-only mode explicitly has no confidence metrics or ranking. | PXDesign-d backbones are sequence-designed and independently evaluated with AF2-IG and Protenix. Official `Protenix-basic` uses binder ipTM `>0.8`, binder pTM `>0.8`, complex RMSD `<2.5 A`; strict `Protenix` uses `>0.85`, `>0.88`, `<2.5 A`. These are binder/refolding gates, not bare-backbone gates. |
| BoltzGen | No: raw designs are a separate pipeline artifact. | Official steps are design, inverse folding, complex refolding, design-alone refolding, analysis and filtering. The default hard RMSD threshold is `2.5 A`; filtering uses the refolded structures and can be retuned. Quality ranking and sequence-diversity selection occur after refolding. |

Primary sources:

- [RFdiffusion paper and supplementary information](https://www.nature.com/articles/s41586-023-06415-8)
- [RFdiffusion official repository and potential documentation](https://github.com/RosettaCommons/RFdiffusion)
- [RFD3 official designability/diversity documentation](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/designability_vs_diversity.md)
- [Scaffold-Lab benchmark](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014290)
- [PXDesign official evaluation table](https://github.com/bytedance/PXDesign#32-understanding-metrics--filters-summarycsv)
- [BoltzGen official pipeline](https://github.com/HannesStark/boltzgen)
- [BoltzGen official filtering implementation](https://github.com/HannesStark/boltzgen/blob/main/src/boltzgen/task/filter/filter.py)
- [Ho-Yeung/Chim et al. preprint](https://www.biorxiv.org/content/10.64898/2026.07.02.736098v1)
- author analysis notebook inspected locally at
  `/home/haixi/Documents/HYC_repeat/interface_seeded_oligomers/design_backbone_filtering/backbone_filter.ipynb`,
  SHA-256 `0a0022f335d91b5b1d15ee7257bb3d95fdac9220f1b7b1491c5eb7d9e16bfb77`;
- author ring-analysis notebook inspected locally at
  `/home/haixi/Documents/HYC_repeat/interface_seeded_oligomers/pdb_3mer_ring_analysis/pdb_3mer_ring_analysis.ipynb`,
  SHA-256 `6f2159c4481883221d58d062f8649ee048afa919979fb96378835377ce22b254`.

## Exact formulas

### Radius of gyration

For coordinates \(r_i\),

\[
c=\frac{1}{N}\sum_i r_i,
\qquad
R_g=\sqrt{\frac{1}{N}\sum_i\lVert r_i-c\rVert^2}.
\]

The Ho-Yeung notebook applies this to chain-A backbone atom name `C`
(carbonyl carbon), not C-alpha atoms.  Its cohort gate is

\[
R_g(x)<\operatorname{median}_{y\in\mathcal C}R_g(y).
\]

No paper-backed, length-independent value such as `Rg < 25 A` is a universal
protein-backbone success criterion.

### Ho-Yeung loop prefilter

Let PyMOL assign chain-A residues and let \(L_i=1\) when residue \(i\) has
`ss == 'L'`:

\[
f_{loop}=\frac{\sum_i L_i}{N_A},
\qquad
\ell_{max}=\max\{\text{length of each contiguous run of }L_i=1\}.
\]

The exact author selection is sequential strict inequality:

\[
f_{loop}<\operatorname{median}(f_{loop}),\quad
\ell_{max}<\operatorname{median}(\ell_{max}),\quad
R_g<\operatorname{median}(R_g).
\]

The medians depend on the declared cohort.  STRIDE `C+T` is not identical to
PyMOL `ss='L'`; Mosaic must label a STRIDE calculation as an approximation.

### Self-consistency RMSD

After sequence design and independent refolding, with \(K\) aligned backbone
atoms,

\[
\operatorname{scRMSD}
=\sqrt{\frac{1}{K}\sum_{i=1}^{K}\lVert
R\hat r_i+t-r_i\rVert^2},
\]

where \(R,t\) minimize the error.  `scRMSD < 2 A` is widely used for short
backbones and motif scaffolds, but it is not computable before sequence design
and refolding.

### TM-score and structural diversity

For an aligned model after optimal rigid alignment,

\[
TM=\frac{1}{L_{target}}\sum_{i=1}^{L_{ali}}
\frac{1}{1+(d_i/d_0(L_{target}))^2},
\]

with the standard length-dependent \(d_0\).  TM around `0.5` is commonly used
as a same-fold/clustering boundary.  Scaffold-Lab uses

\[
D=\frac{K}{N},
\]

where \(K\) is the number of Foldseek clusters at declared TM threshold `0.5`
and \(N\) is the number of generated backbones.  This evaluates a population;
it cannot classify one structure as diverse.

### RFdiffusion contact guidance

The official RFdiffusion potential uses C-alpha distances and

\[
x=\frac{d-d_0}{r_0},\qquad
s(d)=\frac{1-x^6}{1-x^{12}},
\]

with a continuous extension at the removable singularity.  The oligomer
potential sums weighted intra- and inter-chain contact terms.  Its documented
example uses `weight_intra=1`, `weight_inter=0.1`, but the authors explicitly
present these as tunable guidance values.  The potential is maximized during
sampling; its final numerical value is not a published pass threshold.

### RFdiffusion radius-of-gyration guidance

The official implementation minimizes a capped radial statistic:

\[
R_{cap}=\sqrt{\frac{1}{L}\sum_i
\max(15\,\mathrm{A},\lVert r_i-c\rVert)^2},
\qquad V=-wR_{cap}.
\]

This is a sampler bias: once every radial distance is below the floor, further
compaction is not rewarded.  It does not say that every accepted protein must
have one fixed Rg.

### BoltzGen post-refolding quality/diversity selection

Its official filter first requires all configured hard thresholds.  For each
ranking metric it ranks the pair `(number of filters passed, metric)`, scales
that rank by the metric's inverse-importance weight, and uses the worst scaled
rank as the quality key.  Its lazy-greedy diversity objective is documented as

\[
(1-\alpha)Q+\alpha(1-\mathrm{sequence\ identity}),
\]

with \(\alpha=0\) quality-only and \(\alpha=1\) diversity-only.  Again, this
operates on inverse-folded/refolded designs, not raw backbone coordinates.

### Raw-coordinate stereochemistry

MolProbity defines a serious all-atom clash as a non-bonded overlap greater
than `0.4 A` after adding hydrogens, and

\[
\mathrm{clashscore}=1000\frac{N_{serious\ overlaps}}{N_{atoms}}.
\]

This is a defensible raw-structure diagnostic.  Mosaic's current C-alpha
distance count is not MolProbity clashscore and must not be named as such.

## Consequences for Mosaic

- `GENERATED`: a parseable finite coordinate output exists.
- `CONTRACT_MET` / `CONTRACT_FLAGGED`: report whether user-declared fixed,
  rigid-body, symmetry, topology and output-count requirements were met. A
  contract flag is important, but it does not make the coordinate file vanish.
- `BACKBONE_SCREEN`: report continuity, exact constraint recovery,
  stereochemistry, secondary structure, Rg distributions, contacts,
  morphology and population diversity without declaring user-independent
  biological success.
- `DESIGNABLE`: reserve this word for a declared sequence-design/refolding
  protocol with its predictor, number of sequences, alignment atoms and
  thresholds recorded.
- `SELECTED`: user- or protocol-specific ranking result.  It is not synonymous
  with generation success.

For the current LHD101 cohort, historical `27/40` means 27 structures met that
revision's configured engineering check bundle.  It is not the RFdiffusion,
RFD3, Ho-Yeung, PXDesign, Protenix or BoltzGen definition of design success.
Likewise, `0/40` for an uncalibrated Mosaic compactness/support target is an
advisory controller result, not evidence that no backbone was generated.

## Executable advisory policy

Public designs may declare:

```yaml
sampling:
  screening:
    mode: advisory              # or off
    protocol: auto              # generic_backbone / hoyeung_lhd101
    retain_all_outputs: true    # intentionally cannot be false
```

For each output Mosaic writes `audits/<design-id>/screening_advice.json`.
The recommendation is lexicographic rather than a fabricated weighted score:

1. explicit geometry/runtime contract flags request contract review;
2. otherwise task-dependent proxy flags request advisory review;
3. otherwise the backbone is recommended for the declared next stage.

The record always states that the generated output was retained. The
Ho-Yeung protocol name does not apply its cohort medians to a single design;
that selection is performed only by the campaign comparison tool after all
available structures have been measured.
