# Structure metric provenance and interpretation

RFD3-Mosaic reports measurements; it does not decide whether a user should
like, retain or experimentally test a generated structure.  A raw structure
that was produced successfully remains a generated result.  Checks and
quality estimates are separate evidence.

## Evidence policy

Every reported metric belongs to exactly one of these classes:

1. **Published scientific metric.**  The definition is traceable to a primary
   paper.  A numerical cutoff is used only when that paper defines one for a
   comparable task.
2. **Cohort-relative published protocol.**  The paper defines ranking or a
   percentile, not a universal absolute cutoff.  Mosaic must reproduce the
   cohort calculation rather than invent a replacement number.
3. **User preference.**  Pore size, outer diameter and desired interface area
   can be legitimate design objectives without being universal measures of
   quality.
4. **Engineering invariant or diagnostic.**  Exact symmetry residuals,
   coordinate preservation and fast coarse clash screens verify software
   behavior.  They are not claims of biological quality.

An absolute quality cutoff with neither a published source nor explicit user
authorization must remain measurement-only.

This rule applies to sampling-controller outputs too. Contact coverage,
contiguous-patch length, Mosaic's shape loss and edge counts may help compare
controller revisions, but none may become a default backbone/interface
acceptance threshold unless a primary source defines the same quantity and
cutoff for a comparable generation task. A paper that reports a downstream
ProteinMPNN/refolding/Rosetta filter is not evidence for a raw-backbone gate.

## Match the metric to the development stage

The current Mosaic product ends at **oligomeric backbone generation**.  Its
primary comparator is therefore the backbone emitted by RFdiffusion,
RFdiffusion3 or another backbone generator before ProteinMPNN, AlphaFold,
Rosetta or experimental selection.  It is not scientifically valid to apply a
later-stage sequence-design or refolding success criterion to this earlier
artifact and call the backbone-generation run failed.

Use the following three-tier scorecard:

### Tier 1: generation and conditioning contract

Generation and the declared contract are reported as two related but distinct
facts:

- the requested number of parseable coordinate files was produced;
- the expected chains/residues and conditioned motif atoms are present;
- user-requested fixed or joint-rigid geometry is numerically preserved;
- the declared symmetry is numerically realized;
- the output contains finite coordinates and a traceable pose/diffusion seed.

A parseable finite coordinate file is reported as `GENERATED`. Motif,
rigid-body, topology, symmetry and requested-output checks are separately
reported as contract met/flagged. Continuity, stereochemistry, compactness and
interface measurements accompany the output as screening evidence; none erase
the generated artifact.

### Tier 2: backbone-only screening and population quality

Published generator benchmarks primarily characterize a *population*, not a
single universal good/bad boundary.  Mosaic should report:

- output yield and conditioning/motif recovery;
- secondary-structure and loop-percentage distributions;
- chain-length-conditioned radius-of-gyration distributions;
- MolProbity-compatible clashscore, bad bond lengths and bad bond angles;
- pairwise structural diversity and clusters per backbone at a declared
  TM-score or Foldseek clustering threshold;
- oligomer/interface descriptors such as contacts, buried surface area,
  shape complementarity and morphology, without pretending that one value is
  optimal for every user's design objective.

This follows the stage-matched Scaffold-Lab benchmark, which compares backbone
generators using secondary-structure content, length-conditioned radius of
gyration and stereochemical validity, and reports diversity at explicit
clustering thresholds.  RFdiffusion similarly emits a broad backbone set and
uses compactness/contact potentials as tunable sampling guidance rather than
as universal output acceptance laws.

### Tier 3: sequence-conditioned designability

ProteinMPNN followed by AF2/AF3 self-consistency, interface pAE/pTM, sequence
recovery, Rosetta energy and experimental assembly belong to the later
sequence/refolding pipeline.  RFdiffusion and RFD3 papers frequently use these
stages to define *design success*, but those numbers cannot be presented as
backbone-generator-only pass rates.  Mosaic will add them when that downstream
pipeline is implemented; until then they are explicitly out of scope.

Consequently, the current Mosaic normalized-Rg, tertiary-support and
long-range-contact targets are useful controller-development diagnostics.
They are not RFdiffusion/RFD3-standard backbone acceptance criteria and must
remain advisory until calibrated on a declared reference cohort.

## Current metric audit

| Mosaic quantity | Correct interpretation | Published basis | Current action |
|---|---|---|---|
| raw CIF count | execution artifact count | none required | report as `GENERATED`, never as aesthetic success/failure |
| fixed motif/interface RMSD | user contract and numerical preservation check | RFdiffusion uses motif backbone RMSD in its later AF2 validation, but Mosaic's current `0.5 A` runtime cutoff is not a universal experimental-quality threshold | keep as engineering invariant; expose RMSD |
| exact-symmetry coordinate and distance-matrix residuals | numerical implementation invariant | none required | keep as engineering invariant; do not call it fold quality |
| peptide continuity | backbone geometry diagnostic | Engh & Huber stereochemical parameters, Acta Cryst. A47, 392-400 (1991), DOI `10.1107/S0108767391001071` | cite the definition; validate exact tolerances against the reference distribution |
| current `CA < 3.0 A` clash count | fast coarse warning | not the MolProbity definition | label provisional engineering diagnostic; add published all-atom clashscore |
| all-atom clashscore | serious non-bonded overlaps per 1000 atoms | Davis et al., *Nucleic Acids Research* 35, W375-W383 (2007), DOI `10.1093/nar/gkm216`; Chen et al., *Acta Cryst. D* 66, 12-21 (2010), DOI `10.1107/S0907444909042073` | preferred published steric-quality metric |
| absolute chain `Rg <= 25 A` | historical coarse expansion screen | no universal published cutoff independent of chain length | remove from scientific-pass language; retain only as diagnostic until replaced |
| length-normalized Rg | compactness measurement | folded proteins follow length-dependent scaling close to `Rg ~ N^0.34`; Hofmann et al., *PNAS* 109, 16155-16160 (2012), DOI `10.1073/pnas.1207719109` | report continuously or relative to an explicit reference cohort; no invented universal cutoff |
| Ho-Yeung chain-A loop fraction, longest loop and carbonyl-C Rg | cohort-relative backbone screen | Chim et al., bioRxiv 2026.07.02.736098 and the author `backbone_filter.ipynb`; each of the three values must be strictly below its cohort median, with no universal absolute cutoff | reproduce all three definitions; label STRIDE C+T as an approximation to PyMOL `ss='L'` |
| long-range/tertiary contacts | topology and folding-support descriptor | Plaxco, Simons & Baker, *J. Mol. Biol.* 277, 985-994 (1998), DOI `10.1006/jmbi.1998.1645` | metric definition is literature-motivated; current Mosaic support-fraction cutoff is provisional and stays advisory |
| interface shape complementarity | protein-interface surface complementarity | Lawrence & Colman, *J. Mol. Biol.* 234, 946-950 (1993), DOI `10.1006/jmbi.1993.1648` | implement/report the published `Sc` statistic; do not rename Mosaic's current proxy loss as `Sc` |
| buried interface surface area | physical interface size descriptor | Janin & Rodier, *Proteins* 23, 580-587 (1995), DOI `10.1002/prot.340230413` | report BSA; use task/reference distributions, not one universal cutoff |
| current coverage/continuity/shape proxy thresholds | online sampling objectives | no universal published cutoff for the current proxy implementation | retain as controller diagnostics and calibration targets, not authoritative user verdicts |
| pore diameter and outer diameter | morphology/user design preference | no universal quality optimum | measurement-only unless the user supplies bounds |
| backbone-generator population benchmark | distributions of secondary structure, length-conditioned Rg, stereochemical validity and structural diversity | Zheng et al., *PLOS Computational Biology* (2026), Scaffold-Lab, DOI `10.1371/journal.pcbi.1014290` | adopt as the stage-matched evaluation framework; declare clustering settings and avoid universal aesthetic cutoffs |
| AF2/RF validation | sequence-conditioned designability check after sequence design | Watson et al., *Nature* 620, 1089-1100 (2023), DOI `10.1038/s41586-023-06415-8`: mean pAE below 5, design-vs-AF2 backbone RMSD below 2 A, and scaffolded-site backbone RMSD below 1 A for that paper's in-silico benchmark | future sequence/refolding stage; do not apply to backbone-only output |

## Consequence for the LHD101 40-design cohort

All forty submitted LHD101 structures were generated, retained the supplied
interface and had continuous backbones.  The historical `27/40` number means
that 27 met the complete set of checks configured in that software revision.
It is not a published Ho-Yeung acceptance rate and it is not a statement that
the other thirteen structures are unusable.

In particular, the historical bundle included a provisional CA-distance clash
screen and an engineering chain-Rg ceiling.  The separate `0/40` monomer-core
target result used additional uncalibrated normalized-Rg/contact-support
targets.  Neither number is an RFdiffusion/RFD3 backbone-generation success
rate.  Those flags remain useful for inspection and controller development,
but they must be reported as measurements until the corresponding published
or cohort-calibrated validation is available.  The directory name
`accepted_strict_27` is retained only for path stability and historical
reproducibility.

## Required follow-up

1. Add an all-atom MolProbity-compatible clashscore after structure generation.
2. Compute secondary-structure/loop percentage reproducibly, then implement
   the Ho-Yeung cohort percentile exactly.
3. Replace the scientific interpretation of absolute Rg with length-aware and
   cohort-relative reporting.
4. Add published interface BSA and Lawrence-Colman `Sc` measurements.
5. Add population-level TM/Foldseek diversity with the clustering threshold,
   tool version and denominator written into every campaign report.
6. Keep Mosaic's online proxy objectives for sampling, but label their values
   as proxy/controller diagnostics until calibrated against the published
   metrics and reference structures.

The detailed cross-system evidence table, equations and exact stage boundaries
are maintained in [BACKBONE_EVALUATION_EVIDENCE.md](BACKBONE_EVALUATION_EVIDENCE.md).
