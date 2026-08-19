# Ho-Yeung LHD101 backbone comparison

This is a paper-specific validation experiment, not a public Mosaic design
mode. It uses Mosaic/RFD3 on the LHD101 interface-seed task reported in
[Chim et al., bioRxiv 2026.07.02.736098](https://www.biorxiv.org/content/10.64898/2026.07.02.736098v1).
It does not execute the authors' RFdiffusion1 fork.

## Matched generation task

- interface seed: LHD101, PDB 7MWR;
- symmetry: C3;
- generated connection: 70–100 residues between the two supplied seed sides;
- diffusion: 50 steps;
- sample size: 1,000 independently sampled Mosaic backbones;
- complete LHD101 interface geometry remains one joint rigid body;
- Mosaic additionally allows bounded radial, axial and rotational motion of
  that complete body while reconstructing exact C3 copies;
- the current LHD101 template uses `intra_chain_weight: 1.0` and
  `inter_chain_weight: 0.10`. It therefore prioritizes a supported compact
  monomer scaffold while retaining only weak generated--generated attraction;
  excessive cross-chain contacts receive a normalized soft penalty rather
  than an absolute contact ban;
- the complete supplied interface seed may move only as one bounded rigid
  body. Every update is followed by the same exact C3/fixed-target projector.

Bare RFD3 is not the scientific comparator because it does not implement the
published interface-seed assembly mechanism. The primary comparison is the
matched published Ho-Yeung task/result protocol versus the complete
RFD3-Mosaic method. An unguided RFD3 run may be retained only as an internal
ablation and is not allocated the 1,000-backbone main budget.

The experiment is frozen in
`experiments/lrz_mosaic_lhd101_c3_guided_50step_template.yaml`. The launcher
splits 1,000 outputs into recoverable jobs while retaining one campaign
manifest and unique deterministic seeds.

## What the paper measured at the backbone stage

The paper and its supplementary Methods report:

- 5,000–10,000 generated backbones per broader design objective;
- a 1,000-backbone LHD101 diversity analysis;
- loop percentage and radius of gyration, retaining the lowest 50th
  percentile and approximately 10% of structures for sequence design;
- Foldseek `easy-cluster` diversity, defined as clusters divided by
  backbones;
- more than 13-fold higher diversity when seed orientation was sampled;
- final ring-interface Flatness and Twist distributions.

The supplement does not give absolute loop/Rg thresholds. It also does not
publish the raw Foldseek assignments. Mosaic therefore computes cohort
percentiles and never invents an absolute cutoff or reference distribution.
As of 2026-08-18, the paper-linked `interface_seeded_oligomers` analysis
repository returned 404; Flatness/Twist are transparently reimplemented from
the written definitions rather than described as byte-identical author code.

## Mosaic outputs collected

For every backbone, the comparison records:

- generation and strict-audit acceptance;
- supplied-seed atom completeness and RMSD;
- exact-symmetry coordinate RMSD;
- chain breaks and CA clashes;
- chain CA radius of gyration;
- length-normalized chain CA radius of gyration;
- intra-chain long-range contacts and generated-residue tertiary-support
  coverage;
- neighbouring-chain CA contacts and minimum distance;
- generated--generated cross-chain contact count, coverage, and the soft
  excess objective;
- Flatness and Twist in a symmetry-aligned local ring frame;
- optional STRIDE coil/turn percentage.

Missing STRIDE does not make a valid Mosaic backbone fail. It leaves only the
paper's loop-percentile column pending. ProteinMPNN, AF2/AF3, Rosetta and wet
experiments are intentionally outside this backbone-only comparison.

## Run the experiment

First CPU-screen initial poses, then submit a small multi-pose pilot. Keeping
one diffusion seed and one initial pose fixed is useful for code debugging,
but it is not evidence that the method is robust. The maintained pilot keeps
the selected pose seeds and independent diffusion seeds explicit:

```bash
python scripts/rfd3_mosaic/submit_mosaic_lhd101_c3_1000.py \
  --mode pilot \
  --seed-start 20000 \
  --pose-seeds 10063 10039 10048 10027 \
  --submit
```

The listed poses come from a 64-pose replayable random screen. They span
different orientations and avoid the seed-10000 pathology where the straight
scaffold corridor passes within 0.07 A of unrelated fixed atoms. Pilot pose
selection is an engineering release gate; the formal 1,000-backbone cohort
retains its declared random-pose protocol and is not cherry-picked from this
list.

Do not scale an older pilot that lacks
`scaffold_core_guidance_audit.json`: it tested exact motif mobility and
scaffold validity only, not the current intra/inter scaffold objective.
Scale-up requires all constraint, mobility, scaffold and core-guidance audits
to pass.

After its required audits pass, submit exactly 1,000 backbones:

```bash
python scripts/rfd3_mosaic/submit_mosaic_lhd101_c3_1000.py \
  --mode full \
  --total-designs 1000 \
  --designs-per-job 10 \
  --submit
```

The launcher prints the campaign directory. When every shard is complete,
generate JSON, CSV and Markdown comparison artifacts:

```bash
python scripts/rfd3_mosaic/compare_hoyeung_lhd101_backbones.py \
  --campaign-manifest "$CAMPAIGN_DIR/campaign_manifest.json" \
  --run-root "$RFD3_MOSAIC_RUN_ROOT" \
  --output-dir "$CAMPAIGN_DIR/comparison"
```

If STRIDE is installed, add `--stride /absolute/path/to/stride` to reproduce
the reported loop-percentage ranking. Foldseek diversity is a separate
external clustering step because raw paper cluster assignments were not
released in the supplement.
