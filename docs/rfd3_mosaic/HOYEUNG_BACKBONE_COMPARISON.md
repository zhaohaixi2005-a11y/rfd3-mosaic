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
- `sampling.scaffold_packing: symmetric_generated` builds the three physical
  C3 neighbour edges over generated residues and jointly optimizes reciprocal
  contiguous patch coverage, orientation, shape, clashes and backbone
  continuity. Generated patches and the complete mobile seed orbit are
  accepted or rolled back as one transaction.

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
- neighbouring-chain CA contacts and minimum distance;
- final generated-patch coverage/continuity/orientation/shape evidence and
  whether every packing target was satisfied;
- Flatness and Twist in a symmetry-aligned local ring frame;
- optional STRIDE coil/turn percentage.

Missing STRIDE does not make a valid Mosaic backbone fail. It leaves only the
paper's loop-percentile column pending. ProteinMPNN, AF2/AF3, Rosetta and wet
experiments are intentionally outside this backbone-only comparison.

## Run the experiment

First submit one pilot:

```bash
python scripts/rfd3_mosaic/submit_mosaic_lhd101_c3_1000.py \
  --mode pilot \
  --submit
```

Do not scale an older pilot that lacks
`graph_interface_guidance_audit.json`: it tested exact motif mobility and
scaffold validity only, not scientific interface packing. Scale-up requires
all constraint, mobility, scaffold and packing audits to pass.

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
