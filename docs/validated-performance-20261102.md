# Validated performance primitives, 2026-11-02

This branch preserves three narrowly scoped performance changes on top of the
published `refactor/product-core-v1` commit
`36c287c4917392786a5facc18c089985849b711a`. It does not change the default branch,
scientific objectives, sampling steps, precision, acceptance thresholds, or repair
policy. It is an incremental performance release, not a declaration that Mosaic
V2 or all finite-group design workflows are complete.

## Included changes

1. Batch sparse fixed-position pair embeddings in at most 128 query rows at a
   time instead of launching the same learned embedding separately per atom.
   Both motif and reference positions are read on every call, including after a
   motif moves. Invalid rows retain the previous skip behavior. Floating-point
   operation grouping can change last-bit rounding.
2. Read discrete adjacent-pair and generated-token metadata once per scaffold
   proposal. Preserve the original sequential update order and all three sweeps.
3. Transfer homogeneous scalar diagnostics to the host together. Preserve order,
   NaN/Inf/signed-zero values and use the legacy conversion for heterogeneous
   tensors. This does not remove checks or alter acceptance decisions.

## Validation on this publication checkout

- CPU sparse pair regression: 56 cases x 3 phases (initial, moving coordinates,
  next sample), cached and uncached, empty/full/partial/disjoint masks, batch sizes
  1 and 2, atom counts from 0 to 257 and query-chunk boundaries 127/128/129.
  Maximum absolute difference: `9.5367431640625e-7`; comparison tolerance
  `rtol=2e-5, atol=2e-5`. Input tensors remain unchanged.
- Full CPU scaffold proposals: 16 comparisons have bitwise-equal coordinates and
  identical diagnostics; another 4 inputs produce the same validation errors.
- Scalar conversion tests: 2 passed, covering float16, bfloat16, float32, float64,
  empty/heterogeneous inputs, signed zero, NaN/Inf and preserved gradients.

These components were previously exercised in the isolated GPU performance
investigation. This public-base integration was rechecked on CPU; no new GPU
experiment was run for this branch. Earlier end-to-end speedups from a larger
experimental stack are not a measured speedup of this exact public checkout.
Neither numerical regression nor speed establishes foldability or design success.

## Reproduce

Use an existing compatible RFD3 environment. No checkpoint is needed for these
CPU checks. Set `PYTHONPATH=src:models/rfd3/src` and `TYPE_CHECK=false`.

```sh
python tests/rfd3_mosaic/performance/validate_operator.py --device cpu --output /tmp/mosaic-pair-cpu.json
python -m unittest discover -s tests/rfd3_mosaic/unit -p test_scalar_metrics.py
reference_file=$(mktemp)
git show 36c287c4917392786a5facc18c089985849b711a:models/rfd3/src/rfd3/inference/symmetry/scaffold_core_guidance.py > "$reference_file"
python tests/rfd3_mosaic/performance/validate_scaffold_cpu.py "$reference_file"
```

The frozen pair implementation in `tests/rfd3_mosaic/performance` is exactly the
same file as the published baseline, retained solely as a numerical oracle.

## Deliberately separate development

Native-report/optional-repair policies, direct reference-pose entry, diagnostic
backward removal, geometry-first trial rejection, and experimental motion/gate/
peptide-guard changes remain in the separate integration work pending migration
and regression against the public base. New T/O/I and geometry-conditioned pose
work is experimental and is not included here. No private structures, benchmark
logs, cluster configuration, credentials or model checkpoints are part of this
change. Existing branches and historical versions remain available.
