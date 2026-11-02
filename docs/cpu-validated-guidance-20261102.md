# CPU-validated guidance optimizations

This branch extends `perf/validated-fp32-20261102`. It does not include the separate research branch's pose strategies, acceptance-policy changes, terminal repair policies, or finite-group prototypes. It is not the complete previously timed research configuration.

Two behavior-preserving changes are included:

- Outside the scaffold update window, retain energy diagnostics without building an unused autograd graph.
- Cache coordinate-independent boundary exclusion masks and per-chain contact, adjacency, and clash pair indices. Each topology owns one cache entry. Before reuse, compare all dependency tensor values, shapes, dtypes, devices and layouts, plus chain/count/sequence-separation metadata. This catches in-place and alias mutations, including changes that bypass tensor version counters. Failed construction does not publish a new entry. No coordinates, gradients, RNG state or global sample cache are retained.

The value checks deliberately remain conservative. They can synchronize on CUDA. **No GPU speedup or memory improvement has been measured for these changes.** CPU equivalence is not evidence of improved design quality or an end-to-end throughput gain. GPU comparison remains necessary before recommending this cache for performance-sensitive campaigns.

## Validation

On this publication branch, 60 unit tests pass across the cache, inactive-window behavior, scaffold guidance and scaffold core suites. A baseline comparison additionally covers:

- 36 core energy/gradient cases with cold/warm caches, residue/index/mask changes and changed sequence separation;
- 60 sequential core proposal steps, comparing complete diagnostics, coordinates and RNG;
- eight full rigid-controller steps, including committed updates;
- eight boundary energy/gradient cases with changed dependencies;
- one accepted scaffold control, 12 invalid-shape comparisons and five invalid-projector comparisons.

Tests use CPU FP32/FP64 and synthetic fixtures, not model inference. CPU/meta-device cache invalidation is covered; CUDA execution is not. Model parameters, sampling schedule, thresholds and objectives are unchanged.

From a configured project environment:

```sh
PYTHONPATH=src:models/rfd3/src:tests/rfd3_mosaic/unit python -m unittest test_topology_index_cache test_core_inactive_autograd test_scaffold_guidance test_scaffold_core_guidance
PYTHONPATH=src:models/rfd3/src python tests/rfd3_mosaic/regression/check_energy_cache_equivalence.py /tmp/mosaic-cache-regression
```

The second command reads baseline commit `1d997cc711219ed7f151db133a1e85c7b419cb99` from local git objects and writes its JSON results to the supplied directory. It does not fetch code, load checkpoints or use a GPU. It requires the repository's test dependencies.
