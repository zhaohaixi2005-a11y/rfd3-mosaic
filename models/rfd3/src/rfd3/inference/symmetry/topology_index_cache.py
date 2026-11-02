"""Per-topology, value-validated cache for coordinate-independent indices.

Snapshots check values, not only Tensor._version: aliases and inference
tensors must not reuse stale indices. No coordinates, gradients, random
state or process-global entries are retained.
"""
from typing import Any, Callable

import torch


class TopologyIndexCache:
    def __init__(self) -> None:
        self._entry = None

    def get_or_compute(
        self, tensors: tuple[torch.Tensor, ...], metadata: tuple,
        build: Callable[[], Any],
    ) -> Any:
        entry = self._entry
        if entry is not None:
            old_metadata, snapshots, result = entry
            if old_metadata == metadata and len(snapshots) == len(tensors):
                matches = all(
                    a.shape == b.shape and a.dtype == b.dtype
                    and a.device == b.device and a.layout == b.layout
                    and torch.equal(a, b)
                    for a, b in zip(tensors, snapshots)
                )
                if matches:
                    return result
        # Publish only after construction succeeds. A failed rebuild cannot
        # validate a new dependency set against the previous result.
        result = build()
        snapshots = tuple(t.detach().clone() for t in tensors)
        self._entry = (metadata, snapshots, result)
        return result
