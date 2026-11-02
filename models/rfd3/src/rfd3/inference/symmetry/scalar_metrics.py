"""Transfer same-dtype scalar diagnostics together without changing their values."""
from collections.abc import Mapping

import torch


def detached_scalar_metrics(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    if not values:
        return {}
    tensors = list(values.values())
    first = tensors[0]
    # Preserve the legacy conversion for heterogeneous tensors; a promotion
    # could otherwise round an integer or lose precision before float().
    if any(t.dtype != first.dtype or t.device != first.device or t.numel() != 1
           for t in tensors):
        return {key: float(value.detach().cpu().item()) for key, value in values.items()}
    packed = torch.stack([value.detach().reshape(()) for value in tensors])
    return {key: float(value) for key, value in zip(values, packed.cpu().tolist())}
