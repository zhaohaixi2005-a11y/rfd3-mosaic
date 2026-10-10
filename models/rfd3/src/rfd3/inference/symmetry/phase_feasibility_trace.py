"""Opt-in immutable phase evidence; never changes sampler coordinates or RNG."""

import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch


class PhaseFeasibilityTrace:
    def __init__(self, directory, *, seed, limit=256, max_bytes=64 * 1024 * 1024):
        self.directory = Path(directory) / f"seed-{seed}-{uuid.uuid4().hex}"
        self.seed, self.limit, self.max_bytes = int(seed), limit, max_bytes
        self.count = self.bytes = self.skipped = 0
        self.errors = []

    @classmethod
    def from_environment(cls, *, seed):
        directory = os.environ.get("MOSAIC_FEASIBILITY_TRACE_DIR")
        return cls(directory, seed=seed) if directory else None

    def record(self, stage, coordinates, *, reference, fixed_mask, step=-1, **extra):
        if self.count >= self.limit or self.bytes >= self.max_bytes:
            self.skipped += 1
            return
        try:
            if reference is None:
                return
            cpu = coordinates.detach().contiguous().cpu()
            payload = {
                "schema_version": 1, "seed": self.seed, "stage": stage,
                "step": int(step), "shape": list(cpu.shape),
                "dtype": str(cpu.dtype), "source_device": str(coordinates.device),
                "contract": reference.contract, "extra": extra,
            }
            arrays = {
                "coordinates_storage": cpu.reshape(-1).view(torch.uint8).numpy().copy(),
                "backbone_atom_indices": reference.backbone_atom_indices.detach().cpu().numpy().copy(),
                "fixed_atom_mask": fixed_mask.detach().cpu().numpy().copy(),
                "metadata_json": np.frombuffer(json.dumps(payload, allow_nan=False).encode(), dtype=np.uint8),
            }
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"phase-{self.count:04d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            self.bytes += path.stat().st_size
            self.count += 1
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def diagnostics(self):
        return {"directory": str(self.directory), "seed": self.seed,
                "captured": self.count, "bytes": self.bytes,
                "limit": self.limit, "max_bytes": self.max_bytes,
                "skipped": self.skipped, "errors": list(self.errors),
                "scope": "actual stage coordinates and reference; opt-in observation only"}
