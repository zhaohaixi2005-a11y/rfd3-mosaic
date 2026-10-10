"""Optional exact interface-stage evidence; reads tensors without changing state."""
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch


class InterfaceObservationTrace:
    def __init__(self, directory, *, seed, limit=1024, max_bytes=64 * 1024 * 1024):
        self.directory = Path(directory) / f"seed-{seed}-{uuid.uuid4().hex}"
        self.seed, self.limit, self.max_bytes = int(seed), limit, max_bytes
        self.count = self.bytes = self.skipped = 0
        self.errors = []

    @classmethod
    def from_environment(cls, *, seed, features):
        directory = os.environ.get("MOSAIC_INTERFACE_TRACE_DIR")
        if not directory:
            return None
        trace = cls(directory, seed=seed)
        trace.save_features(features)
        return trace

    def save_features(self, features):
        try:
            arrays, values = {}, {}
            names = {"atom_to_token_map", "asym_id", "entity_id", "residue_index", "restype",
                     "ref_atom_name_chars", "ref_element", "motif_pos", "is_ca", "is_virtual",
                     "is_protein", "is_backbone", "is_motif_atom_with_fixed_coord",
                     "is_motif_atom_with_fixed_seq", "token_bonds",
                     "cylindrical_reference", "cylindrical_keep_mask", "cylindrical_axis", "is_sym_asu"}
            for name, value in features.items():
                if name not in names and not name.startswith(("assembly_interface_", "sym", "motif_constraint_")):
                    continue
                if isinstance(value, torch.Tensor):
                    cpu = value.detach().contiguous().cpu()
                    arrays[name] = cpu.float().numpy().copy() if cpu.dtype == torch.bfloat16 else cpu.numpy().copy()
                else:
                    values[name] = value
            arrays["metadata_json"] = np.frombuffer(json.dumps(values, allow_nan=False, default=lambda value: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value.tolist() if isinstance(value, np.ndarray) else (_ for _ in ()).throw(TypeError(type(value).__name__))).encode(), dtype=np.uint8)
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / "features.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            self.bytes += (self.directory / "features.npz").stat().st_size
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def record(self, stage, coordinates, *, step=-1, gradient=None, **extra):
        if self.count >= self.limit or self.bytes >= self.max_bytes:
            self.skipped += 1
            return
        try:
            cpu = coordinates.detach().contiguous().cpu()
            payload = {"schema_version": 1, "seed": self.seed, "stage": stage,
                       "step": int(step), "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                       "source_device": str(coordinates.device), "extra": extra}
            arrays = {"coordinates_storage": cpu.reshape(-1).view(torch.uint8).numpy().copy(),
                      "metadata_json": np.frombuffer(json.dumps(payload, allow_nan=False).encode(), dtype=np.uint8)}
            if gradient is not None:
                g = gradient.detach().contiguous().cpu()
                arrays["gradient"] = g.float().numpy().copy()
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"interface-{self.count:04d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            self.bytes += path.stat().st_size
            self.count += 1
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def diagnostics(self):
        return {"directory": str(self.directory), "seed": self.seed, "captured": self.count,
                "bytes": self.bytes, "limit": self.limit, "max_bytes": self.max_bytes,
                "skipped": self.skipped, "errors": list(self.errors),
                "scope": "actual clean, noisy and projected trial coordinates; optional actual graph gradients; observation only"}
