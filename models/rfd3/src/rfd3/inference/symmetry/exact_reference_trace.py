"""Bounded opt-in evidence for rejected reference-transport guards.

NPZ contains ordinary numeric arrays plus UTF-8 JSON, never pickled objects.
The captured scope is the exact guard inputs/outputs, not a full model replay.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .generated_routes import route_nonregression_check
from .reference_scaffold import (
    ReferenceScaffold,
    reference_scaffold_guidance_deficits,
)

MAX_SNAPSHOTS = 3
MAX_TOTAL_BYTES = 100 * 1024 * 1024
_TENSOR_FIELDS = (
    "ca_atom_indices", "reference_ca", "fixed_ca",
    "backbone_atom_indices", "reference_backbone",
)
_DTYPES = {str(dtype): dtype for dtype in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class ExactReferenceTrace:
    """At most three atomic snapshots / 100 MiB across one trace directory."""

    def __init__(self, directory: Path, *, limit=MAX_SNAPSHOTS, max_bytes=MAX_TOTAL_BYTES):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SNAPSHOTS:
            raise ValueError("Exact trace limit must be an integer in [1, 3]")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_TOTAL_BYTES:
            raise ValueError("Exact trace byte budget must be in [1, 100 MiB]")
        self.directory = Path(directory)
        self.limit, self.max_bytes = limit, max_bytes
        self.rejections_seen = self.captured = self.skipped = self.failures = 0
        self.skipped_after_failure = 0
        self.exhausted = self.failed_permanently = False
        self.last_error = None
        self.descriptors: list[dict[str, Any]] = []
        self.manifest_sha256 = None
        self.configuration_error = None

    @classmethod
    def from_environment(cls):
        path = os.environ.get("MOSAIC_TRANSPORT_TRACE_DIR")
        if not path:
            return None
        recorder = cls(Path(path))
        try:
            raw = os.environ.get("MOSAIC_TRANSPORT_TRACE_LIMIT", str(MAX_SNAPSHOTS))
            limit = int(raw)
            if not 1 <= limit <= MAX_SNAPSHOTS:
                raise ValueError("MOSAIC_TRANSPORT_TRACE_LIMIT must be in [1, 3]")
            recorder.limit = limit
        except (ValueError, TypeError) as error:
            # Diagnostic configuration cannot change the original acceptance.
            recorder.configuration_error = str(error)
        return recorder

    def diagnostics(self):
        return {
            "enabled": True,
            "directory": str(self.directory),
            "scope": "exact_reference_guard_states_not_full_model_replay",
            "limit_per_directory": self.limit,
            "max_total_bytes": self.max_bytes,
            "rejections_seen": self.rejections_seen,
            "captured": self.captured,
            "skipped_due_to_limit": self.skipped,
            "skipped_after_capture_failure": self.skipped_after_failure,
            "capture_exhausted": self.exhausted,
            "capture_failures": self.failures,
            "last_error": self.last_error,
            "configuration_error": self.configuration_error,
            "snapshots": list(self.descriptors),
            "manifest_sha256_after_last_capture": self.manifest_sha256,
        }

    def record_rejection(self, *, coordinates_before, coordinates_after,
                         residual_before, residual_after, reference_before,
                         reference_after, guard, geometry_tolerance, plan_sha256):
        """Best-effort evidence only; never alter the caller's rejection."""
        self.rejections_seen += 1
        if self.exhausted or self.captured >= self.limit:
            self.skipped += 1
            return
        if self.failed_permanently:
            self.skipped_after_failure += 1
            return
        try:
            if self.configuration_error:
                raise ValueError(self.configuration_error)
            if guard.get("passed") is not False:
                raise ValueError("Only an actual failed guard may be traced")
            # A later seed/process must not download tensors and compress
            # hundreds of candidates after an earlier seed filled this case.
            if self.directory.exists():
                if not self.directory.is_dir():
                    raise NotADirectoryError(str(self.directory))
                paths = list(self.directory.glob("capture-*.npz"))
                if len(paths) >= self.limit or sum(p.stat().st_size for p in paths) >= self.max_bytes:
                    self.exhausted = True
                    self.skipped += 1
                    return
            arrays = {}
            descriptors = {}

            def tensor(value, name):
                if value is None:
                    return None
                if not isinstance(value, torch.Tensor) or str(value.dtype) not in _DTYPES:
                    raise ValueError(f"Unsupported exact trace tensor {name}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite exact trace tensor {name}")
                cpu = value.detach().contiguous().cpu()
                # Raw numeric storage also preserves bfloat16 without pickle.
                array = cpu.reshape(-1).view(torch.uint8).numpy().copy()
                arrays[name] = array
                descriptors[name] = {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "device": str(value.device), "num_bytes": int(array.nbytes),
                }
                return name

            def reference(value, prefix):
                return {
                    "contract": value.contract,
                    # Contracts retain pre-cast Python-double reference data;
                    # these separate tensors retain what the guard actually read.
                    "tensors": {key: tensor(getattr(value, key), prefix + "." + key)
                                for key in _TENSOR_FIELDS},
                    "chains": [tensor(ids, f"{prefix}.chain.{i}") for i, ids in enumerate(value.chains)],
                    "blocks": {key: tensor(ids, f"{prefix}.block.{i}")
                               for i, (key, ids) in enumerate(sorted(value.blocks.items()))},
                    "reference_separations": [[i, j, distance] for (i, j), distance in sorted(value.reference_separations.items())],
                }

            metadata = {
                "schema_version": 1,
                "scope": "exact_reference_guard_states_not_full_model_replay",
                "rejection_index_in_transport": self.rejections_seen,
                "process_id": os.getpid(), "torch_version": torch.__version__,
                "plan_sha256": plan_sha256,
                "geometry_tolerance": float(geometry_tolerance),
                "guard": guard,
                "state_tensors": {name: tensor(value, name) for name, value in (
                    ("coordinates_before", coordinates_before), ("coordinates_after", coordinates_after),
                    ("residual_before", residual_before), ("residual_after", residual_after),
                )},
                "references": {"before": reference(reference_before, "before_reference"),
                               "after": reference(reference_after, "after_reference")},
                "tensor_descriptors": descriptors,
            }
            encoded = _json_bytes(metadata)
            # Check uncompressed evidence size before allocating compression.
            if sum(a.nbytes for a in arrays.values()) + len(encoded) > self.max_bytes:
                self.exhausted = True
                self.skipped += 1
                return
            arrays["metadata_json"] = np.frombuffer(encoded, dtype=np.uint8)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            payload = buffer.getvalue()
            self._write(payload)
        except Exception as error:
            self.failed_permanently = True
            self.failures += 1
            self.last_error = f"{type(error).__name__}: {error}"[:1000]

    def _write(self, payload):
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        # Multiple native workers may share one case directory. The budget
        # and fixed numbered slots are guarded across processes.
        with (self.directory / ".capture.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            paths = sorted(self.directory.glob("capture-*.npz"))
            if len(paths) >= self.limit or sum(p.stat().st_size for p in paths) + len(payload) > self.max_bytes:
                self.exhausted = True
                self.skipped += 1
                return
            manifest_path = self.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
                "schema_version": 1, "scope": "exact_reference_guard_states_not_full_model_replay",
                "limit_per_directory": self.limit, "max_total_bytes": self.max_bytes,
                "snapshots": [],
            }
            used = {p.name for p in paths}
            name = next(f"capture-{i:03d}.npz" for i in range(self.limit) if f"capture-{i:03d}.npz" not in used)
            destination = self.directory / name
            pending = self.directory / (name + ".pending")
            # No random filename generation and no model RNG consumption.
            with pending.open("xb") as handle:
                handle.write(payload)
            os.replace(pending, destination)
            descriptor = {"file": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            manifest["snapshots"].append(descriptor)
            manifest["total_snapshot_bytes"] = sum(p.stat().st_size for p in self.directory.glob("capture-*.npz"))
            encoded = _json_bytes(manifest)
            temporary = self.directory / "manifest.json.pending"
            temporary.write_bytes(encoded)
            os.replace(temporary, manifest_path)
            self.manifest_sha256 = hashlib.sha256(encoded).hexdigest()
            self.descriptors.append(descriptor)
            self.captured += 1


def load_reference_trace(path: str | Path):
    """Load only numeric NPZ arrays and explicit JSON; reconstruct on CPU."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported exact reference trace schema")

        def tensor(name):
            if name is None:
                return None
            info = metadata["tensor_descriptors"][name]
            array = archive[name]
            if array.dtype != np.uint8 or array.nbytes != info["num_bytes"]:
                raise ValueError("Exact trace numeric storage mismatch")
            return torch.from_numpy(array.copy()).view(_DTYPES[info["dtype"]]).reshape(info["shape"])

        def reference(record):
            return ReferenceScaffold(
                contract=record["contract"],
                **{key: tensor(name) for key, name in record["tensors"].items()},
                chains=tuple(tensor(name) for name in record["chains"]),
                blocks={key: tensor(name) for key, name in record["blocks"].items()},
                reference_separations={(int(i), int(j)): value for i, j, value in record["reference_separations"]},
            )

        return {
            "metadata": metadata,
            **{key: tensor(name) for key, name in metadata["state_tensors"].items()},
            "reference_before": reference(metadata["references"]["before"]),
            "reference_after": reference(metadata["references"]["after"]),
        }


def replay_reference_trace(path: str | Path, *, dtype=None):
    """Reevaluate identical stored coordinates/references at chosen precision.

    Original-device versus CPU reduction differences remain measurable; this
    does not replace tensor references with the pre-cast contract coordinates.
    """
    captured = load_reference_trace(path)
    before_xyz = captured["coordinates_before"]
    after_xyz = captured["coordinates_after"]
    stored_dtype = before_xyz.dtype
    evaluation_dtype = captured["residual_before"].dtype if dtype is None else dtype
    before_xyz, after_xyz = before_xyz.to(evaluation_dtype), after_xyz.to(evaluation_dtype)
    with torch.no_grad():
        before = reference_scaffold_guidance_deficits(before_xyz, captured["reference_before"])
        after = reference_scaffold_guidance_deficits(after_xyz, captured["reference_after"])
    guard = route_nonregression_check(before, after, tolerance=captured["metadata"]["geometry_tolerance"])
    if "native_stored_dtype_guard" in captured["metadata"]["guard"]:
        with torch.no_grad():
            native = route_nonregression_check(
                reference_scaffold_guidance_deficits(captured["coordinates_before"].to(stored_dtype), captured["reference_before"]),
                reference_scaffold_guidance_deficits(captured["coordinates_after"].to(stored_dtype), captured["reference_after"]),
                tolerance=captured["metadata"]["geometry_tolerance"],
            )
        guard["native_stored_dtype_guard"] = native
        guard["passed"] = guard["passed"] and native["passed"]
    return {
        "saved_guard": captured["metadata"]["guard"],
        "replayed_guard": guard,
        "evaluation_dtype": str(before_xyz.dtype), "evaluation_device": "cpu",
        "maximum_before_residual_difference": float((before - captured["residual_before"].to(before)).abs().max()),
        "maximum_after_residual_difference": float((after - captured["residual_after"].to(after)).abs().max()),
    }
