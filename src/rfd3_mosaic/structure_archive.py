"""Create a user-facing archive containing only generated mmCIF structures."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _structure_for_result(result_json: Path) -> Path:
    stem = result_json.stem
    candidates = (
        result_json.with_name(f"{stem}.cif.gz"),
        result_json.with_name(f"{stem}.cif"),
    )
    existing = tuple(path for path in candidates if path.is_file())
    if len(existing) != 1:
        raise RuntimeError(
            "Expected exactly one generated mmCIF for result metadata "
            f"{result_json}, observed {[str(path) for path in existing]}"
        )
    return existing[0]


def _plain_cif_name(structure: Path) -> str:
    name = structure.name
    if name.endswith(".cif.gz"):
        return name.removesuffix(".gz")
    if name.endswith(".cif"):
        return name
    raise ValueError(f"Generated structure is not an mmCIF file: {structure}")


def materialize_plain_cif(
    structure: str | Path,
    destination_directory: str | Path,
) -> Path:
    """Atomically copy or decompress one generated structure as plain mmCIF.

    RFD3 writes compressed structures directly into the run directory.  The
    temporary destination is deliberately kept beside the final plain CIF so
    readers never observe a partially decompressed structure.  Reading a
    still-growing gzip stream raises before ``replace()`` and is safe to retry.
    """

    source = Path(structure).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_directory = Path(destination_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / _plain_cif_name(source)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        opener = gzip.open if source.name.endswith(".cif.gz") else Path.open
        with opener(source, "rb") as input_handle, temporary.open("wb") as target:
            shutil.copyfileobj(input_handle, target, length=1024 * 1024)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_plain_cif_manifest(directory: Path) -> Path:
    members = sorted(path.name for path in directory.glob("*model_0.cif"))
    manifest = directory / "manifest.json"
    temporary = directory / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "directory/plain_mmcif_members",
                "produced_designs": len(members),
                "members": members,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return manifest


class GeneratedCifMirror:
    """Incrementally mirror completed RFD3 gzip outputs as plain CIF files."""

    def __init__(
        self,
        run_directory: str | Path,
        destination_directory: str | Path,
        *,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.run_directory = Path(run_directory).resolve()
        self.destination_directory = Path(destination_directory).resolve()
        self.poll_interval_seconds = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _candidates(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                {
                    *self.run_directory.glob("*model_0.cif.gz"),
                    *self.run_directory.glob("*model_0.cif"),
                }
            )
        )

    def scan(self, *, tolerate_incomplete_gzip: bool) -> tuple[Path, ...]:
        produced: list[Path] = []
        changed = False
        for source in self._candidates():
            destination = self.destination_directory / _plain_cif_name(source)
            if destination.is_file():
                produced.append(destination)
                continue
            try:
                destination = materialize_plain_cif(
                    source,
                    self.destination_directory,
                )
            except (EOFError, gzip.BadGzipFile, OSError):
                if tolerate_incomplete_gzip:
                    continue
                raise
            produced.append(destination)
            changed = True
        if changed or not (self.destination_directory / "manifest.json").is_file():
            _write_plain_cif_manifest(self.destination_directory)
        return tuple(produced)

    def _watch(self) -> None:
        try:
            while not self._stop.wait(self.poll_interval_seconds):
                self.scan(tolerate_incomplete_gzip=True)
        except BaseException as error:  # Propagate background I/O errors on stop.
            self._error = error
            self._stop.set()

    def start(self) -> "GeneratedCifMirror":
        if self._thread is not None:
            raise RuntimeError("Generated CIF mirror is already running")
        self.destination_directory.mkdir(parents=True, exist_ok=True)
        _write_plain_cif_manifest(self.destination_directory)
        self._thread = threading.Thread(
            target=self._watch,
            name="rfd3-mosaic-plain-cif-mirror",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, inference_succeeded: bool) -> tuple[Path, ...]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise RuntimeError("Generated CIF mirror failed") from self._error
        return self.scan(tolerate_incomplete_gzip=not inference_succeeded)

    def __enter__(self) -> "GeneratedCifMirror":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.stop(inference_succeeded=exc_type is None)


def create_generated_cif_archive(
    result_jsons: Iterable[str | Path],
    destination: str | Path,
    *,
    requested_designs: int,
) -> dict[str, object]:
    """Write one ZIP whose members are plain CIF files and nothing else."""

    results = tuple(Path(path).resolve() for path in result_jsons)
    if requested_designs < 1:
        raise ValueError("requested_designs must be positive")
    structures = tuple(_structure_for_result(path) for path in results)
    member_names = tuple(f"{path.stem.removesuffix('.cif')}.cif" for path in structures)
    if len(set(member_names)) != len(member_names):
        raise RuntimeError("Generated structure archive member names are not unique")

    output = Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for structure, member_name in zip(
                structures,
                member_names,
                strict=True,
            ):
                opener = gzip.open if structure.suffix == ".gz" else Path.open
                with opener(structure, "rb") as source, archive.open(
                    member_name,
                    mode="w",
                ) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "archive": str(output),
        "sha256": _sha256(output),
        "requested_designs": requested_designs,
        "produced_designs": len(structures),
        "complete": len(structures) == requested_designs,
        "format": "zip/plain_mmcif_members",
        "member_count": len(member_names),
        "members": list(member_names),
    }
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


__all__ = [
    "GeneratedCifMirror",
    "create_generated_cif_archive",
    "materialize_plain_cif",
]
