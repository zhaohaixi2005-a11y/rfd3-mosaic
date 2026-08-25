"""Create a user-facing archive containing only generated mmCIF structures."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
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


__all__ = ["create_generated_cif_archive"]
