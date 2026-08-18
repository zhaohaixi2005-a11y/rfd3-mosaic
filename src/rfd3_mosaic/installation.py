"""Installation layout and packaged resource discovery for RFD3-Mosaic."""

from __future__ import annotations

from importlib import metadata, resources
from pathlib import Path


def source_repository_root(start: Path | None = None) -> Path | None:
    """Return the source checkout root, or ``None`` for a wheel install."""

    origins = [start.resolve()] if start is not None else []
    origins.extend((Path.cwd().resolve(), Path(__file__).resolve()))
    visited: set[Path] = set()
    for origin in origins:
        candidates = (origin, *origin.parents)
        for candidate in candidates:
            if candidate in visited:
                continue
            visited.add(candidate)
            if (candidate / ".project-root").is_file():
                return candidate
    return None


def installation_root() -> Path:
    """Return a stable root for provenance in either installation layout."""

    repository = source_repository_root()
    if repository is not None:
        return repository
    return Path(__file__).resolve().parent


def bundled_resource_path(relative: str | Path) -> Path:
    """Resolve a packaged Mosaic resource as a normal filesystem path."""

    normalized = Path(relative)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Invalid bundled resource path: {relative!r}")
    repository = source_repository_root()
    if repository is not None:
        source_candidate = repository / normalized
        if source_candidate.is_file():
            return source_candidate.resolve()
    packaged = resources.files("rfd3_mosaic").joinpath("resources", *normalized.parts)
    candidate = Path(str(packaged))
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Bundled RFD3-Mosaic resource is missing: {normalized}"
        )
    return candidate.resolve()


def distribution_version() -> str:
    """Return the installed fork version without requiring a Git checkout."""

    for distribution in ("rfd3-mosaic", "rc-foundry"):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return "0+unknown"
