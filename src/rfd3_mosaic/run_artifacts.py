"""Resolve frozen artifact identities after a complete run is relocated."""

from pathlib import Path
from typing import Any, Mapping


def resolve_run_artifact(
    run_directory: Path,
    value: str | Path,
    summary: Mapping[str, Any],
) -> Path:
    """Preserve the path within the run, never read a surviving original copy.

    Worker result JSONs live directly under the run root. They therefore bind
    the old root even when the summary itself has no run_directory field.
    """

    root = run_directory.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Run artifact escapes its run directory: {path}")
        return resolved
    path = path.resolve()
    if path.is_relative_to(root):
        return path

    result_paths = [summary.get("result_json"), *(summary.get("result_jsons") or [])]
    result_paths.extend(
        record.get("result_json")
        for record in summary.get("design_results", [])
        if isinstance(record, dict)
    )
    old_roots = {
        Path(str(result)).parent
        for result in result_paths
        if result and Path(str(result)).is_absolute()
    }
    if summary.get("run_directory"):
        old_roots.add(Path(str(summary["run_directory"])))
    candidates = {
        root / path.relative_to(old_root)
        for old_root in old_roots
        if old_root.is_absolute() and path.is_relative_to(old_root)
    }
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        raise ValueError(f"Ambiguous recorded run roots for artifact: {path}")

    # Older summaries may lack result identities. Preserve known layout
    # suffixes before considering a unique legacy basename.
    for marker in ("audits", "input", "adapter"):
        if marker in path.parts:
            relative = Path(*path.parts[path.parts.index(marker):])
            return root / relative
    matches = [
        candidate for candidate in root.rglob(path.name)
        if candidate.is_file() and "software" not in candidate.relative_to(root).parts
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous relocated run artifact: {path}")
    return matches[0] if matches else root / path.name
