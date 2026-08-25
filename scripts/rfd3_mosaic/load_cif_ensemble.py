"""Load many Mosaic CIF outputs into one PyMOL multi-state object.

Run inside PyMOL, then call for example::

    run /path/to/load_cif_ensemble.py
    load_cif_ensemble /path/to/generated_structures, mosaic_batch

The left/right and PageUp/PageDown keys then move between designs.
"""

from __future__ import annotations

import glob
import shutil
import tempfile
import zipfile
from pathlib import Path

from pymol import cmd

_ACTIVE_ENSEMBLE: str | None = None


def _paths(source: str, recursive: int) -> tuple[Path, ...]:
    root = Path(source).expanduser()
    if root.is_dir():
        iterator = root.rglob("*") if int(recursive) else root.glob("*")
        paths = tuple(
            sorted(
                path.resolve()
                for path in iterator
                if path.is_file()
                and (path.name.endswith(".cif") or path.name.endswith(".cif.gz"))
            )
        )
    else:
        paths = tuple(
            sorted(
                Path(path).resolve()
                for path in glob.glob(str(root), recursive=bool(int(recursive)))
                if path.endswith(".cif") or path.endswith(".cif.gz")
            )
        )
    if not paths:
        raise ValueError(f"No .cif or .cif.gz structures found for {source!r}")
    return paths


def _set_state(state: int) -> None:
    if _ACTIVE_ENSEMBLE is None:
        print("No active CIF ensemble; call load_cif_ensemble first")
        return
    state_count = cmd.count_states(_ACTIVE_ENSEMBLE)
    target = ((int(state) - 1) % state_count) + 1
    cmd.set("state", target)
    cmd.refresh()
    print(f"{_ACTIVE_ENSEMBLE}: state {target}/{state_count}")


def ensemble_next() -> None:
    _set_state(cmd.get_state() + 1)


def ensemble_previous() -> None:
    _set_state(cmd.get_state() - 1)


def load_cif_ensemble(
    source: str,
    object_name: str = "mosaic_ensemble",
    recursive: int = 1,
) -> None:
    """Load a directory or glob as independent states of one discrete object."""

    global _ACTIVE_ENSEMBLE
    if object_name in cmd.get_names("objects"):
        raise ValueError(
            f"PyMOL object {object_name!r} already exists; choose another name "
            "or delete it explicitly before loading"
        )
    source_path = Path(source).expanduser()
    if source_path.is_file() and source_path.suffix == ".zip":
        with (
            zipfile.ZipFile(source_path) as archive,
            tempfile.TemporaryDirectory() as directory,
        ):
            members = tuple(
                name for name in archive.namelist() if name.endswith(".cif")
            )
            if not members:
                raise ValueError(f"No plain .cif members found in {source_path}")
            temporary_root = Path(directory)
            paths = []
            for index, member in enumerate(members, start=1):
                destination = temporary_root / f"{index:05d}_{Path(member).name}"
                with archive.open(member) as input_handle, destination.open(
                    "wb"
                ) as output_handle:
                    shutil.copyfileobj(input_handle, output_handle)
                paths.append(destination)
            for state, path in enumerate(paths, start=1):
                cmd.load(
                    str(path),
                    object_name,
                    state=state,
                    discrete=1,
                    quiet=1,
                )
    else:
        paths = _paths(source, int(recursive))
        for state, path in enumerate(paths, start=1):
            cmd.load(
                str(path),
                object_name,
                state=state,
                discrete=1,
                quiet=1,
            )
    _ACTIVE_ENSEMBLE = object_name
    cmd.set("state", 1)
    cmd.set_key("RIGHT", ensemble_next)
    cmd.set_key("LEFT", ensemble_previous)
    cmd.set_key("PGDN", ensemble_next)
    cmd.set_key("PGUP", ensemble_previous)
    cmd.orient(object_name)
    print(
        f"Loaded {len(paths)} structures into {object_name!r}; "
        "use Left/Right or PageUp/PageDown to switch"
    )


cmd.extend("load_cif_ensemble", load_cif_ensemble)
cmd.extend("ensemble_next", ensemble_next)
cmd.extend("ensemble_previous", ensemble_previous)
