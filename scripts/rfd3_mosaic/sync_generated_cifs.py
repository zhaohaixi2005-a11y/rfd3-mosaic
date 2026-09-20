#!/usr/bin/env python3
"""Pull verified final CIFs over SSH; Python standard library only, no inference."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import tempfile
import zlib

OWNER = "rfd3-mosaic-final-cif-sync-v1"
STATE_NAME = ".mosaic_sync.json"
CHUNK = 1024 * 1024

# Executed read-only on the remote host. JSON sidecars identify final outputs;
# trajectory filenames alone must never qualify as completed designs.
REMOTE = r'''
import gzip, hashlib, json, pathlib, sys, tarfile, time
root = pathlib.Path(sys.argv[2]).resolve(strict=True)
if not root.is_dir():
    raise ValueError("Run is not a directory")
def source(name):
    path = root / name
    if pathlib.Path(name).name != name or path.resolve().parent != root:
        raise ValueError("Source escapes run directory")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing source: " + str(path))
    return path
def digest(handle):
    size, sha = 0, hashlib.sha256()
    for block in iter(lambda: handle.read(1048576), b""):
        size += len(block)
        sha.update(block)
    return {"size": size, "sha256": sha.hexdigest()}
if sys.argv[1] == "inventory":
    files, pending = [], []
    for result in sorted(root.glob("*_model_0.json")):
        if result.name.endswith(("_noisy_model_0.json", "_denoised_model_0.json")):
            continue
        for attempt in range(3):
            try:
                metadata = json.loads(source(result.name).read_text())
                if not isinstance(metadata, dict):
                    raise ValueError("Result metadata is not a JSON object")
                break
            except (ValueError, UnicodeError) as error:
                if attempt == 2:
                    pending.append({"result_json": result.name, "reason": str(error)})
                else:
                    time.sleep(0.2)
        else:
            continue
        name = result.stem + ".cif"
        path = source(name + ".gz" if (root / (name + ".gz")).is_file() else name)
        before = path.stat()
        with path.open("rb") as handle:
            raw = digest(handle)
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                plain = digest(handle)  # includes gzip EOF/CRC verification
        else:
            plain = raw
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed during inventory: " + str(path))
        if not plain["size"]:
            raise ValueError("Empty CIF: " + str(path))
        files.append({"source": path.name, "name": name, "raw": raw, "plain": plain})
    print(json.dumps({"remote_run_dir": str(root), "files": files, "pending": pending}))
elif sys.argv[1] == "transfer":
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name in json.loads(sys.argv[3]):
            archive.add(source(name), arcname=name, recursive=False)
else:
    raise ValueError("Unknown operation")
'''


def digest_stream(handle, target=None):
    size, sha = 0, hashlib.sha256()
    for block in iter(lambda: handle.read(CHUNK), b""):
        size += len(block)
        sha.update(block)
        if target is not None:
            target.write(block)
    return {"size": size, "sha256": sha.hexdigest()}


def digest_file(path):
    with path.open("rb") as handle:
        return digest_stream(handle)


def ssh_command(args, operation, run_dir, names=None):
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if args.socket:
        command.extend(["-S", args.socket])
    for option in args.ssh_option:
        command.extend(["-o", option])
    remote = [args.remote_python, "-c", REMOTE, operation, run_dir]
    if names is not None:
        remote.append(json.dumps(names))
    return command + [args.host, shlex.join(remote)]


def invoke(args, operation, run_dir, *, names=None, stdout=subprocess.PIPE):
    process = subprocess.run(
        ssh_command(args, operation, run_dir, names),
        stdout=stdout, stderr=subprocess.PIPE, timeout=args.timeout, check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
    return process.stdout


def output_directory(root, relative):
    parts = PurePosixPath(relative)
    if not relative or parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError("destination must be a nonempty relative path without '..'")
    root.mkdir(parents=True, exist_ok=True)
    destination = root.resolve()
    for part in (*parts.parts, "generated_structures_cif"):
        destination /= part
        if destination.is_symlink():
            raise ValueError("Refusing symlink destination: " + str(destination))
        destination.mkdir(exist_ok=True)
    return destination


def write_state(directory, state):
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".sync-state-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, directory / STATE_NAME)
        finally:
            temporary.unlink(missing_ok=True)


def sync_run(args, run):
    remote_dir = run["remote_run_dir"]
    if not isinstance(remote_dir, str) or not remote_dir.startswith("/"):
        raise ValueError("remote_run_dir must be an absolute remote path")
    inventory = json.loads(invoke(args, "inventory", remote_dir))
    directory = output_directory(args.destination_root, run["destination"])
    lock = directory / ".mosaic-sync.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Sync lock exists; check for an active process before removing " + str(lock))
    os.close(descriptor)
    try:
        return sync_inventory(args, inventory, directory)
    finally:
        lock.unlink()


def sync_inventory(args, inventory, directory):
    state_path = directory / STATE_NAME
    identity = {"owner": OWNER, "host": args.host, "remote_run_dir": inventory["remote_run_dir"]}
    state = {**identity, "files": {}}
    if state_path.is_symlink():
        raise ValueError("Refusing symlink state file")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("Destination already belongs to a different run or unknown manifest")
    state["pending"] = inventory.get("pending", [])
    required, seen, original = [], set(), {}
    for entry in inventory["files"]:
        name, source = entry["name"], entry["source"]
        if (
            Path(name).name != name or not name.endswith("_model_0.cif")
            or name.endswith(("_noisy_model_0.cif", "_denoised_model_0.cif"))
            or source not in (name, name + ".gz") or name in seen
        ):
            raise ValueError("Invalid or duplicate inventory member: " + name)
        seen.add(name)
        target = directory / name
        if target.is_symlink():
            raise ValueError("Refusing symlink target: " + str(target))
        original[name] = None
        if target.exists():
            observed = digest_file(target)
            original[name] = observed
            if observed == entry["plain"]:
                state["files"][name] = entry  # identical existing files can be adopted
                continue
            previous = state["files"].get(name)
            if previous is None or observed != previous["plain"]:
                raise ValueError("Refusing to overwrite unrecognized or locally modified file: " + str(target))
        required.append(entry)
    if required:
        with tempfile.TemporaryDirectory(prefix=".mosaic-sync-", dir=directory) as temporary:
            staging = Path(temporary)
            bundle = staging / "transfer.tar"
            with bundle.open("wb") as handle:
                invoke(args, "transfer", inventory["remote_run_dir"],
                       names=[entry["source"] for entry in required], stdout=handle)
            with tarfile.open(bundle, "r:") as archive:
                members = archive.getmembers()
                expected = {entry["source"] for entry in required}
                if (len(members) != len(expected) or {item.name for item in members} != expected
                        or any(not item.isfile() for item in members)):
                    raise ValueError("Archive members do not match requested regular files")
                for entry in required:
                    raw = staging / "source.bin"
                    with archive.extractfile(entry["source"]) as source, raw.open("wb") as target:
                        observed = digest_stream(source, target)
                    if observed != entry["raw"]:
                        raise ValueError("Compressed/source hash or size mismatch: " + entry["source"])
                    plain = staging / "output.cif"
                    opener = gzip.open if entry["source"].endswith(".gz") else open
                    with opener(raw, "rb") as source, plain.open("wb") as target:
                        observed = digest_stream(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                    if observed != entry["plain"]:
                        raise ValueError("Plain CIF hash or size mismatch: " + entry["name"])
                    destination = directory / entry["name"]
                    if destination.is_symlink() or (
                        (digest_file(destination) if destination.exists() else None)
                        != original[entry["name"]]
                    ):
                        raise ValueError("Destination changed during transfer: " + str(destination))
                    os.replace(plain, destination)
                    state["files"][entry["name"]] = entry
                    write_state(directory, state)
    write_state(directory, state)
    return {"remote_run_dir": inventory["remote_run_dir"], "destination": str(directory),
            "final_files": len(seen), "downloaded": len(required), "skipped": len(seen) - len(required),
            "pending": state["pending"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help='JSON: {"runs": [{"remote_run_dir": "/run", "destination": "task/job"}]}')
    source.add_argument("--run-dir", help="One absolute remote run directory")
    parser.add_argument("--relative-destination", help="Required with --run-dir; task/job under destination root")
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--socket", help="Existing SSH control socket")
    parser.add_argument("--ssh-option", action="append", default=[], help="Repeat SSH -o options, e.g. UserKnownHostsFile=/path")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--timeout", type=float, default=600, help="Timeout per SSH invocation, seconds")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=1,
                        help="Independent runs to sync concurrently (1-4, default 1)")
    args = parser.parse_args(argv)
    if args.host.startswith("-") or not args.host or args.timeout <= 0:
        parser.error("Invalid host or timeout")
    if bool(args.run_dir) != bool(args.relative_destination):
        parser.error("--relative-destination is required only with --run-dir")
    try:
        runs = (json.loads(args.manifest.read_text())["runs"] if args.manifest else
                [{"remote_run_dir": args.run_dir, "destination": args.relative_destination}])
        if not isinstance(runs, list) or not runs:
            raise ValueError("Manifest runs must be a nonempty list")
        destinations = [str(PurePosixPath(run["destination"])) for run in runs]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Manifest contains duplicate destination directories")
        incomplete = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(sync_run, args, run): run for run in runs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    incomplete += bool(result.get("pending"))
                except Exception as error:
                    result = {**futures[future], "error": str(error), "status": "failed"}
                    incomplete += 1
                print(json.dumps(result, ensure_ascii=False), flush=True)
        if incomplete:
            parser.exit(1, f"Sync incomplete: {incomplete} run(s) failed or have pending metadata; rerun to retry.\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError,
            tarfile.TarError, subprocess.TimeoutExpired, zlib.error) as error:
        parser.exit(1, "Sync failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
