"""Local protocol tests; never connect to SSH or import the inference stack."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/rfd3_mosaic/sync_generated_cifs.py"
SPEC = importlib.util.spec_from_file_location("sync_generated_cifs", SCRIPT)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class SyncGeneratedCifsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote run"
        self.remote.mkdir()
        self.args = argparse.Namespace(
            host="test-host", socket=None, ssh_option=[], remote_python="python3",
            timeout=30, destination_root=self.root / "local",
        )
        self.run = {"remote_run_dir": str(self.remote), "destination": "task/job"}
        self.destination = self.args.destination_root / "task/job/generated_structures_cif"
        self.calls = []

        def local_command(args, operation, run_dir, names=None):
            self.calls.append(operation)
            command = [sys.executable, "-c", sync.REMOTE, operation, run_dir]
            return command + ([json.dumps(names)] if names is not None else [])

        override = patch.object(sync, "ssh_command", side_effect=local_command)
        override.start()
        self.addCleanup(override.stop)

    def make_result(self, name="design_model_0", content=b"data_design\n", compressed=True):
        (self.remote / (name + ".json")).write_text("{}")
        source = self.remote / (name + (".cif.gz" if compressed else ".cif"))
        source.write_bytes(gzip.compress(content) if compressed else content)
        return source

    def test_batch_transfers_final_sidecar_members_and_skips_verified_files(self):
        self.make_result()
        self.make_result("second_model_0", b"data_second\n", compressed=False)
        self.make_result("design_denoised_model_0", b"trajectory")
        self.make_result("design_noisy_model_0", b"trajectory")
        (self.remote / "unregistered_model_0.cif").write_bytes(b"no metadata")
        result = sync.sync_run(self.args, self.run)
        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(self.calls, ["inventory", "transfer"])
        self.assertEqual(sorted(path.name for path in self.destination.glob("*.cif")),
                         ["design_model_0.cif", "second_model_0.cif"])
        self.assertEqual((self.destination / "design_model_0.cif").read_bytes(), b"data_design\n")
        self.calls.clear()
        self.assertEqual(sync.sync_run(self.args, self.run)["skipped"], 2)
        self.assertEqual(self.calls, ["inventory"])

    def test_interrupted_transfer_preserves_formal_file_and_cleans_staging(self):
        self.make_result()
        sync.sync_run(self.args, self.run)
        self.make_result(content=b"data_updated\n")
        original_invoke = sync.invoke

        def interrupt(args, operation, run_dir, **kwargs):
            if operation == "transfer":
                kwargs["stdout"].write(b"truncated archive")
                raise RuntimeError("Connection lost")
            return original_invoke(args, operation, run_dir, **kwargs)

        with patch.object(sync, "invoke", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "Connection lost"):
                sync.sync_run(self.args, self.run)
        self.assertEqual((self.destination / "design_model_0.cif").read_bytes(), b"data_design\n")
        self.assertEqual(list(self.destination.glob(".mosaic-sync*")), [])

    def test_changed_source_after_inventory_fails_hash_without_replacing_final(self):
        self.make_result()
        sync.sync_run(self.args, self.run)
        self.make_result(content=b"data_next\n")
        original_invoke = sync.invoke

        def change(args, operation, run_dir, **kwargs):
            if operation == "transfer":
                self.make_result(content=b"data_changed_while_downloading\n")
            return original_invoke(args, operation, run_dir, **kwargs)

        with patch.object(sync, "invoke", side_effect=change):
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                sync.sync_run(self.args, self.run)
        self.assertEqual((self.destination / "design_model_0.cif").read_bytes(), b"data_design\n")

    def test_unrecognized_and_locally_edited_files_are_not_overwritten(self):
        self.make_result()
        self.destination.mkdir(parents=True)
        target = self.destination / "design_model_0.cif"
        target.write_bytes(b"user content")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            sync.sync_run(self.args, self.run)
        self.assertEqual(target.read_bytes(), b"user content")
        target.unlink()
        sync.sync_run(self.args, self.run)
        target.write_bytes(b"user edit")
        with self.assertRaisesRegex(ValueError, "locally modified"):
            sync.sync_run(self.args, self.run)
        self.assertEqual(target.read_bytes(), b"user edit")

    def test_remote_truncated_gzip_refused_before_local_output(self):
        source = self.make_result()
        source.write_bytes(source.read_bytes()[:-8])
        with self.assertRaises(RuntimeError):
            sync.sync_run(self.args, self.run)
        self.assertFalse((self.destination / "design_model_0.cif").exists())

    def test_local_gzip_crc_checked_even_when_raw_hash_matches_inventory(self):
        source = self.make_result()
        inventory = json.loads(sync.invoke(self.args, "inventory", str(self.remote)))
        source.write_bytes(source.read_bytes()[:-8])
        inventory["files"][0]["raw"] = sync.digest_file(source)
        self.destination.mkdir(parents=True)
        with self.assertRaises(EOFError):
            sync.sync_inventory(self.args, inventory, self.destination)
        self.assertFalse((self.destination / "design_model_0.cif").exists())

    def test_archive_member_substitution_is_rejected_without_extraction(self):
        self.make_result()
        original_invoke = sync.invoke

        def substitute(args, operation, run_dir, **kwargs):
            if operation == "transfer":
                with tarfile.open(fileobj=kwargs["stdout"], mode="w|") as archive:
                    entry = tarfile.TarInfo("../escape")
                    entry.size = 4
                    archive.addfile(entry, io.BytesIO(b"evil"))
                return None
            return original_invoke(args, operation, run_dir, **kwargs)

        with patch.object(sync, "invoke", side_effect=substitute):
            with self.assertRaisesRegex(ValueError, "members do not match"):
                sync.sync_run(self.args, self.run)
        self.assertFalse((self.destination.parent / "escape").exists())
        self.assertFalse((self.destination / "design_model_0.cif").exists())

    def test_traversal_and_symlink_destinations_rejected(self):
        with self.assertRaises(ValueError):
            sync.output_directory(self.args.destination_root, "../other")
        self.args.destination_root.mkdir()
        (self.args.destination_root / "linked").symlink_to(self.remote, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            sync.output_directory(self.args.destination_root, "linked/job")

    def test_pending_sidecar_keeps_completed_outputs_and_can_be_retried(self):
        self.make_result()
        self.make_result("next_model_0", b"data_next\n")
        metadata = self.remote / "next_model_0.json"
        metadata.write_text('{"incomplete":')
        result = sync.sync_run(self.args, self.run)
        self.assertEqual(result["downloaded"], 1)
        self.assertEqual(result["pending"][0]["result_json"], metadata.name)
        self.assertTrue((self.destination / "design_model_0.cif").exists())
        self.assertFalse((self.destination / "next_model_0.cif").exists())
        metadata.write_text("{}")
        result = sync.sync_run(self.args, self.run)
        self.assertEqual((result["downloaded"], result["skipped"], result["pending"]), (1, 1, []))

    def test_parallel_cli_collects_failed_and_pending_runs_without_aborting_others(self):
        manifest = self.root / "runs.json"
        manifest.write_text(json.dumps({"runs": [
            {"remote_run_dir": "/" + name, "destination": name}
            for name in ("failed", "pending", "completed")
        ]}))

        def outcome(args, run):
            if run["destination"] == "failed":
                raise RuntimeError("test connection failure")
            return {**run, "pending": ([{"reason": "partial JSON"}]
                                       if run["destination"] == "pending" else [])}

        output = io.StringIO()
        with patch.object(sync, "sync_run", side_effect=outcome) as run_mock:
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_context:
                    sync.main(["--manifest", str(manifest), "--host", "test-host",
                               "--destination-root", str(self.args.destination_root), "--jobs", "2"])
        self.assertEqual(exit_context.exception.code, 1)
        self.assertEqual(run_mock.call_count, 3)
        rows = {item["destination"]: item for item in map(json.loads, output.getvalue().splitlines())}
        self.assertEqual(rows["failed"]["status"], "failed")
        self.assertTrue(rows["pending"]["pending"])
        self.assertEqual(rows["completed"]["pending"], [])


if __name__ == "__main__":
    unittest.main()
