from __future__ import annotations

from pathlib import Path
import runpy
import unittest


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPOSITORY / "scripts" / "rfd3_mosaic"
SCRIPT_ARCHIVE = SCRIPT_ROOT / "archive" / "legacy_direct"
EXPERIMENT_ROOT = REPOSITORY / "experiments"


class RepositoryLayoutTestCase(unittest.TestCase):
    def test_active_scripts_do_not_bypass_the_public_worker(self) -> None:
        forbidden = (
            "python -m rfd3.run_inference",
            "python -m rfd3_mosaic.rfd3_adapter",
        )
        offenders: list[str] = []
        for path in SCRIPT_ROOT.iterdir():
            if not path.is_file() or path.suffix not in {".py", ".sh", ".sbatch"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_historical_direct_scripts_are_physically_isolated(self) -> None:
        self.assertTrue(SCRIPT_ARCHIVE.is_dir())
        direct = []
        for path in SCRIPT_ARCHIVE.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if (
                "python -m rfd3.run_inference" in text
                or "python -m rfd3_mosaic.rfd3_adapter" in text
            ):
                direct.append(path.name)
        self.assertGreater(len(direct), 0)

    def test_release_gates_reference_only_active_experiments(self) -> None:
        namespace = runpy.run_path(
            str(SCRIPT_ROOT / "submit_gpu_release_gates.py")
        )
        gates = namespace["GATES"]
        self.assertGreater(len(gates), 0)
        for gate_name, gate in gates.items():
            with self.subTest(gate=gate_name):
                design = REPOSITORY / str(gate["design"])
                self.assertTrue(design.is_file(), design)
                self.assertNotIn("archive", design.parts)

    def test_remaining_closure_gates_record_acceptance_contracts(self) -> None:
        namespace = runpy.run_path(
            str(SCRIPT_ROOT / "submit_gpu_release_gates.py")
        )
        gates = namespace["GATES"]
        closure = {
            name: gate
            for name, gate in gates.items()
            if gate["tier"] == "closure"
        }
        self.assertEqual(
            set(closure),
            {
                "cross-chain-topology",
                "c4-c2-quotient",
                "t-dynamic",
                "o-dynamic",
                "i-continuity",
            },
        )
        for gate_name, gate in closure.items():
            with self.subTest(gate=gate_name):
                self.assertTrue(gate.get("acceptance"))

    def test_superseded_experiments_are_not_in_active_directory(self) -> None:
        superseded = EXPERIMENT_ROOT / "archive" / "superseded"
        self.assertTrue(superseded.is_dir())
        archived_names = {
            path.name for path in superseded.glob("*.yaml") if path.is_file()
        }
        active_names = {
            path.name for path in EXPERIMENT_ROOT.glob("*.yaml") if path.is_file()
        }
        self.assertTrue(archived_names)
        self.assertTrue(archived_names.isdisjoint(active_names))


if __name__ == "__main__":
    unittest.main()
