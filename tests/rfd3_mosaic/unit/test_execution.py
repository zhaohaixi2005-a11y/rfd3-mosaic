from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from rfd3_mosaic.execution import SlurmExecutor, executor_for_id


class ExecutionTestCase(unittest.TestCase):
    def test_slurm_uses_parsable_job_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "job.sbatch"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="5734000;cluster\n", stderr=""
            )
            with patch("subprocess.run", return_value=completed) as run:
                result = SlurmExecutor().submit(script)

        self.assertEqual(result.executor, "slurm")
        self.assertEqual(result.job_id, "5734000")
        self.assertEqual(result.message, "Submitted batch job 5734000")
        self.assertEqual(
            run.call_args.args[0],
            ["sbatch", "--parsable", str(script)],
        )

    def test_slurm_rejects_unparseable_scheduler_output(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-a-job\n", stderr=""
        )
        with patch("subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "invalid JobID"):
                SlurmExecutor().submit(Path("job.sbatch"))

    def test_unknown_executor_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown executor"):
            executor_for_id("mystery")

    def test_slurm_failure_is_reported_without_a_python_traceback_contract(
        self,
    ) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["sbatch"],
            stderr="QOSMaxSubmitJobPerUserLimit",
        )
        with patch("subprocess.run", side_effect=error):
            with self.assertRaisesRegex(
                RuntimeError, "QOSMaxSubmitJobPerUserLimit"
            ):
                SlurmExecutor().submit(Path("job.sbatch"))


if __name__ == "__main__":
    unittest.main()
