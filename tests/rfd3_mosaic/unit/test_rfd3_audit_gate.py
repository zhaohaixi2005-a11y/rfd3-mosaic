import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.rfd3_audit_gate import failed_audit_paths


class RFD3AuditGateTestCase(unittest.TestCase):
    def test_returns_every_failed_or_malformed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passed = root / "passed.json"
            failed = root / "failed.json"
            missing_flag = root / "missing-flag.json"
            passed.write_text(json.dumps({"passed": True}))
            failed.write_text(json.dumps({"passed": False}))
            missing_flag.write_text(json.dumps({"status": "passed"}))

            observed = failed_audit_paths(
                [passed, failed, missing_flag]
            )

            self.assertEqual(observed, [failed, missing_flag])


if __name__ == "__main__":
    unittest.main()
