import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rfd3_mosaic.assembly_compiler import CompiledAudit
from rfd3_mosaic.audit_evidence import (
    BINDING_NAME,
    audit_evidence,
    verify_audit_evidence,
)


class AuditEvidenceTestCase(unittest.TestCase):
    def test_legacy_seed_dependencies_and_auditor_revision_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = root / "input.json"
            compiled.write_text('{"example": {}}')
            result = root / "result_model_0.json"
            result.write_text("{}")
            result.with_suffix(".cif").write_text("data_fixture\n")
            mapping = root / "mapping.json"
            mapping.write_text("{}")
            source = root / "seed.pdb"
            source.write_text("REMARK seed fixture\n")
            config = root / "assembly_specification.yaml"
            config.write_text(yaml.safe_dump({"interface_seed": {"fragments": {
                "F1": {"source": "seed.pdb", "selection": "A1"},
            }}}))
            report = root / "seed_integrity_audit.json"
            report.write_text('{"passed": true}')
            audit = CompiledAudit(
                module="rfd3_mosaic.rfd3_seed_audit", report_name=report.name,
                input_arguments=(("--adapter-mapping", str(mapping)), ("--config", str(config))),
            )
            kwargs = dict(compiled_input=compiled, result_json=result,
                          reports=(report,), semantic_audits=(audit,))
            for changed in (mapping, config, source, report):
                with self.subTest(path=changed.name):
                    original = changed.read_text()
                    (root / BINDING_NAME).write_text(json.dumps(audit_evidence(**kwargs)))
                    changed.write_text(original + "\n")
                    with self.assertRaisesRegex(ValueError, "evidence changed"):
                        verify_audit_evidence(directory=root, **kwargs)
                    changed.write_text(original)
            (root / BINDING_NAME).write_text(json.dumps(audit_evidence(**kwargs)))
            with patch("rfd3_mosaic.audit_evidence._auditor_identity", return_value={"revision": "changed"}):
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    verify_audit_evidence(directory=root, **kwargs)

    def test_wheel_installation_does_not_borrow_cwd_checkout(self):
        from rfd3_mosaic.installation import source_repository_root
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "unrelated_checkout"
            checkout.mkdir()
            (checkout / ".project-root").touch()
            installed = root / "site-packages" / "rfd3_mosaic" / "installation.py"
            installed.parent.mkdir(parents=True)
            installed.touch()
            with (
                patch("rfd3_mosaic.installation.__file__", str(installed)),
                patch("pathlib.Path.cwd", return_value=checkout),
            ):
                self.assertIsNone(source_repository_root())


if __name__ == "__main__":
    unittest.main()
