import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts"
    / "rfd3_mosaic"
    / "load_cif_ensemble.py"
)


class _FakeCmd:
    def __init__(self) -> None:
        self.extensions = {}
        self.keys = {}
        self.objects: dict[str, list[bytes]] = {}
        self.state = 1
        self.oriented = None

    def extend(self, name, function) -> None:
        self.extensions[name] = function

    def get_names(self, _kind):
        return list(self.objects)

    def load(self, path, name, *, state, discrete, quiet) -> None:
        self.assert_load_options(discrete, quiet)
        states = self.objects.setdefault(name, [])
        self.assert_next_state(states, state)
        states.append(Path(path).read_bytes())

    @staticmethod
    def assert_load_options(discrete, quiet) -> None:
        if (discrete, quiet) != (1, 1):
            raise AssertionError((discrete, quiet))

    @staticmethod
    def assert_next_state(states, state) -> None:
        if state != len(states) + 1:
            raise AssertionError((len(states), state))

    def set(self, name, value) -> None:
        if name == "state":
            self.state = int(value)

    def set_key(self, key, function) -> None:
        self.keys[key] = function

    def orient(self, name) -> None:
        self.oriented = name

    def count_states(self, name) -> int:
        return len(self.objects[name])

    def get_state(self) -> int:
        return self.state

    def refresh(self) -> None:
        pass


def _load_module(fake_cmd: _FakeCmd):
    fake_pymol = ModuleType("pymol")
    fake_pymol.cmd = fake_cmd
    spec = importlib.util.spec_from_file_location(
        "test_load_cif_ensemble",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pymol": fake_pymol}):
        spec.loader.exec_module(module)
    return module


class PymolCifEnsembleTestCase(unittest.TestCase):
    def test_zip_loads_plain_cifs_as_states_and_installs_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "generated_structures_cif.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("design_00000.cif", "data_first\n")
                archive.writestr("design_00001.cif", "data_second\n")
                archive.writestr("manifest.json", "{}")

            fake_cmd = _FakeCmd()
            module = _load_module(fake_cmd)
            module.load_cif_ensemble(str(archive_path), "campaign")

            self.assertEqual(
                fake_cmd.objects["campaign"],
                [b"data_first\n", b"data_second\n"],
            )
            self.assertEqual(fake_cmd.oriented, "campaign")
            self.assertEqual(
                set(fake_cmd.keys),
                {"RIGHT", "LEFT", "PGDN", "PGUP"},
            )

            fake_cmd.keys["RIGHT"]()
            self.assertEqual(fake_cmd.state, 2)
            fake_cmd.keys["RIGHT"]()
            self.assertEqual(fake_cmd.state, 1)


if __name__ == "__main__":
    unittest.main()
