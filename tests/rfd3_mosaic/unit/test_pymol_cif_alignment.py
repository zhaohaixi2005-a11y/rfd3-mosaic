import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts"
    / "rfd3_mosaic"
    / "pymol_fixed_orbit_alignment.py"
)


class _FakeCmd:
    def __init__(self, objects: dict[str, list[SimpleNamespace]]) -> None:
        self.objects = objects
        self.extensions = {}

    def extend(self, name, function) -> None:
        self.extensions[name] = function

    def get_names(self, _kind, enabled_only=0):
        del enabled_only
        return list(self.objects)

    def get_model(self, name):
        return SimpleNamespace(atom=self.objects[name])

    def create(self, output, source) -> None:
        self.objects[output] = copy.deepcopy(self.objects[source])

    def delete(self, name) -> None:
        self.objects.pop(name, None)

    def get_coords(self, name):
        return np.asarray([atom.coord for atom in self.objects[name]])

    def load_coords(self, coordinates, name) -> None:
        for atom, coordinate in zip(self.objects[name], coordinates):
            atom.coord = np.asarray(coordinate, dtype=float)

    def get_chains(self, name):
        return sorted({atom.chain for atom in self.objects[name]})

    def hide(self, *_args) -> None:
        pass

    def show(self, *_args) -> None:
        pass

    def color(self, *_args) -> None:
        pass

    def set(self, *_args) -> None:
        pass

    def disable(self, *_args) -> None:
        pass

    def zoom(self, *_args) -> None:
        pass


def _residue_atoms(
    chain: str,
    residue: int,
    residue_name: str,
    coordinates: np.ndarray,
) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            chain=chain,
            resi=str(residue),
            resn=residue_name,
            name=atom_name,
            symbol=atom_name[0],
            coord=np.asarray(coordinate, dtype=float),
        )
        for atom_name, coordinate in zip(
            ("N", "CA", "C"),
            coordinates,
            strict=True,
        )
    ]


def _motif(chain: str, offset: np.ndarray) -> list[SimpleNamespace]:
    first = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    second = np.asarray(
        [[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 1.0]]
    )
    return (
        _residue_atoms(chain, 1, "ALA", first + offset)
        + _residue_atoms(chain, 2, "GLY", second + offset)
    )


def _load_alignment_module(fake_cmd: _FakeCmd):
    fake_pymol = ModuleType("pymol")
    fake_pymol.cmd = fake_cmd
    spec = importlib.util.spec_from_file_location(
        "test_pymol_fixed_orbit_alignment",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pymol": fake_pymol}):
        spec.loader.exec_module(module)
    return module


class PymolCifAlignmentTestCase(unittest.TestCase):
    def test_argument_free_alignment_uses_only_two_loaded_structures(
        self,
    ) -> None:
        translation = np.asarray([5.0, -2.0, 1.0])
        reference = _motif("X", np.zeros(3)) + _motif(
            "Y", np.asarray([10.0, 0.0, 0.0])
        )
        design = []
        for chain, offset in (
            ("A", np.zeros(3)),
            ("B", np.asarray([10.0, 0.0, 0.0])),
        ):
            design.extend(
                _residue_atoms(
                    chain,
                    1,
                    "SER",
                    np.asarray(
                        [[-4.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [-3.0, 1.0, 0.0]]
                    )
                    + offset,
                )
            )
            for atom in _motif(chain, offset - translation):
                atom.resi = str(int(atom.resi) + 1)
                design.append(atom)
            design.extend(
                _residue_atoms(
                    chain,
                    4,
                    "THR",
                    np.asarray(
                        [[4.0, 2.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0]]
                    )
                    + offset,
                )
            )

        fake_cmd = _FakeCmd(
            {
                "unknown_output": design,
                "unknown_input": reference,
            }
        )
        module = _load_alignment_module(fake_cmd)

        module.mosaic_align()
        module.mosaic_align()

        self.assertIn("mosaic_aligned", fake_cmd.objects)
        aligned = module._structure_residues("mosaic_aligned")
        expected = module._structure_residues("unknown_input")
        for aligned_chain, reference_chain in (("A", "X"), ("B", "Y")):
            aligned_motif = aligned[aligned_chain][1:3]
            for observed_residue, expected_residue in zip(
                aligned_motif,
                expected[reference_chain],
                strict=True,
            ):
                np.testing.assert_allclose(
                    observed_residue["atoms"]["CA"],
                    expected_residue["atoms"]["CA"],
                    atol=1e-7,
                )


if __name__ == "__main__":
    unittest.main()
