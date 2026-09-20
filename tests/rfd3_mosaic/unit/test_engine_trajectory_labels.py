"""Exported noisy/denoised trajectories must retain their sampler meanings."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from biotite.structure import AtomArray

from rfd3.engine import RFD3InferenceEngine


class EngineTrajectoryLabelsTestCase(unittest.TestCase):
    def _forward(self, *, align: bool):
        atom_array = AtomArray(4)
        atom_array.atom_name = np.asarray(["N", "CA", "C", "O"])
        atom_array.res_name = np.asarray(["GLY"] * 4)
        atom_array.chain_id = np.asarray(["A"] * 4)
        atom_array.res_id = np.ones(4, dtype=int)
        atom_array.element = np.asarray(["N", "C", "C", "O"])
        base = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        denoised = [
            torch.stack([base + 20.0 * design + 4.0 * step for design in range(2)])
            for step in range(3)
        ]
        noisy = [
            torch.stack([
                base * (step + 2.0) + 100.0 + 30.0 * design
                for design in range(2)
            ])
            for step in range(3)
        ]
        output = {
            "network_output": {
                "X_noisy_L_traj": noisy,
                "X_denoised_L_traj": denoised,
            },
            "predicted_atom_array_stack": [atom_array.copy(), atom_array.copy()],
            "prediction_metadata": [{}, {}],
        }
        engine = SimpleNamespace(
            dump_trajectories=True,
            align_trajectory_structures=align,
            dump_prediction_metadata_json=False,
            trainer=SimpleNamespace(
                fabric=SimpleNamespace(to_device=lambda batch: batch),
                validation_step=lambda **kwargs: output,
            ),
        )
        batch = {
            "coord_atom_lvl_to_be_noised": base,
            "atom_array": atom_array,
            "example_id": "trajectory_probe",
        }
        results = RFD3InferenceEngine._model_forward(engine, batch)
        return results, noisy, denoised

    def test_sampler_sources_reach_correct_export_labels_for_each_design(self):
        results, noisy, denoised = self._forward(align=False)
        self.assertEqual(len(results), 2)
        with tempfile.TemporaryDirectory() as temporary:
            for design, result in enumerate(results):
                expected_noisy = torch.stack(noisy)[:, design].flip(0).numpy()
                expected_denoised = torch.stack(denoised)[:, design].flip(0).numpy()
                np.testing.assert_array_equal(result.noisy_trajectory_stack.coord, expected_noisy)
                np.testing.assert_array_equal(result.denoised_trajectory_stack.coord, expected_denoised)
                with patch("rfd3.engine.to_cif_file") as write_cif:
                    result.dump(temporary, verbose=False)
                exported = {
                    Path(str(call.args[1])).name: call.args[0].coord
                    for call in write_cif.call_args_list
                }
                np.testing.assert_array_equal(
                    exported[f"trajectory_probe_noisy_model_{design}"], expected_noisy
                )
                np.testing.assert_array_equal(
                    exported[f"trajectory_probe_denoised_model_{design}"], expected_denoised
                )

    def test_optional_alignment_preserves_noisy_trajectory_coordinates(self):
        results, noisy, denoised = self._forward(align=True)
        for design, result in enumerate(results):
            np.testing.assert_array_equal(
                result.noisy_trajectory_stack.coord,
                torch.stack(noisy)[:, design].flip(0).numpy(),
            )
            expected = denoised[-1][design].numpy()
            for frame in result.denoised_trajectory_stack.coord:
                np.testing.assert_allclose(frame, expected, atol=1e-4, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
