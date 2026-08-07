import unittest
from unittest.mock import patch

import numpy as np

from rfd3.inference.symmetry.frames import (
    RTs_to_framecoords,
    framecoords_to_RTs,
    get_dihedral_frames,
    get_symmetry_frames_from_symmetry_id,
    get_symmetry_multiplicity_from_id,
)
from rfd3.inference.symmetry.symmetry_utils import (
    SymmetryConfig,
    _resolve_symmetry_frames,
    make_symmetric_atom_array_for_partial_diffusion,
)
from rfd3_mosaic.geometry import build_polyhedral_registry
from rfd3_mosaic.schema.specs import SymmetryType


class RFD3DihedralFrameCompatibilityTestCase(unittest.TestCase):
    def test_compiler_validated_multi_asu_uses_declared_frames(self) -> None:
        native = get_symmetry_frames_from_symmetry_id("D3")
        order = [f"D3:t{index}" for index in range(len(native))]
        matrices = {}
        for transform_id, (rotation, translation) in zip(order, native):
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
            matrices[transform_id] = matrix.tolist()
        config = SymmetryConfig(
            id="D3",
            is_symmetric_motif=True,
            use_declared_frames=True,
            declared_transform_order=order,
            declared_transform_matrices=matrices,
        )
        source = object()
        with patch(
            "rfd3.inference.symmetry.symmetry_utils."
            "get_symmetry_frames_from_atom_array"
        ) as recover:
            frames = _resolve_symmetry_frames(config, source)

        self.assertEqual(len(frames), 6)
        recover.assert_not_called()

    def test_declared_registry_order_controls_runtime_frame_order(self) -> None:
        native = get_symmetry_frames_from_symmetry_id("C3")
        order = ["C3:e", "C3:r2", "C3:r1"]
        matrices = {}
        for transform_id, frame in zip(
            order,
            (native[0], native[2], native[1]),
        ):
            rotation, translation = frame
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
            matrices[transform_id] = matrix.tolist()
        config = SymmetryConfig(
            id="C3",
            is_symmetric_motif=True,
            use_declared_frames=True,
            declared_transform_order=order,
            declared_transform_matrices=matrices,
        )

        frames = _resolve_symmetry_frames(config, object())

        self.assertTrue(np.allclose(frames[1][0], native[2][0]))
        self.assertTrue(np.allclose(frames[2][0], native[1][0]))

    def test_default_symmetric_input_still_recovers_frames(self) -> None:
        config = SymmetryConfig(id="D3", is_symmetric_motif=True)
        source = object()
        recovered = [(np.eye(3), np.zeros(3))]
        with patch(
            "rfd3.inference.symmetry.symmetry_utils."
            "get_symmetry_frames_from_atom_array",
            return_value=recovered,
        ) as recover:
            frames = _resolve_symmetry_frames(config, source)

        self.assertIs(frames, recovered)
        recover.assert_called_once()

    def test_c12_has_twelve_native_frames(self) -> None:
        self.assertEqual(
            len(get_symmetry_frames_from_symmetry_id("C12")),
            12,
        )

    def test_polyhedral_ids_report_finite_group_multiplicity(self) -> None:
        for symmetry_id, expected in (("T", 12), ("O", 24), ("I", 60)):
            with self.subTest(symmetry_id=symmetry_id):
                self.assertEqual(
                    get_symmetry_multiplicity_from_id(symmetry_id),
                    expected,
                )

    def test_tetrahedral_declared_frames_bypass_legacy_generator(self) -> None:
        registry = build_polyhedral_registry(SymmetryType.TETRAHEDRAL)
        order = list(registry.transform_ids)
        matrices = {
            transform_id: registry.transform(transform_id).tolist()
            for transform_id in order
        }
        config = SymmetryConfig(
            id="T",
            is_symmetric_motif=True,
            use_declared_frames=True,
            declared_transform_order=order,
            declared_transform_matrices=matrices,
        )

        with patch(
            "rfd3.inference.symmetry.symmetry_utils."
            "get_symmetry_frames_from_symmetry_id"
        ) as legacy:
            frames = _resolve_symmetry_frames(config, None)

        self.assertEqual(len(frames), 12)
        np.testing.assert_allclose(frames[0][0], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(frames[0][1], np.zeros(3), atol=1e-7)
        legacy.assert_not_called()

    def test_polyhedral_declared_frames_require_complete_group(self) -> None:
        registry = build_polyhedral_registry(SymmetryType.TETRAHEDRAL)
        order = list(registry.transform_ids[:-1])
        matrices = {
            transform_id: registry.transform(transform_id).tolist()
            for transform_id in order
        }
        config = SymmetryConfig(
            id="T",
            use_declared_frames=True,
            declared_transform_order=order,
            declared_transform_matrices=matrices,
        )

        with self.assertRaisesRegex(ValueError, "count does not match"):
            _resolve_symmetry_frames(config, None)

    def test_polyhedral_partial_diffusion_fails_closed(self) -> None:
        config = SymmetryConfig(
            id="T",
            use_declared_frames=True,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "Partial diffusion for declared-frame T/O/I",
        ):
            make_symmetric_atom_array_for_partial_diffusion(None, config)

    def test_polyhedral_frames_survive_rfd3_virtual_frame_transport(
        self,
    ) -> None:
        for symmetry_type in (
            SymmetryType.TETRAHEDRAL,
            SymmetryType.OCTAHEDRAL,
            SymmetryType.ICOSAHEDRAL,
        ):
            with self.subTest(symmetry_type=symmetry_type.value):
                registry = build_polyhedral_registry(symmetry_type)
                rotations = np.stack(
                    [
                        registry.transform(transform_id)[:3, :3]
                        for transform_id in registry.transform_ids
                    ]
                )
                translations = np.stack(
                    [
                        registry.transform(transform_id)[:3, 3]
                        for transform_id in registry.transform_ids
                    ]
                )

                origins, x_axes, y_axes = RTs_to_framecoords(
                    rotations,
                    translations,
                )
                recovered_rotations, recovered_translations = (
                    framecoords_to_RTs(origins, x_axes, y_axes)
                )

                np.testing.assert_allclose(
                    recovered_rotations.numpy(),
                    rotations,
                    atol=5e-6,
                )
                np.testing.assert_allclose(
                    recovered_translations.numpy(),
                    translations,
                    atol=1e-7,
                )

    def test_orders_divisible_by_three_have_all_unique_frames(self) -> None:
        for order in (3, 6):
            with self.subTest(order=order):
                rotations = [
                    rotation
                    for rotation, _ in get_dihedral_frames(order)
                ]
                for left_index, left in enumerate(rotations):
                    for right_index, right in enumerate(rotations):
                        if left_index == right_index:
                            continue
                        self.assertFalse(
                            np.allclose(left, right, atol=1e-9)
                        )

    def test_common_dihedral_groups_are_closed(self) -> None:
        for order in (2, 3, 5, 6):
            with self.subTest(order=order):
                rotations = [
                    rotation
                    for rotation, _ in get_dihedral_frames(order)
                ]
                for left in rotations:
                    for right in rotations:
                        composed = left @ right
                        self.assertTrue(
                            any(
                                np.allclose(
                                    composed,
                                    candidate,
                                    atol=1e-9,
                                )
                                for candidate in rotations
                            )
                        )


if __name__ == "__main__":
    unittest.main()
