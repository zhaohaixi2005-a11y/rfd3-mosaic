import unittest

from pydantic import ValidationError

from rfd3_mosaic.schema import (
    EntityType,
    FragmentRole,
    FragmentSpec,
    FrameMethod,
    InterfacePortFrameSpec,
    InterfacePortSpec,
    MotionBounds,
    MotionGroupSpec,
    MotionMode,
)


class SpecsTestCase(unittest.TestCase):
    def test_fragment_spec_accepts_valid_fragment(self) -> None:
        fragment = FragmentSpec(
            source="inputs/7mwr_interface.pdb",
            selection="A/165-194/*",
            entity_type=EntityType.PROTEIN,
            role=FragmentRole.INTERFACE_MOTIF,
            fixed_atoms="backbone",
        )

        self.assertEqual(fragment.selection, "A/165-194/*")
        self.assertEqual(fragment.entity_type, EntityType.PROTEIN)

    def test_fragment_spec_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            FragmentSpec(
                source="inputs/7mwr_interface.pdb",
                selection="A/165-194/*",
                entity_type="protein",
                role="interface_motif",
                flexibility="rigid",
            )

    def test_soft_rigid_group_requires_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            MotionGroupSpec(
                members=["left", "right"],
                mode=MotionMode.SOFT_RIGID,
            )

    def test_rigid_group_rejects_motion_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            MotionGroupSpec(
                members=["left", "right"],
                mode=MotionMode.RIGID,
                bounds=MotionBounds(
                    max_translation=1.0,
                    max_rotation_deg=5.0,
                ),
            )

    def test_motion_group_rejects_duplicate_members(self) -> None:
        with self.assertRaises(ValidationError):
            MotionGroupSpec(
                members=["left", "left"],
                mode=MotionMode.RIGID,
            )

    def test_anchor_frame_accepts_valid_definition(self) -> None:
        frame = InterfacePortFrameSpec(
            method=FrameMethod.ANCHORS,
            origin_atoms=["left:A165:CA"],
            x_axis_atoms=["left:A165:CA", "left:A175:CA"],
            xy_plane_atoms=[
                "left:A165:CA",
                "left:A175:CA",
                "left:A185:CA",
            ],
        )

        self.assertEqual(frame.method, FrameMethod.ANCHORS)

    def test_anchor_frame_rejects_missing_plane_atoms(self) -> None:
        with self.assertRaises(ValidationError):
            InterfacePortFrameSpec(
                method=FrameMethod.ANCHORS,
                origin_atoms=["left:A165:CA"],
                x_axis_atoms=["left:A165:CA", "left:A175:CA"],
            )

    def test_precomputed_frame_requires_4x4_transform(self) -> None:
        with self.assertRaises(ValidationError):
            InterfacePortFrameSpec(
                method=FrameMethod.PRECOMPUTED,
                transform=[
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            )

    def test_interface_port_accepts_valid_definition(self) -> None:
        port = InterfacePortSpec(
            group="primary_seed",
            fragments=["left"],
            atoms="heavy",
            frame=InterfacePortFrameSpec(
                method=FrameMethod.REFERENCE_INTERFACE_PCA,
            ),
        )

        self.assertEqual(port.group, "primary_seed")
        self.assertEqual(port.fragments, ["left"])


if __name__ == "__main__":
    unittest.main()