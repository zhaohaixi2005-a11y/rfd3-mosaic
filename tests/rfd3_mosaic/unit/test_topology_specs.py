import unittest

from pydantic import ValidationError

from rfd3_mosaic.schema import (
    MotionGroupInitializationSpec,
    CopyRelationSpec,
    GeometricConstraintsGeometry,
    InterfaceEdgeSpec,
    LinkLengthSpec,
    ReferenceTransformGeometry,
    ScaffoldEndpointSpec,
    ScaffoldLinkSpec,
    SymmetryOrbitSpec,
    SymmetryTransformSetSpec,
    SymmetryType,
    Terminus,
)


class TopologySpecsTestCase(unittest.TestCase):
    def test_copy_relation_accepts_orbit_offset(self) -> None:
        relation = CopyRelationSpec(orbit_offset=-1)

        self.assertEqual(relation.orbit_offset, -1)
        self.assertIsNone(relation.transform)

    def test_copy_relation_accepts_named_group_transform(self) -> None:
        relation = CopyRelationSpec(transform="D3:s0")

        self.assertEqual(relation.transform, "D3:s0")
        self.assertIsNone(relation.orbit_offset)

    def test_copy_relation_rejects_malformed_transform_id(self) -> None:
        with self.assertRaises(ValidationError):
            CopyRelationSpec(transform="D3_s0")

    def test_copy_relation_rejects_two_relations(self) -> None:
        with self.assertRaises(ValidationError):
            CopyRelationSpec(
                orbit_offset=-1,
                transform="C3:r2",
            )

    def test_copy_relation_rejects_empty_relation(self) -> None:
        with self.assertRaises(ValidationError):
            CopyRelationSpec()

    def test_reference_geometry_uses_reference_seed(self) -> None:
        geometry = ReferenceTransformGeometry(
            from_reference_seed=True,
            translation_tolerance=2.0,
            rotation_tolerance_deg=10.0,
        )

        self.assertTrue(geometry.from_reference_seed)
        self.assertIsNone(geometry.target_transform)

    def test_explicit_reference_geometry_requires_transform(self) -> None:
        with self.assertRaises(ValidationError):
            ReferenceTransformGeometry(
                from_reference_seed=False,
            )

    def test_empty_geometric_constraints_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GeometricConstraintsGeometry()

    def test_interface_edge_parses_discriminated_geometry(self) -> None:
        edge = InterfaceEdgeSpec(
            left_port="left_port",
            right_port="right_port",
            copy_relation={"orbit_offset": -1},
            target_geometry={
                "mode": "reference_transform",
                "from_reference_seed": True,
                "translation_tolerance": 2.0,
                "rotation_tolerance_deg": 10.0,
            },
        )

        self.assertIsInstance(
            edge.target_geometry,
            ReferenceTransformGeometry,
        )

    def test_zero_symmetry_axis_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SymmetryTransformSetSpec(
                type=SymmetryType.CYCLIC,
                order=3,
                axis=(0.0, 0.0, 0.0),
            )

    def test_symmetry_orbit_rejects_duplicate_groups(self) -> None:
        with self.assertRaises(ValidationError):
            SymmetryOrbitSpec(
                transform_set="ring_c3",
                master_groups=["primary_seed", "primary_seed"],
            )

    def test_valid_directed_scaffold_link(self) -> None:
        link = ScaffoldLinkSpec(
            from_endpoint=ScaffoldEndpointSpec(
                fragment="right",
                terminus=Terminus.C,
            ),
            to_endpoint=ScaffoldEndpointSpec(
                fragment="left",
                terminus=Terminus.N,
            ),
            length=LinkLengthSpec(
                minimum=70,
                maximum=100,
            ),
            tie_group="protomer_length",
        )

        self.assertEqual(link.from_endpoint.terminus, Terminus.C)
        self.assertEqual(link.to_endpoint.terminus, Terminus.N)

    def test_reverse_scaffold_direction_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScaffoldLinkSpec(
                from_endpoint=ScaffoldEndpointSpec(
                    fragment="right",
                    terminus=Terminus.N,
                ),
                to_endpoint=ScaffoldEndpointSpec(
                    fragment="left",
                    terminus=Terminus.C,
                ),
                length=LinkLengthSpec(
                    minimum=70,
                    maximum=100,
                ),
            )

    def test_invalid_link_length_range_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            LinkLengthSpec(
                minimum=100,
                maximum=70,
            )

    def test_radial_initialization_accepts_explicit_pose(self) -> None:
        initialization = MotionGroupInitializationSpec(
            center_method="interface_heavy_atom_com",
            orientation={
                "method": "fixed",
                "rotation_deg": (0.0, 0.0, 0.0),
            },
            placement={
                "radius": {"mean": 25.0, "range": 0.0},
                "axial_offset": {"mean": 0.0, "range": 0.0},
                "radial_direction": (1.0, 0.0, 0.0),
            },
        )

        self.assertEqual(initialization.placement.radius.mean, 25.0)

    def test_radial_initialization_accepts_uniform_so3_sampling(self) -> None:
        initialization = MotionGroupInitializationSpec(
            orientation={"method": "uniform_so3"},
            placement={
                "radius": {"mean": 25.0, "range": 5.0},
                "radial_direction": (1.0, 0.0, 0.0),
            },
        )

        self.assertEqual(initialization.orientation.method, "uniform_so3")
        self.assertEqual(initialization.placement.radius.range, 5.0)

    def test_radial_direction_cannot_be_zero(self) -> None:
        with self.assertRaises(ValidationError):
            MotionGroupInitializationSpec(
                placement={
                    "radius": {"mean": 25.0},
                    "radial_direction": (0.0, 0.0, 0.0),
                }
            )


if __name__ == "__main__":
    unittest.main()
