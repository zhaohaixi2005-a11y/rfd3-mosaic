from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import yaml

from rfd3_mosaic.geometry.symmetry_registry import (
    SymmetryTransformRegistry,
    build_transform_registry,
)
from rfd3_mosaic.geometry.frames import reference_interface_pca_frame
from rfd3_mosaic.geometry.se3 import validate_transform
from rfd3_mosaic.geometry.se3 import (
    axis_angle_rotation,
    compose_transforms,
    make_transform,
)
from rfd3_mosaic.schema import (
    CompiledInstanceSet,
    FragmentInstance,
    InterfaceEdgeInstance,
    InterfacePortInstance,
    InterfaceSeedSpec,
    MotionGroupInstance,
    ScaffoldLinkInstance,
)
from rfd3_mosaic.schema.instances import TransformMatrix
from rfd3_mosaic.schema.specs import CenterMethod, FrameMethod
from rfd3_mosaic.structure import (
    AtomRecord,
    load_selected_atoms,
    select_atom_subset,
)


ExpansionRecord: TypeAlias = tuple[
    str | None,
    str | None,
    int,
    str,
    TransformMatrix,
]


def _matrix_to_tuple(matrix: np.ndarray) -> TransformMatrix:
    return tuple(
        tuple(float(value) for value in row)
        for row in matrix
    )  # type: ignore[return-value]


def _instance_id(
    source_id: str,
    orbit_id: str | None,
    copy_index: int,
) -> str:
    scope = orbit_id if orbit_id is not None else "identity"
    return f"{source_id}@{scope}[{copy_index}]"


def _atom_coordinates(atoms: tuple[AtomRecord, ...]) -> np.ndarray:
    return np.asarray(
        [atom.coordinate for atom in atoms],
        dtype=np.float64,
    )


def load_interface_seed_config(
    config_path: str | Path,
) -> InterfaceSeedSpec:
    """Load and validate an Interface-Seed configuration."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Interface-Seed config does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Interface-Seed config must contain a YAML mapping"
        )

    payload = raw_config.get("interface_seed", raw_config)

    if not isinstance(payload, dict):
        raise ValueError(
            "The interface_seed field must contain a YAML mapping"
        )

    return InterfaceSeedSpec.model_validate(payload)


def expand_symmetry_instances(
    spec: InterfaceSeedSpec,
    *,
    master_transforms: dict[str, np.ndarray] | None = None,
) -> CompiledInstanceSet:
    """Expand specs into deterministic motion-group, fragment, and port copies."""

    registries: dict[str, SymmetryTransformRegistry] = {
        transform_set_id: build_transform_registry(transform_set_spec)
        for transform_set_id, transform_set_spec
        in spec.symmetry.transform_sets.items()
    }
    orbit_registries = {
        orbit_id: registries[orbit.transform_set]
        for orbit_id, orbit in spec.symmetry.orbits.items()
    }

    def resolve_copy_relation(
        orbit_id: str | None,
        source_copy_index: int,
        *,
        orbit_offset: int | None,
        transform: str | None,
    ) -> int:
        if orbit_id is None:
            if orbit_offset != 0 or transform is not None:
                raise ValueError(
                    "An unsymmetrized object only supports orbit_offset 0"
                )
            return 0
        registry = orbit_registries[orbit_id]
        if orbit_offset is not None:
            target_transform_id = registry.transform_id_for_offset(
                orbit_offset,
                source_copy_index=source_copy_index,
            )
        else:
            assert transform is not None
            target_transform_id = registry.transform_id_for_relation(
                transform,
                source_copy_index=source_copy_index,
            )
        return registry.transform_ids.index(target_transform_id)

    group_to_orbit: dict[str, str] = {}
    for orbit_id, orbit in spec.symmetry.orbits.items():
        for group_id in orbit.master_groups:
            group_to_orbit[group_id] = orbit_id

    identity_matrix = np.eye(4, dtype=np.float64)
    provided_master_transforms = master_transforms or {}
    unknown_groups = set(provided_master_transforms) - set(spec.motion_groups)
    if unknown_groups:
        raise ValueError(
            f"Master transforms reference unknown groups: "
            f"{sorted(unknown_groups)}"
        )
    expansions: dict[str, list[ExpansionRecord]] = {}
    for group_id in spec.motion_groups:
        master_transform = validate_transform(
            provided_master_transforms.get(group_id, identity_matrix)
        )
        orbit_id = group_to_orbit.get(group_id)
        if orbit_id is None:
            expansions[group_id] = [
                (
                    None,
                    None,
                    0,
                    "E:e",
                    _matrix_to_tuple(master_transform),
                )
            ]
            continue

        orbit = spec.symmetry.orbits[orbit_id]
        registry = registries[orbit.transform_set]
        expansions[group_id] = [
            (
                orbit_id,
                orbit.transform_set,
                copy_index,
                transform_id,
                _matrix_to_tuple(
                    compose_transforms(
                        registry.transform(transform_id),
                        master_transform,
                    )
                ),
            )
            for copy_index, transform_id in enumerate(registry.transform_ids)
        ]

    motion_group_instances: dict[str, MotionGroupInstance] = {}
    fragment_instances: dict[str, FragmentInstance] = {}
    fragment_index: dict[tuple[str, str | None, int], str] = {}

    for group_id, group_spec in spec.motion_groups.items():
        for (
            orbit_id,
            transform_set_id,
            copy_index,
            transform_id,
            transform,
        ) in expansions[group_id]:
            group_instance_id = _instance_id(
                group_id,
                orbit_id,
                copy_index,
            )
            group_fragment_ids: list[str] = []
            for fragment_id in group_spec.members:
                fragment_instance_id = _instance_id(
                    fragment_id,
                    orbit_id,
                    copy_index,
                )
                group_fragment_ids.append(fragment_instance_id)
                fragment_index[(fragment_id, orbit_id, copy_index)] = (
                    fragment_instance_id
                )
                fragment_instances[fragment_instance_id] = FragmentInstance(
                    id=fragment_instance_id,
                    source_id=fragment_id,
                    motion_group_instance_id=group_instance_id,
                    orbit_id=orbit_id,
                    transform_set_id=transform_set_id,
                    copy_index=copy_index,
                    transform_id=transform_id,
                    transform=transform,
                )

            motion_group_instances[group_instance_id] = MotionGroupInstance(
                id=group_instance_id,
                source_id=group_id,
                fragment_instance_ids=tuple(group_fragment_ids),
                orbit_id=orbit_id,
                transform_set_id=transform_set_id,
                copy_index=copy_index,
                transform_id=transform_id,
                transform=transform,
            )

    port_instances: dict[str, InterfacePortInstance] = {}
    port_index: dict[tuple[str, str | None, int], str] = {}
    for port_id, port_spec in spec.ports.items():
        for (
            orbit_id,
            transform_set_id,
            copy_index,
            transform_id,
            transform,
        ) in expansions[port_spec.group]:
            port_instance_id = _instance_id(
                port_id,
                orbit_id,
                copy_index,
            )
            group_instance_id = _instance_id(
                port_spec.group,
                orbit_id,
                copy_index,
            )
            port_fragment_ids = tuple(
                fragment_index[(fragment_id, orbit_id, copy_index)]
                for fragment_id in port_spec.fragments
            )
            port_instances[port_instance_id] = InterfacePortInstance(
                id=port_instance_id,
                source_id=port_id,
                motion_group_instance_id=group_instance_id,
                fragment_instance_ids=port_fragment_ids,
                orbit_id=orbit_id,
                transform_set_id=transform_set_id,
                copy_index=copy_index,
                transform_id=transform_id,
                transform=transform,
            )
            port_index[(port_id, orbit_id, copy_index)] = port_instance_id

    interface_instances: dict[str, InterfaceEdgeInstance] = {}
    for edge_id, edge_spec in spec.interfaces.items():
        left_group_id = spec.ports[edge_spec.left_port].group
        right_group_id = spec.ports[edge_spec.right_port].group
        left_copy_keys = [
            (record[0], record[2]) for record in expansions[left_group_id]
        ]
        right_copy_keys = {
            (record[0], record[2]) for record in expansions[right_group_id]
        }
        for orbit_id, source_copy_index in left_copy_keys:
            target_copy_index = resolve_copy_relation(
                orbit_id,
                source_copy_index,
                orbit_offset=edge_spec.copy_relation.orbit_offset,
                transform=edge_spec.copy_relation.transform,
            )
            if (orbit_id, target_copy_index) not in right_copy_keys:
                raise ValueError(
                    f"Interface {edge_id!r} connects incompatible symmetry "
                    "orbits"
                )
            edge_instance_id = _instance_id(
                edge_id,
                orbit_id,
                source_copy_index,
            )
            interface_instances[edge_instance_id] = InterfaceEdgeInstance(
                id=edge_instance_id,
                source_id=edge_id,
                left_port_instance_id=port_index[
                    (edge_spec.left_port, orbit_id, source_copy_index)
                ],
                right_port_instance_id=port_index[
                    (edge_spec.right_port, orbit_id, target_copy_index)
                ],
                required=edge_spec.required,
                target_geometry=edge_spec.target_geometry,
                orbit_id=orbit_id,
                source_copy_index=source_copy_index,
                target_copy_index=target_copy_index,
            )

    fragment_to_group: dict[str, str] = {}
    for group_id, group_spec in spec.motion_groups.items():
        for fragment_id in group_spec.members:
            fragment_to_group[fragment_id] = group_id

    scaffold_link_instances: dict[str, ScaffoldLinkInstance] = {}
    for link_id, link_spec in spec.scaffold_links.items():
        from_fragment_id = link_spec.from_endpoint.fragment
        to_fragment_id = link_spec.to_endpoint.fragment
        from_group_id = fragment_to_group[from_fragment_id]
        to_group_id = fragment_to_group[to_fragment_id]
        from_expansions = expansions[from_group_id]
        to_expansions = expansions[to_group_id]
        from_copy_keys = [
            (record[0], record[2]) for record in from_expansions
        ]
        to_copy_keys = {
            (record[0], record[2]) for record in to_expansions
        }
        for orbit_id, copy_index in from_copy_keys:
            target_copy_index = resolve_copy_relation(
                orbit_id,
                copy_index,
                orbit_offset=link_spec.copy_relation.orbit_offset,
                transform=link_spec.copy_relation.transform,
            )
            if (orbit_id, target_copy_index) not in to_copy_keys:
                raise ValueError(
                    f"Scaffold link {link_id!r} connects fragments with "
                    "incompatible symmetry expansions"
                )
            link_instance_id = _instance_id(
                link_id,
                orbit_id,
                copy_index,
            )
            scaffold_link_instances[link_instance_id] = (
                ScaffoldLinkInstance(
                    id=link_instance_id,
                    source_id=link_id,
                    from_fragment_instance_id=fragment_index[
                        (from_fragment_id, orbit_id, copy_index)
                    ],
                    from_terminus=link_spec.from_endpoint.terminus,
                    to_fragment_instance_id=fragment_index[
                        (to_fragment_id, orbit_id, target_copy_index)
                    ],
                    to_terminus=link_spec.to_endpoint.terminus,
                    minimum_length=link_spec.length.minimum,
                    maximum_length=link_spec.length.maximum,
                    tie_group=link_spec.tie_group,
                    chain_break=link_spec.chain_break,
                    orbit_id=orbit_id,
                    copy_index=copy_index,
                    target_copy_index=target_copy_index,
                )
            )

    return CompiledInstanceSet(
        motion_groups=motion_group_instances,
        fragments=fragment_instances,
        ports=port_instances,
        interfaces=interface_instances,
        scaffold_links=scaffold_link_instances,
    )


def _fixed_xyz_rotation(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rotation_deg)
    rotate_x = axis_angle_rotation((1.0, 0.0, 0.0), float(rx))
    rotate_y = axis_angle_rotation((0.0, 1.0, 0.0), float(ry))
    rotate_z = axis_angle_rotation((0.0, 0.0, 1.0), float(rz))
    return rotate_z @ rotate_y @ rotate_x


def _unit_interval_value(value: float, *, name: str) -> float:
    unit = float(value)
    if not 0.0 <= unit <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {unit}")
    return unit


def _sample_range(
    mean: float,
    variation: float,
    rng: np.random.Generator,
    *,
    unit_value: float | None = None,
) -> float:
    if variation == 0.0:
        return mean
    unit = (
        float(rng.random())
        if unit_value is None
        else _unit_interval_value(unit_value, name="Range unit sample")
    )
    return float(mean - variation + 2.0 * variation * unit)


def _uniform_so3_rotation(
    rng: np.random.Generator,
    *,
    unit_values: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Haar-uniform SO(3) rotation using a unit quaternion.

    Returns the rotation and the quaternion in ``(x, y, z, w)`` order.
    The Shoemake construction avoids the pole bias of independently sampled
    Euler angles.
    """

    if unit_values is None:
        u1, u2, u3 = (float(value) for value in rng.random(3))
    else:
        u1, u2, u3 = (
            _unit_interval_value(value, name="SO(3) unit sample")
            for value in unit_values
        )
    quaternion = np.asarray(
        (
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ),
        dtype=np.float64,
    )
    x, y, z, w = quaternion
    rotation = np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )
    return rotation, quaternion


def _principal_axis(coordinates: np.ndarray) -> np.ndarray | None:
    """Return a deterministic longest PCA axis, or None when degenerate."""

    centered = coordinates - coordinates.mean(axis=0)
    covariance = centered.T @ centered / float(coordinates.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    leading = float(eigenvalues[order[0]])
    second = float(eigenvalues[order[1]])
    if leading <= 1e-12 or leading - second <= 1e-8 * max(leading, 1.0):
        return None
    axis = eigenvectors[:, order[0]]
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis *= -1.0
    return axis / np.linalg.norm(axis)


def build_master_group_transforms(
    spec: InterfaceSeedSpec,
    *,
    base_directory: str | Path = ".",
    random_seed: int | None = None,
    sample_metadata: dict[str, Any] | None = None,
    sample_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Resolve explicit SE(3) master placement before symmetry expansion."""

    fragment_atoms = {
        fragment_id: load_selected_atoms(
            fragment_spec,
            base_directory=base_directory,
        )
        for fragment_id, fragment_spec in spec.fragments.items()
    }
    group_to_transform_set: dict[str, str] = {}
    for orbit in spec.symmetry.orbits.values():
        for group_id in orbit.master_groups:
            group_to_transform_set[group_id] = orbit.transform_set

    effective_seed = spec.random_seed if random_seed is None else random_seed
    rng = np.random.default_rng(effective_seed)
    transforms: dict[str, np.ndarray] = {}
    for group_id, initialization in spec.initialization.items():
        group_override = (sample_overrides or {}).get(group_id, {})
        group = spec.motion_groups[group_id]
        atoms = tuple(
            atom
            for fragment_id in group.members
            for atom in fragment_atoms[fragment_id]
        )
        if not atoms:
            raise ValueError(
                f"Motion group {group_id!r} has no atoms for initialization"
            )
        coordinates = _atom_coordinates(atoms)
        if initialization.center_method == CenterMethod.NONE:
            source_center = np.zeros(3, dtype=np.float64)
        else:
            heavy_atoms = select_atom_subset(atoms, "heavy")
            source_center = _atom_coordinates(heavy_atoms).mean(axis=0)

        transform_set_id = group_to_transform_set.get(group_id)
        if transform_set_id is None:
            axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        else:
            axis = np.asarray(
                spec.symmetry.transform_sets[transform_set_id].axis,
                dtype=np.float64,
            )
        axis /= np.linalg.norm(axis)
        requested_radial = np.asarray(
            initialization.placement.radial_direction,
            dtype=np.float64,
        )
        radial = requested_radial - np.dot(requested_radial, axis) * axis
        radial_norm = np.linalg.norm(radial)
        if radial_norm <= 1e-8:
            raise ValueError(
                f"Initialization radial direction for {group_id!r} is "
                "parallel to the symmetry axis"
            )
        radial /= radial_norm
        radius = _sample_range(
            initialization.placement.radius.mean,
            initialization.placement.radius.range,
            rng,
            unit_value=group_override.get("radius_unit"),
        )
        axial_offset = _sample_range(
            initialization.placement.axial_offset.mean,
            initialization.placement.axial_offset.range,
            rng,
            unit_value=group_override.get("axial_offset_unit"),
        )
        target_center = radial * radius + axis * axial_offset
        quaternion: np.ndarray | None = None
        if initialization.orientation.method == "fixed":
            rotation = _fixed_xyz_rotation(
                initialization.orientation.rotation_deg
            )
        else:
            so3_unit = group_override.get("so3_unit")
            rotation, quaternion = _uniform_so3_rotation(
                rng,
                unit_values=(
                    tuple(float(value) for value in so3_unit)
                    if so3_unit is not None
                    else None
                ),
            )
        translation = target_center - rotation @ source_center
        transforms[group_id] = make_transform(rotation, translation)
        if sample_metadata is not None:
            principal_axis_source = _principal_axis(coordinates)
            principal_axis_world = (
                rotation @ principal_axis_source
                if principal_axis_source is not None
                else None
            )
            principal_axis_tilt = (
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                abs(np.dot(principal_axis_world, axis)),
                                0.0,
                                1.0,
                            )
                        )
                    )
                )
                if principal_axis_world is not None
                else None
            )
            sample_metadata[group_id] = {
                "random_seed": effective_seed,
                "orientation_method": initialization.orientation.method,
                "quaternion_xyzw": (
                    quaternion.tolist() if quaternion is not None else None
                ),
                "rotation_matrix": rotation.tolist(),
                "unit_samples": {
                    "radius": group_override.get("radius_unit"),
                    "axial_offset": group_override.get(
                        "axial_offset_unit"
                    ),
                    "so3": group_override.get("so3_unit"),
                },
                "sampled_radius": radius,
                "sampled_axial_offset": axial_offset,
                "radial_direction": radial.tolist(),
                "source_center": source_center.tolist(),
                "target_center": target_center.tolist(),
                "principal_axis_source": (
                    principal_axis_source.tolist()
                    if principal_axis_source is not None
                    else None
                ),
                "principal_axis_world": (
                    principal_axis_world.tolist()
                    if principal_axis_world is not None
                    else None
                ),
                "principal_axis_tilt_deg": principal_axis_tilt,
            }

    return transforms


def resolve_reference_port_frames(
    spec: InterfaceSeedSpec,
    *,
    base_directory: str | Path = ".",
) -> dict[str, TransformMatrix]:
    """Resolve master-copy port frames directly from reference structures."""

    fragment_atoms = {
        fragment_id: load_selected_atoms(
            fragment_spec,
            base_directory=base_directory,
        )
        for fragment_id, fragment_spec in spec.fragments.items()
    }
    port_atoms: dict[str, tuple[AtomRecord, ...]] = {}
    for port_id, port_spec in spec.ports.items():
        contributed = tuple(
            atom
            for fragment_id in port_spec.fragments
            for atom in fragment_atoms[fragment_id]
        )
        port_atoms[port_id] = select_atom_subset(
            contributed,
            port_spec.atoms,
        )

    partners: dict[str, set[str]] = {
        port_id: set() for port_id in spec.ports
    }
    for edge in spec.interfaces.values():
        partners[edge.left_port].add(edge.right_port)
        partners[edge.right_port].add(edge.left_port)

    resolved: dict[str, TransformMatrix] = {}
    for port_id, port_spec in spec.ports.items():
        frame_spec = port_spec.frame
        if frame_spec.method == FrameMethod.PRECOMPUTED:
            transform = validate_transform(frame_spec.transform)
        elif frame_spec.method == FrameMethod.REFERENCE_INTERFACE_PCA:
            partner_ids = partners[port_id]
            if len(partner_ids) > 1:
                raise ValueError(
                    f"Port {port_id!r} has multiple reference partners; "
                    "define a separate port for each interface"
                )
            partner_coordinates = None
            if partner_ids:
                partner_id = next(iter(partner_ids))
                partner_coordinates = _atom_coordinates(
                    port_atoms[partner_id]
                )
            transform = reference_interface_pca_frame(
                _atom_coordinates(port_atoms[port_id]),
                partner_coordinates=partner_coordinates,
            )
        else:
            raise NotImplementedError(
                f"Port frame method {frame_spec.method.value!r} is not "
                "implemented by the reference resolver yet"
            )
        resolved[port_id] = _matrix_to_tuple(transform)

    return resolved
