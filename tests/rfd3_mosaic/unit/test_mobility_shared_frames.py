"""Motion proposals and symmetry projection must use one proper frame registry."""

import copy

import pytest
import torch
from rfd3.inference.symmetry.motif_mobility import OrbitRigidMotifController
from rfd3.inference.symmetry.symmetry_utils import (
    build_symmetry_orbit_layout,
    normalize_symmetry_transforms,
    project_symmetry_orbit_average,
)

from rfd3_mosaic.geometry import (
    build_cyclic_registry,
    build_dihedral_registry,
    build_polyhedral_registry,
)
from rfd3_mosaic.schema.specs import SymmetryType


def _case(registry=None, *, dtype=torch.float32, frames_dtype=None):
    """Two mobile motifs, sparse IDs and a nonidentity ASU around a shifted axis."""
    registry = registry or build_cyclic_registry(3)
    count = registry.order
    ids = [7 + 13 * i for i in range(count)]
    center = torch.tensor([3.0, -2.0, 1.0], dtype=dtype)
    canonical = torch.tensor(
        [
            [30.0, 2.0, 18.0],
            [31.5, 2.1, 18.2],
            [30.3, 3.2, 19.0],
            [22.0, -4.0, -15.0],
            [23.0, -3.0, -15.0],
            [22.0, -3.0, -13.0],
        ],
        dtype=dtype,
    )
    transforms = {}
    for index, name in enumerate(registry.transform_ids):
        r = torch.tensor(registry.transform(name)[:3, :3], dtype=dtype)
        # Reproduce native virtual-frame scale/shear (~6e-6 orthogonality).
        if index:
            r = r @ torch.diag(
                torch.tensor([1.0 - 2e-6, 1.0 - 3e-6, 1.0 - 3e-6], dtype=dtype)
            )
        t = (
            center
            - torch.tensor(registry.transform(name)[:3, :3], dtype=dtype) @ center
        )
        transforms[str(ids[index])] = (
            r.to(frames_dtype or dtype),
            t.to(frames_dtype or dtype),
        )
    features = {
        "sym_transform": transforms,
        "sym_entity_id": torch.zeros(count * 6, dtype=torch.long),
        "sym_transform_id": torch.tensor(ids).repeat_interleave(6),
        "is_sym_asu": torch.arange(count).repeat_interleave(6) == min(1, count - 1),
        "sym_orbit_slot": torch.arange(6).repeat(count),
        "sym_orbit_slot_verified": torch.tensor(True),
        "motif_constraint_group_orbit_index": torch.tensor(
            [i for i in range(2) for _ in ids]
        ),
        "motif_constraint_group_orbit_transform_id": torch.tensor(ids * 2),
        "motif_constraint_group_atom_indices": torch.tensor(
            [
                [copy * 6 + motif * 3 + atom for atom in range(3)]
                for motif in range(2)
                for copy in range(count)
            ]
        ),
        "motif_constraint_group_atom_mask": torch.ones(
            (2 * count, 3), dtype=torch.bool
        ),
        "motif_constraint_orbit_master_group_index": torch.tensor([0, count]),
        "motif_constraint_orbit_mobility_mode": torch.tensor([1, 1]),
        "motif_constraint_orbit_bounds": torch.tensor([[2.0, 10.0], [2.0, 10.0]]),
    }
    placeholder = canonical.repeat(count, 1)[None]
    layout = build_symmetry_orbit_layout(features, like=placeholder)
    target = torch.cat(
        [canonical @ r.T + t for r, t in layout.sym_transforms.values()]
    )[None]
    return features, target, layout


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_c3_frames_remove_native_scale_drift_without_moving_the_master(dtype):
    features, target, layout = _case(dtype=dtype)
    controller = OrbitRigidMotifController.from_features(
        features,
        target,
        normalized_symmetry_transforms=layout.sym_transforms,
    )
    assert controller is not None
    for key, (r, t) in layout.sym_transforms.items():
        assert controller.sym_transforms[key][0] is r
        assert controller.sym_transforms[key][1] is t
    actual = controller.materialize_target()
    limit = 4e-6 if dtype == torch.float32 else 1e-12
    assert float((actual - target).abs().max()) < limit
    # The historical mapping demonstrably adds motion at zero pose.
    old_frames = controller.sym_transforms
    controller.sym_transforms = {
        int(k): v for k, v in features["sym_transform"].items()
    }
    historical = controller.materialize_target()
    assert float((historical - target).abs().max()) > 4e-5
    controller.sym_transforms = old_frames
    for motif in controller.motifs:
        master = motif.master_atom_indices
        torch.testing.assert_close(
            actual[:, master], target[:, master], rtol=0, atol=limit
        )


@pytest.mark.parametrize(
    "registry",
    [
        build_cyclic_registry(3),
        build_dihedral_registry(3),
        *(
            build_polyhedral_registry(group)
            for group in (
                SymmetryType.TETRAHEDRAL,
                SymmetryType.OCTAHEDRAL,
                SymmetryType.ICOSAHEDRAL,
            )
        ),
    ],
)
def test_shared_registry_keeps_nonzero_motion_on_projector_orbits(registry):
    features, target, layout = _case(registry, dtype=torch.float64)
    controller = OrbitRigidMotifController.from_features(
        features,
        target,
        normalized_symmetry_transforms=layout.sym_transforms,
    )
    angle = torch.tensor(0.04, dtype=target.dtype)
    rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, angle.cos(), -angle.sin()],
            [0.0, angle.sin(), angle.cos()],
        ],
        dtype=target.dtype,
    )
    for index, motif in enumerate(controller.motifs):
        motif.state.rotation[0] = rotation
        motif.state.translation[0] = target.new_tensor([0.1, -0.2, 0.05]) * (index + 1)
    moved = controller.materialize_target()
    projected = project_symmetry_orbit_average(moved, features, layout=layout)
    torch.testing.assert_close(projected, moved, rtol=0, atol=1e-12)
    assert float((moved - target).abs().max()) > 0.1


def test_standalone_normalizes_bfloat16_frames_without_orbit_slot_features():
    features, target, layout = _case(frames_dtype=torch.bfloat16)
    for key in (
        "sym_entity_id",
        "sym_transform_id",
        "is_sym_asu",
        "sym_orbit_slot",
        "sym_orbit_slot_verified",
    ):
        features.pop(key)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        controller = OrbitRigidMotifController.from_features(features, target)
    for key, (r, t) in controller.sym_transforms.items():
        assert r.dtype == t.dtype == torch.float32
        torch.testing.assert_close(r, layout.sym_transforms[key][0], rtol=0, atol=0)
        torch.testing.assert_close(t, layout.sym_transforms[key][1], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["reflection", "shear", "nan", "shape", "alias"])
def test_invalid_raw_frames_still_fail_closed(invalid):
    features, target, _ = _case()
    transforms = copy.deepcopy(features["sym_transform"])
    key = next(iter(transforms))
    r, t = transforms[key]
    if invalid == "reflection":
        r[0, 0] = -1
    if invalid == "shear":
        r[0, 1] = 0.2
    if invalid == "nan":
        t[0] = float("nan")
    if invalid == "shape":
        transforms[key] = (r[:2], t)
    if invalid == "alias":
        transforms[int(key)] = transforms[key]
    with pytest.raises(ValueError):
        normalize_symmetry_transforms(transforms, like=target)


def test_shared_registry_requires_same_transform_ids_and_preserves_fixed_sentinel():
    features, target, layout = _case()
    wrong = dict(layout.sym_transforms)
    wrong[999] = wrong.pop(next(iter(wrong)))
    with pytest.raises(ValueError, match="transform IDs"):
        OrbitRigidMotifController.from_features(
            features, target, normalized_symmetry_transforms=wrong
        )
    with_fixed = dict(layout.sym_transforms)
    with_fixed[-1] = (torch.eye(3), torch.tensor([1.0, 2.0, 3.0]))
    actual = normalize_symmetry_transforms(
        with_fixed, like=target, already_normalized=True
    )
    assert set(actual) == set(layout.sym_transforms)
    for key in actual:
        assert actual[key][0] is layout.sym_transforms[key][0]
