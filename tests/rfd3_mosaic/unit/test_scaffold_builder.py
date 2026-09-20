"""A CPU backbone witness must satisfy geometry, not just optimizer status."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

from rfd3_mosaic import scaffold_builder as builder


def _reference(length=8, *, compact=False):
    # A deterministic ideal-coordinate fixture, rather than a deposited PDB.
    left = np.array(
        [
            [-1.458, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.525 * np.cos(np.deg2rad(68.8)), 1.525 * np.sin(np.deg2rad(68.8)), 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    left[3] = builder._place_atom(*left[:3], 1.229, np.deg2rad(120.8), np.deg2rad(135))
    torsions = np.tile(np.deg2rad([-60.0, -45.0]), length + 1)[: 2 * length + 1]
    if compact:
        assert length == 20
        # Two eight-residue helices separated by a four-residue return turn.
        torsions[16:24] = np.deg2rad(
            [
                -2.6043840119,
                -177.8295094939,
                81.1105167073,
                151.2715726321,
                2.5179819157,
                130.4470157504,
                -26.5721584091,
                -120.9331641067,
            ]
        )
    xyz = builder._ChainModel.from_anchors(left, left, length).forward(torsions)[0]
    xyz[-1, 3] = builder._place_atom(
        *xyz[-1, :3], 1.229, np.deg2rad(120.8), np.deg2rad(135)
    )
    return xyz


def _assert_geometry(xyz, left, right, length):
    assert xyz.shape == (length + 2, 4, 3)
    assert np.isfinite(xyz).all()
    np.testing.assert_array_equal(xyz[0], left)
    np.testing.assert_array_equal(xyz[-1], right)
    # Check directly from returned Cartesian atoms, including both junctions.
    np.testing.assert_allclose(
        np.linalg.norm(xyz[1:, 0] - xyz[:-1, 2], axis=1), 1.329, atol=0.04
    )
    np.testing.assert_allclose(
        np.linalg.norm(xyz[1:-1, 1] - xyz[1:-1, 0], axis=1), 1.458, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(xyz[1:-1, 2] - xyz[1:-1, 1], axis=1), 1.525, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(xyz[1:-1, 3] - xyz[1:-1, 2], axis=1), 1.229, atol=1e-10
    )
    for before, after in zip(xyz[:-1], xyz[1:]):
        a, b = before[1] - before[2], after[0] - before[2]
        angle = np.rad2deg(
            np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))
        )
        assert abs(angle - 116.2) <= 8.0
        # Normal vectors to the trans peptide's two planes must be opposite.
        first = np.cross(before[2] - before[1], after[0] - before[2])
        second = np.cross(after[0] - before[2], after[1] - after[0])
        assert first @ second / np.linalg.norm(first) / np.linalg.norm(second) < -0.98


@pytest.mark.parametrize("length", [0, 8])
def test_reference_reconstruction_preserves_all_anchor_atoms(length):
    reference = _reference(length)
    original = reference.copy()
    xyz, report = builder.repair_backbone(
        reference, reference[0], reference[-1], generated_length=length, attempts=1
    )
    assert report["status"] == "verified_backbone_geometry"
    _assert_geometry(xyz, reference[0], reference[-1], length)
    np.testing.assert_allclose(xyz, reference, atol=1e-8)
    np.testing.assert_array_equal(reference, original)


def test_changed_target_is_closed_without_moving_fixed_residues():
    reference = _reference(12)
    target = reference[-1] + np.array([0.15, -0.1, 0.1])
    xyz, report = builder.repair_backbone(
        reference,
        reference[0],
        target,
        generated_length=12,
        attempts=2,
        max_nfev=250,
    )
    assert xyz is not None, report
    _assert_geometry(xyz, reference[0], target, 12)
    assert np.max(np.linalg.norm(xyz[1:-1] - reference[1:-1], axis=-1)) > 0.01
    assert (
        report["attempts"][report["selected_attempt"]][
            "maximum_anchor_closure_error_angstrom"
        ]
        <= 0.005
    )


@pytest.mark.parametrize("shape_weight", [0.0, 1.0])
def test_reference_repair_is_rigid_frame_equivariant(shape_weight):
    reference = _reference(12)
    target = reference[-1] + np.array([0.1, -0.05, 0.07])
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    translation = np.array([81.0, -19.0, 4.0])
    first, first_report = builder.repair_backbone(
        reference,
        reference[0],
        target,
        generated_length=12,
        attempts=1,
        max_nfev=250,
        reference_shape_weight=shape_weight,
        maximum_reference_ca_deviation=2.0,
    )
    transformed = reference @ rotation.T + translation
    second, second_report = builder.repair_backbone(
        transformed,
        transformed[0],
        target @ rotation.T + translation,
        generated_length=12,
        attempts=1,
        max_nfev=250,
        reference_shape_weight=shape_weight,
        maximum_reference_ca_deviation=2.0,
    )
    assert first is not None, first_report
    assert second is not None, second_report
    np.testing.assert_allclose(second, first @ rotation.T + translation, atol=2e-5)


def test_explicit_hl_constructs_full_backbone_without_reference():
    expected = _reference(8)
    xyz, report = builder.repair_backbone(
        None,
        expected[0],
        expected[-1],
        generated_length=8,
        secondary_structure="H" * 8,
        attempts=1,
    )
    assert xyz is not None, report
    assert report["initialization"] == "explicit_HL_internal_coordinates"
    _assert_geometry(xyz, expected[0], expected[-1], 8)


def test_explicit_block_rebuild_has_requested_length():
    old, target = _reference(8), _reference(10)
    xyz, report = builder.repair_backbone(
        old,
        target[0],
        target[-1],
        generated_length=10,
        secondary_structure="H" * 10,
        reference_secondary_structure="H" * 8,
        attempts=1,
    )
    assert xyz is not None, report
    assert report["initialization"] == "explicit_HL_block_torsion_rebuild"
    _assert_geometry(xyz, target[0], target[-1], 10)


def test_compact_reference_retains_measured_nonlocal_contacts():
    reference = _reference(20, compact=True)
    xyz, report = builder.repair_backbone(
        reference, reference[0], reference[-1], generated_length=20, attempts=1
    )
    assert xyz is not None, report
    assert report["reference_packing_contract"]
    assert report["reference_contact_count"] > 0
    for (i, j), distance in zip(
        report["reference_contact_pairs"],
        report["reference_contact_distances_angstrom"],
    ):
        assert j - i >= 8
        assert distance <= 8.0
        assert abs(np.linalg.norm(xyz[i, 1] - xyz[j, 1]) - distance) <= 2.0
    _assert_geometry(xyz, reference[0], reference[-1], 20)


def test_shape_bound_rejects_despite_successful_closure_and_geometry():
    reference = _reference(8)
    xyz, report = builder.repair_backbone(
        reference,
        reference[0],
        reference[-1],
        generated_length=8,
        attempts=1,
        reference_ca_target=reference[1:-1, 1] + [3.0, 0.0, 0.0],
        maximum_reference_ca_deviation=1.0,
        reference_shape_weight=0.0,
    )
    assert xyz is None
    record = report["attempts"][0]
    assert record["geometry"]["passed"]
    assert record["maximum_anchor_closure_error_angstrom"] < 0.005
    assert record["generated_ca_maximum_displacement_angstrom"] == pytest.approx(3.0)
    assert not record["accepted"]


def test_shape_objective_reduces_contact_underconstrained_fold_drift():
    reference = _reference(20, compact=True)
    target = reference[-1] + [0.2, -0.1, 0.1]
    common = dict(generated_length=20, attempts=1, max_nfev=200)
    free, free_report = builder.repair_backbone(
        reference, reference[0], target, **common
    )
    shaped, shaped_report = builder.repair_backbone(
        reference,
        reference[0],
        target,
        reference_shape_weight=1.0,
        maximum_reference_ca_deviation=0.3,
        **common,
    )
    assert free is not None, free_report
    assert shaped is not None, shaped_report
    free_rms = np.sqrt(
        np.mean(np.sum((free[1:-1, 1] - reference[1:-1, 1]) ** 2, axis=1))
    )
    shaped_rms = np.sqrt(
        np.mean(np.sum((shaped[1:-1, 1] - reference[1:-1, 1]) ** 2, axis=1))
    )
    assert shaped_rms < free_rms / 2
    assert shaped_report["generated_ca_maximum_displacement_angstrom"] <= 0.3
    _assert_geometry(shaped, reference[0], target, 20)


def test_resized_reference_shape_needs_explicit_new_length_target():
    old, target = _reference(8), _reference(10)
    args = dict(
        generated_length=10,
        secondary_structure="H" * 10,
        reference_secondary_structure="H" * 8,
        attempts=1,
        reference_shape_weight=1.0,
        maximum_reference_ca_deviation=1.0,
    )
    with pytest.raises(ValueError, match="explicit reference_ca_target"):
        builder.repair_backbone(old, target[0], target[-1], **args)
    xyz, report = builder.repair_backbone(
        old,
        target[0],
        target[-1],
        reference_ca_target=target[1:-1, 1],
        **args,
    )
    assert xyz is not None, report
    assert report["reference_shape_target_source"] == "explicit_generated_CA_target"
    assert report["generated_ca_maximum_displacement_angstrom"] < 1e-8
    _assert_geometry(xyz, target[0], target[-1], 10)


def test_analytic_torsion_jacobian_includes_oxygen_and_terminal_frame():
    reference = _reference(8)
    model = builder._ChainModel.from_anchors(reference[0], reference[-1], 8)
    q = builder._reference_torsions(reference)
    xyz, origins, axes, affected = model.forward(q)
    indices = np.array([0, 4, 7, 9, 15, 4 * 9, 4 * 9 + 1, 4 * 9 + 2, 4 * 9 + 3])
    analytic = builder._point_jacobian(xyz, origins, axes, affected, indices)
    finite = np.empty_like(analytic)
    for index in range(len(q)):
        offset = np.zeros_like(q)
        offset[index] = 1e-6
        plus = model.forward(q + offset)[0].reshape(-1, 3)[indices]
        minus = model.forward(q - offset)[0].reshape(-1, 3)[indices]
        finite[:, :, index] = (plus - minus) / 2e-6
    np.testing.assert_allclose(analytic, finite, atol=2e-7, rtol=1e-7)
    np.testing.assert_array_equal(analytic[0], 0.0)


def _colliding_reference():
    reference = _reference(20, compact=True)
    q = builder._reference_torsions(reference)
    q[19] += np.deg2rad(30.0)
    xyz = builder._ChainModel.from_anchors(reference[0], reference[-1], 20).forward(q)[
        0
    ]
    xyz[-1, 3] = builder._place_atom(
        *xyz[-1, :3], 1.229, np.deg2rad(120.8), np.deg2rad(135)
    )
    return xyz


def test_nonlocal_repulsion_repairs_collision_missing_from_contact_objective():
    reference = _colliding_reference()
    common = dict(generated_length=20, attempts=1, max_nfev=150)
    without, old_report = builder.repair_backbone(
        reference,
        reference[0],
        reference[-1],
        nonlocal_clash_weight=0.0,
        **common,
    )
    assert without is None
    old = old_report["attempts"][0]
    assert old["maximum_anchor_closure_error_angstrom"] < 0.005
    assert old["maximum_reference_contact_error_angstrom"] < 1e-8
    assert old["geometry"]["minimum_nonlocal_backbone_distance_angstrom"] < 1.9
    repaired, report = builder.repair_backbone(
        reference, reference[0], reference[-1], **common
    )
    assert repaired is not None, report
    assert report["nonlocal_clash_weight_per_angstrom_squared"] == 0.25
    assert report["nonlocal_clash_target_distance_angstrom"] == 2.1
    assert (
        report["attempts"][0]["geometry"]["minimum_nonlocal_backbone_distance_angstrom"]
        >= 2.0
    )
    _assert_geometry(repaired, reference[0], reference[-1], 20)


def test_repulsion_does_not_waive_final_clash_cutoff_when_budget_exhausted():
    reference = _colliding_reference()
    result, report = builder.repair_backbone(
        reference,
        reference[0],
        reference[-1],
        generated_length=20,
        attempts=1,
        max_nfev=1,
    )
    assert result is None
    assert report["status"] == "unresolved"
    assert not report["attempts"][0]["geometry"]["passed"]


def test_clash_nearest_pair_jacobian_uses_immutable_final_right_anchor():
    reference = _colliding_reference()
    model = builder._ChainModel.from_anchors(reference[0], reference[-1], 20)
    q = builder._reference_torsions(reference) + np.random.default_rng(113).normal(
        0.0, 0.03, 41
    )
    pairs = np.triu_indices(len(reference), k=2)

    def evaluate(torsions):
        xyz, origins, axes, affected = model.forward(torsions)
        return builder._nonlocal_clash_terms(
            xyz,
            origins,
            axes,
            affected,
            pairs,
            reference[-1],
            clearance=3.0,
            weight=0.25,
        )

    residual, analytic = evaluate(q)
    analytic = analytic.toarray()
    assert np.any(residual > 0)
    finite = np.empty_like(analytic)
    for index in range(len(q)):
        offset = np.zeros_like(q)
        offset[index] = 1e-6
        finite[:, index] = (evaluate(q + offset)[0] - evaluate(q - offset)[0]) / 2e-6
    np.testing.assert_allclose(analytic, finite, atol=2e-7, rtol=1e-6)


def test_optimizer_success_does_not_authorize_unclosed_backbone(monkeypatch):
    reference = _reference(8)

    def dishonest_solver(fun, initial, **kwargs):
        return SimpleNamespace(x=initial, nfev=1, success=True)

    monkeypatch.setattr(scipy.optimize, "least_squares", dishonest_solver)
    xyz, report = builder.repair_backbone(
        reference,
        reference[0],
        reference[-1] + [1.0, 0.0, 0.0],
        generated_length=8,
        attempts=1,
    )
    assert xyz is None
    assert not report["geometry_verified"]
    assert report["attempts"][0]["optimizer_success"]
    assert not report["attempts"][0]["accepted"]


def test_unreachable_contour_is_explicit_and_does_not_run_optimizer(monkeypatch):
    reference = _reference(8)

    def must_not_run(*args, **kwargs):
        pytest.fail("Contour impossibility should be identified before optimization")

    monkeypatch.setattr(scipy.optimize, "least_squares", must_not_run)
    xyz, report = builder.repair_backbone(
        reference,
        reference[0],
        reference[-1] + [1000.0, 0.0, 0.0],
        generated_length=8,
    )
    assert xyz is None
    assert report["reason"] == "target_exceeds_internal_coordinate_contour"
    assert report["attempts"] == []


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"generated_length": 9}, "length mismatch"),
        ({"generated_length": 8, "secondary_structure": "HHL"}, "exactly 8"),
        ({"generated_length": 8, "attempts": 0}, "attempts"),
        ({"generated_length": 8, "reference_sequence_separation": 2.5}, "integer"),
        (
            {"generated_length": 8, "nonlocal_clash_weight": -1.0},
            "nonlocal_clash_weight",
        ),
        (
            {"generated_length": 8, "clash_clearance_margin": -1.0},
            "clash_clearance_margin",
        ),
    ],
)
def test_invalid_requests_fail_before_search(kwargs, message):
    reference = _reference(8)
    with pytest.raises(ValueError, match=message):
        builder.repair_backbone(reference, reference[0], reference[-1], **kwargs)


def test_nonfinite_coordinates_are_not_sanitized_into_a_solution():
    reference = _reference(8)
    target = reference[-1].copy()
    target[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        builder.repair_backbone(reference, reference[0], target, generated_length=8)
