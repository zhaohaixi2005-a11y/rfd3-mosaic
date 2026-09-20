"""Check the joint solver's derivatives and symmetry without a model/GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_scaffold_builder import _reference

from rfd3_mosaic.scaffold_assembly import _landmarks, solve_assembly


def fixture(dihedral=False):
    xyz = _reference(20, compact=True) + [35, 0, 10]
    pattern = ["A1", "A2"] + [None] * 18 + ["A21", "A22"]
    source = {
        c: {
            "name": "ALA",
            "atoms": dict(zip(("N", "CA", "C", "O"), xyz[i], strict=True)),
        }
        for i, c in enumerate(pattern)
        if c
    }
    rotations = [np.eye(3), np.diag([-1.0, -1.0, 1.0])]
    if dihedral:
        rotations += [np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0])]
    transforms = []
    for r in rotations:
        m = np.eye(4)
        m[:3, :3] = r
        transforms.append(m)
    extra = {
        "registry_transform_order": list(range(len(transforms))),
        "registry_transform_matrices": dict(enumerate(transforms)),
        "motif_constraint_groups": [
            {
                "group_id": "seed",
                "members": [{"sym_transform_id": 0, "src_components": list(source)}],
            }
        ],
        "motif_constraint_orbits": [{"master_group_id": "seed"}],
    }
    spec = {
        "maximum_template_seed_rmsd": 0.1,
        "construction": {"attempts": 1, "max_nfev": 1},
        "helix_blocks": [
            {"id": "left", "entity": 0, "start": 2, "end": 9},
            {"id": "right", "entity": 0, "start": 13, "end": 21},
        ],
        "support_edges": [{"left": "left", "right": "right"}],
        "limits": {
            "maximum_ca_deviation": 1.0,
            "minimum_helix_contact_fraction": 0.2,
            "contact_distance": 8.0,
            "maximum_unsupported_run": 6,
        },
    }
    return {
        "source": source,
        "patterns": [pattern],
        "runs": [[(2, 20, 2, 20)]],
        "aligned": [(xyz, None)],
        "extra": extra,
        "spec": spec,
    }


@pytest.mark.parametrize("dihedral", [False, True])
def test_full_joint_residual_jacobian_matches_numerical_derivative(
    monkeypatch, dihedral
):
    calls = []

    def check(fun, initial, *, jac, **kwargs):
        trial = initial + np.random.default_rng(7).normal(0, 0.002, len(initial))
        analytic = jac(trial).toarray()
        for column in (0, 2, 3, 5, 6, 12, len(initial) - 1):
            plus, minus = trial.copy(), trial.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            numerical = (fun(plus) - fun(minus)) / 2e-6
            np.testing.assert_allclose(
                analytic[:, column], numerical, atol=8e-4, rtol=3e-5
            )
        calls.append(True)
        return SimpleNamespace(x=initial, nfev=1, success=True)

    monkeypatch.setattr("scipy.optimize.least_squares", check)
    args = fixture(dihedral)
    asu, report = solve_assembly(**args, candidate_audit=lambda x: {"passed": True})
    assert calls and asu is not None
    assert report["status"] == "verified_assembly_geometry"
    for i, component in enumerate(args["patterns"][0]):
        if component:
            np.testing.assert_allclose(
                list(asu[0][i]["atoms"].values()),
                list(args["source"][component]["atoms"].values()),
                atol=1e-12,
            )


def test_optimizer_success_cannot_override_independent_assembly_rejection():
    args = fixture()
    asu, report = solve_assembly(
        **args,
        candidate_audit=lambda x: {
            "passed": False,
            "reason": "full_assembly_collision",
        },
    )
    assert asu is None and report["status"] == "unresolved"
    assert not report["attempts"][0]["accepted"]


@pytest.mark.parametrize(
    "old,new", [("HHHLLHHH", "HHHHHLLLHH"), ("HHHHHLLLHH", "HHHLLHHH")]
)
def test_changed_length_landmarks_are_injective_and_respect_block_identity(old, new):
    mapping = _landmarks(old, new)
    assert len(set(mapping.values())) == len(mapping)
    assert all(old[a] == new[b] for a, b in mapping.items())
    assert list(mapping) == sorted(mapping)
    assert list(mapping.values()) == sorted(mapping.values())
