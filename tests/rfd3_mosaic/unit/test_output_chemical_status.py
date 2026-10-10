import pytest

from rfd3_mosaic.output_chemical_status import joint_task_and_chemical_status


@pytest.mark.parametrize(
    ("contract_status", "chemical_passed", "status", "passed"),
    [
        ("met", True, "passed", True),
        ("met", False, "failed", False),
        ("flagged", True, "failed", False),
        ("flagged", None, "failed", False),
        ("not_evaluated", False, "failed", False),
        ("met", None, "not_evaluated", None),
        ("not_evaluated", True, "not_evaluated", None),
        ("not_evaluated", None, "not_evaluated", None),
    ],
)
def test_joint_task_and_chemical_status_is_strict_and_three_valued(
    contract_status, chemical_passed, status, passed
):
    result = joint_task_and_chemical_status(
        contract_status,
        {"status": "fixture", "passed": chemical_passed},
    )
    assert result == {
        "status": status,
        "passed": passed,
        "contract_passed": (
            True
            if contract_status == "met"
            else False if contract_status == "flagged" else None
        ),
        "chemical_passed": chemical_passed,
    }


def test_joint_status_rejects_non_boolean_chemical_claims_as_unknown():
    result = joint_task_and_chemical_status("met", {"passed": 1})
    assert result["status"] == "not_evaluated"
    assert result["passed"] is None
