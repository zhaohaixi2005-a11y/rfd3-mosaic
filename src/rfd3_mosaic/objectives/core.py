"""Backend-independent scalar objectives and deterministic score reports."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

from rfd3_mosaic.schema import ObjectiveMode, ObjectiveSpec


@dataclass(frozen=True)
class ObjectiveEvaluation:
    objective_id: str
    metric: str
    mode: str
    value: float
    penalty: float
    weight: float
    weighted_penalty: float
    required: bool
    satisfied: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreReport:
    evaluations: tuple[ObjectiveEvaluation, ...]
    total_weighted_penalty: float
    required_failure_count: int
    all_required_satisfied: bool

    @property
    def ranking_key(self) -> tuple[int, float]:
        """Sort feasible candidates before infeasible ones, then by score."""

        return self.required_failure_count, self.total_weighted_penalty

    def to_dict(self) -> dict[str, object]:
        return {
            "total_weighted_penalty": self.total_weighted_penalty,
            "required_failure_count": self.required_failure_count,
            "all_required_satisfied": self.all_required_satisfied,
            "ranking_key": list(self.ranking_key),
            "evaluations": [item.to_dict() for item in self.evaluations],
        }


def _constraint_violation(spec: ObjectiveSpec, value: float) -> tuple[float, bool]:
    if spec.mode == ObjectiveMode.AT_MOST:
        assert spec.threshold is not None
        violation = max(0.0, value - spec.threshold)
    elif spec.mode == ObjectiveMode.AT_LEAST:
        assert spec.threshold is not None
        violation = max(0.0, spec.threshold - value)
    elif spec.mode == ObjectiveMode.TARGET:
        assert spec.target is not None and spec.tolerance is not None
        violation = max(0.0, abs(value - spec.target) - spec.tolerance)
    elif spec.mode == ObjectiveMode.RANGE:
        assert spec.minimum is not None and spec.maximum is not None
        if value < spec.minimum:
            violation = spec.minimum - value
        elif value > spec.maximum:
            violation = value - spec.maximum
        else:
            violation = 0.0
    else:
        raise ValueError(f"{spec.mode.value!r} is not a constraint objective")
    return (violation / spec.scale) ** 2, violation == 0.0


def evaluate_objective(
    objective_id: str,
    spec: ObjectiveSpec,
    metrics: Mapping[str, float],
) -> ObjectiveEvaluation:
    if spec.metric not in metrics:
        raise KeyError(
            f"Objective {objective_id!r} requires missing metric "
            f"{spec.metric!r}"
        )
    value = float(metrics[spec.metric])
    if not isfinite(value):
        raise ValueError(
            f"Objective metric {spec.metric!r} must be finite, got {value}"
        )

    if spec.mode == ObjectiveMode.MINIMIZE:
        penalty = value / spec.scale
        satisfied = None
    elif spec.mode == ObjectiveMode.MAXIMIZE:
        penalty = -value / spec.scale
        satisfied = None
    else:
        penalty, satisfied = _constraint_violation(spec, value)

    return ObjectiveEvaluation(
        objective_id=objective_id,
        metric=spec.metric,
        mode=spec.mode.value,
        value=value,
        penalty=penalty,
        weight=spec.weight,
        weighted_penalty=spec.weight * penalty,
        required=spec.required,
        satisfied=satisfied,
    )


def evaluate_objectives(
    objectives: Mapping[str, ObjectiveSpec],
    metrics: Mapping[str, float],
) -> ScoreReport:
    evaluations = tuple(
        evaluate_objective(objective_id, spec, metrics)
        for objective_id, spec in objectives.items()
    )
    required_failure_count = sum(
        item.required and item.satisfied is False for item in evaluations
    )
    return ScoreReport(
        evaluations=evaluations,
        total_weighted_penalty=sum(
            (item.weighted_penalty for item in evaluations),
            0.0,
        ),
        required_failure_count=required_failure_count,
        all_required_satisfied=required_failure_count == 0,
    )
