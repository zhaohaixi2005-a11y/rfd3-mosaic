"""Differentiable evaluation of the reference transport's principal screw map.

This is an objective evaluator, not an acceptance check. The independent
NumPy implementation still validates all fixed atoms, bounds, conjugacy,
backbone geometry and the reference-transition certificate before commit.
"""

from __future__ import annotations

import math

import torch


def _skew(v):
    x, y, z = v.unbind(dim=-1)
    zero = x * 0.0
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *v.shape[:-1], 3, 3
    )


def _frame(points):
    """A proper frame from an immutable, noncollinear landmark triple."""
    first = points[1] - points[0]
    first = first / torch.linalg.vector_norm(first)
    second = points[2] - points[0]
    second = second - torch.dot(first, second) * first
    second = second / torch.linalg.vector_norm(second)
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=1)


def rigid_landmarks(points):
    """Choose well separated landmarks once, independently of a trial pose."""
    distances = torch.cdist(points, points)
    flat = int(distances.argmax())
    a, b = divmod(flat, len(points))
    areas = torch.linalg.vector_norm(
        torch.linalg.cross(
            (points[b] - points[a]).expand_as(points), points - points[a], dim=-1
        ),
        dim=-1,
    )
    c = int(areas.argmax())
    if not torch.isfinite(areas[c]) or float(areas[c]) <= 1e-10:
        raise ValueError(
            "Differentiable reference transport needs noncollinear seed atoms"
        )
    return torch.tensor([a, b, c], dtype=torch.long, device=points.device)


def rigid_transform(source, target):
    """Recover an exact rigid map without SVD's repeated-singular-value gradient.

    Both arguments are the same three noncollinear atoms. Valid proposals are
    rigid by construction; the commit path separately fits and checks ALL atoms.
    """
    rotation = _frame(target) @ _frame(source).T
    translation = target.mean(dim=0) - rotation @ source.mean(dim=0)
    top = torch.cat((rotation, translation[:, None]), dim=1)
    return torch.cat((top, source.new_tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)


def interpolate_se3(left, right, fraction):
    """Torch equivalent of validation.reference_transport.interpolate_se3.

    Series at identity avoid the undefined derivative of acos(1). The ambiguous
    principal-logarithm pi branch is rejected, just as in the commit evaluator.
    """
    u = float(fraction)
    if not 0.0 <= u <= 1.0:
        raise ValueError("SE(3) interpolation fraction must lie in [0, 1]")
    if u == 0.0:
        return left
    if u == 1.0:
        return right
    return interpolate_se3_batch(left, right, left.new_tensor([u]))[0]


def interpolate_se3_batch(left, right, fractions):
    """One logarithm per anchor pair, batched residue interpolation on device."""
    if fractions.ndim != 1:
        raise ValueError("SE(3) interpolation fractions must be a vector")
    relative_rotation = left[:3, :3].T @ right[:3, :3]
    relative_translation = left[:3, :3].T @ (right[:3, 3] - left[:3, 3])
    sine_vector = (
        torch.stack(
            (
                relative_rotation[2, 1] - relative_rotation[1, 2],
                relative_rotation[0, 2] - relative_rotation[2, 0],
                relative_rotation[1, 0] - relative_rotation[0, 1],
            )
        )
        * 0.5
    )
    cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    sine = torch.linalg.vector_norm(sine_vector)
    if float(cosine.detach()) > 1.0 - 1e-8:
        # theta/sin(theta) = 1 + (1-cos(theta))/3 + 2(1-cos(theta))^2/15 + ...
        deficit = 1.0 - cosine
        omega = sine_vector * (1.0 + deficit / 3.0 + 2.0 * deficit**2 / 15.0)
    else:
        theta = torch.atan2(sine, cosine)
        if float(theta.detach()) >= math.pi - 1e-6:
            raise ValueError(
                "Reference transport relative rotation reaches the ambiguous pi branch"
            )
        omega = sine_vector * (theta / sine)

    identity = torch.eye(3, dtype=left.dtype, device=left.device)

    def exponential(scale):
        scale = torch.as_tensor(scale, dtype=left.dtype, device=left.device)
        vector = scale[..., None] * omega
        w = _skew(vector)
        square_angle = torch.sum(vector**2, dim=-1)
        # Float32 loses both values AND pose derivatives in 1-cos(theta)
        # and theta-sin(theta) long before the angle reaches 1e-4. Expand
        # through theta^10 on the wider single-precision interval; at its
        # 0.5-radian boundary the first omitted coefficient contributes
        # less than 4e-14. Float64 can use direct coefficients above 0.01.
        series_limit = 1e-4 if left.dtype == torch.float64 else 0.25
        small = square_angle < series_limit
        safe_square = square_angle.clamp_min(series_limit)
        angle = torch.sqrt(safe_square)
        s = square_angle
        a = torch.where(
            small,
            1.0
            + s
            * (
                -1.0 / 6.0
                + s
                * (
                    1.0 / 120.0
                    + s * (-1.0 / 5040.0 + s * (1.0 / 362880.0 - s / 39916800.0))
                )
            ),
            torch.sin(angle) / angle,
        )
        b = torch.where(
            small,
            0.5
            + s
            * (
                -1.0 / 24.0
                + s
                * (
                    1.0 / 720.0
                    + s * (-1.0 / 40320.0 + s * (1.0 / 3628800.0 - s / 479001600.0))
                )
            ),
            (1.0 - torch.cos(angle)) / safe_square,
        )
        c = torch.where(
            small,
            1.0 / 6.0
            + s
            * (
                -1.0 / 120.0
                + s
                * (
                    1.0 / 5040.0
                    + s * (-1.0 / 362880.0 + s * (1.0 / 39916800.0 - s / 6227020800.0))
                )
            ),
            (angle - torch.sin(angle)) / (safe_square * angle),
        )
        a, b, c = (coefficient[..., None, None] for coefficient in (a, b, c))
        return identity + a * w + b * (w @ w), identity + b * w + c * (w @ w)

    _, full_v = exponential(1.0)
    velocity = torch.linalg.solve(full_v, relative_translation)
    rotation, v = exponential(fractions)
    translation = (v @ (fractions[:, None] * velocity)[..., None])[..., 0]
    top = torch.cat((rotation, translation[..., None]), dim=-1)
    increment = torch.cat(
        (top, left.new_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(len(fractions), -1, -1)),
        dim=-2,
    )
    result = left @ increment
    result = torch.where((fractions == 0.0)[:, None, None], left, result)
    return torch.where((fractions == 1.0)[:, None, None], right, result)
