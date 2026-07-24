import logging

import torch

logger = logging.getLogger(__name__)


def weighted_rigid_align(
    X_L: torch.Tensor,  # [B, L, 3]
    X_gt_L: torch.Tensor,  # [B, L, 3]
    X_exists_L: torch.Tensor | None = None,  # [L]
    w_L: torch.Tensor | None = None,  # [B, L]
) -> torch.Tensor:
    """
    Weighted rigid body alignment of X_gt_L onto X_L with weights w_L
    Allows for "moving target" ground truth that is se3 invariant
    Following algorithm 28 in AF3 paper
    Returns:
      X_align_L: [B, L, 3]
    """

    # Canonicalize dimensions
    if X_L.ndim == 2:
        X_L = X_L[None]
    if X_gt_L.ndim == 2:
        X_gt_L = X_gt_L[None]
    output_dtype = X_L.dtype
    alignment_dtype = (
        torch.float64
        if X_L.dtype == torch.float64 or X_gt_L.dtype == torch.float64
        else torch.float32
    )
    if X_exists_L is None:
        X_exists_L = torch.ones(
            (X_L.shape[-2]),
            dtype=torch.bool,
            device=X_L.device,
        )
    else:
        X_exists_L = X_exists_L.to(device=X_L.device)
    if w_L is None:
        w_L = torch.ones(
            X_L.shape[:-1],
            dtype=alignment_dtype,
            device=X_L.device,
        )
    else:
        if w_L.ndim == 1:
            w_L = w_L[None]
        w_L = w_L.to(device=X_L.device, dtype=alignment_dtype)

    # Assert `X_exists_L` is a boolean mask
    assert (
        X_exists_L.dtype == torch.bool
    ), "X_exists_L should be a boolean mask! Otherwise, the alignment will be incorrect (silent failure)!"

    assert X_L.shape == X_gt_L.shape
    assert X_L.shape[:-1] == w_L.shape

    # CUDA/CPU SVD does not support float16 or bfloat16.  Motif
    # realignment runs inside RFD3's mixed-precision inference context, so
    # perform only this small Kabsch solve in float32 (or preserve float64)
    # and cast the aligned coordinates back at the end.
    X_work = X_L.to(alignment_dtype)
    X_gt_work = X_gt_L.to(alignment_dtype)
    X_resolved = X_work[:, X_exists_L]
    X_gt_resolved = X_gt_work[:, X_exists_L]
    w_resolved = w_L[:, X_exists_L]
    u_X = torch.sum(X_resolved * w_resolved.unsqueeze(-1), dim=-2) / torch.sum(
        w_resolved, dim=-1, keepdim=True
    )
    u_X_gt = torch.sum(X_gt_resolved * w_resolved.unsqueeze(-1), dim=-2) / torch.sum(
        w_resolved, dim=-1, keepdim=True
    )

    X_resolved = X_resolved - u_X.unsqueeze(-2)
    X_gt_resolved = X_gt_resolved - u_X_gt.unsqueeze(-2)

    # RFD3 calls this function from an outer bfloat16 autocast context.
    # Without explicitly disabling autocast here, CUDA einsum converts the
    # float32 work arrays back to bfloat16 before SVD.
    with torch.autocast(device_type=X_L.device.type, enabled=False):
        # Computation of the covariance matrix
        C = torch.einsum(
            "bji,bjk->bik",
            w_resolved[..., None] * X_gt_resolved,
            X_resolved,
        )

        U, S, V = torch.linalg.svd(C)

        R = U @ V
        B, _, _ = X_L.shape
        # F is the reflection correction for the Kabsch rotation: its last
        # diagonal entry flips to -1 when det(R) < 0 so R stays a proper
        # rotation. It feeds `U @ F @ V` below, where matmul requires a shared
        # dtype. U and V carry `alignment_dtype`, so F must too; this also
        # preserves float64 callers.
        F = torch.eye(
            3, 3, device=X_L.device, dtype=alignment_dtype
        )[None].tile(
            (
                B,
                1,
                1,
            )
        )

        det = torch.linalg.det(R)
        F[..., -1, -1] = torch.sign(det)
        R = U @ F @ V

        X_gt_work = X_gt_work - u_X_gt.unsqueeze(-2)
        X_align_L = X_gt_work @ R + u_X.unsqueeze(-2)

    return X_align_L.to(output_dtype).detach()


def get_rmsd(xyz1: torch.Tensor, xyz2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    L = xyz1.shape[-2]
    rmsd = torch.sqrt(torch.sum((xyz2 - xyz1) * (xyz2 - xyz1), dim=(-1, -2)) / L + eps)
    return rmsd


def superimpose(
    xyz1: torch.Tensor, xyz2: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4
) -> None:
    """
    Superimpose xyz1 onto xyz2 using mask
    """
    L = xyz1.shape[-2]
    assert mask.shape == (L,)
    assert xyz1.shape == xyz2.shape
    assert mask.dtype == torch.bool
