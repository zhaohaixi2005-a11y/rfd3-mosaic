"""Exact per-token cylindrical-coordinate projection around one axis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CylindricalCoordinateProjector:
    """Restore selected radius, azimuth and axial coordinates exactly.

    ``keep_mask`` has shape ``[L, 3]`` in ``(radius, azimuth, axial)``
    order.  The reference and sampled coordinates have shape ``[D, L, 3]``.
    Unselected coordinates are retained unless a protein atom map and CA mask
    request rigid residue translation for CA-only constraints. This transports
    the sampled residue; it does not copy a reference conformation.
    """

    reference: torch.Tensor
    keep_mask: torch.Tensor
    axis: torch.Tensor
    center: torch.Tensor
    atom_to_token_map: torch.Tensor | None = None
    is_ca: torch.Tensor | None = None
    fixed_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        reference = torch.as_tensor(self.reference)
        if reference.ndim != 3 or reference.shape[-1] != 3:
            raise ValueError(
                "Cylindrical reference must have shape [D, L, 3]"
            )
        if not torch.isfinite(reference).all():
            raise ValueError("Cylindrical reference contains NaN or Inf")
        keep = torch.as_tensor(
            self.keep_mask,
            dtype=torch.bool,
            device=reference.device,
        )
        if keep.shape != (reference.shape[1], 3):
            raise ValueError(
                "Cylindrical keep mask must have shape [L, 3]"
            )
        axis = torch.as_tensor(
            self.axis,
            dtype=reference.dtype,
            device=reference.device,
        )
        center = torch.as_tensor(
            self.center,
            dtype=reference.dtype,
            device=reference.device,
        )
        if axis.shape != (3,) or center.shape != (3,):
            raise ValueError("Cylindrical axis and center must have shape [3]")
        if not torch.isfinite(axis).all() or not torch.isfinite(center).all():
            raise ValueError("Cylindrical axis or center contains NaN or Inf")
        norm = torch.linalg.vector_norm(axis)
        if float(norm.item()) <= 1.0e-8:
            raise ValueError("Cylindrical axis must be non-zero")
        axis = axis / norm
        relative = reference - center
        axial = torch.sum(relative * axis, dim=-1)
        radial = relative - axial[..., None] * axis
        radius = torch.linalg.vector_norm(radial, dim=-1)
        epsilon = torch.finfo(reference.dtype).eps * 32.0
        if torch.any(keep[:, 1][None, :] & (radius <= epsilon)):
            raise ValueError(
                "Cannot lock cylindrical azimuth for a reference token "
                "that lies on the symmetry axis"
            )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "keep_mask", keep)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "center", center)
        # A protein CA is a spatial handle for its whole residue. Transport
        # its unconstrained atoms together, preserving all intra-residue
        # covalent geometry and sidechains. This does not certify peptide
        # junctions or nonlocal packing; those need full-backbone acceptance.
        anchors = torch.full((reference.shape[1],), -1, dtype=torch.long, device=reference.device)
        if (self.atom_to_token_map is None) != (self.is_ca is None):
            raise ValueError("Cylindrical residue transport requires both atom map and CA mask")
        if self.atom_to_token_map is not None:
            mapping = torch.as_tensor(self.atom_to_token_map, dtype=torch.long, device=reference.device)
            ca = torch.as_tensor(self.is_ca, dtype=torch.bool, device=reference.device)
            fixed = torch.zeros_like(ca) if self.fixed_mask is None else torch.as_tensor(self.fixed_mask, dtype=torch.bool, device=reference.device)
            if mapping.shape != anchors.shape or ca.shape != anchors.shape or fixed.shape != anchors.shape or torch.any(mapping < 0):
                raise ValueError("Cylindrical residue transport annotations must have shape [L]")
            active = keep.any(dim=1)
            for token in torch.unique(mapping[active & ca]).tolist():
                token_mask = mapping == token
                selected = torch.nonzero(token_mask & active, as_tuple=False).flatten()
                ca_indices = torch.nonzero(token_mask & ca, as_tuple=False).flatten()
                if selected.numel() == 1 and ca_indices.numel() == 1 and selected[0] == ca_indices[0]:
                    if torch.any(token_mask & fixed):
                        raise ValueError("CA cylindrical residue transport cannot move a fixed-XYZ residue")
                    anchors[token_mask] = selected[0]
        object.__setattr__(self, "_transport_anchors", anchors)

    def _parts(
        self,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relative = coordinates - self.center
        axial = torch.sum(relative * self.axis, dim=-1)
        radial = relative - axial[..., None] * self.axis
        radius = torch.linalg.vector_norm(radial, dim=-1)
        return relative, axial, radial, radius

    def project(self, coordinates: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(
            coordinates,
            dtype=self.reference.dtype,
            device=self.reference.device,
        )
        if value.shape != self.reference.shape:
            raise ValueError(
                "Cylindrical coordinates must match reference shape"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Cylindrical coordinates contain NaN or Inf")

        _, axial, radial, radius = self._parts(value)
        _, target_axial, target_radial, target_radius = self._parts(
            self.reference
        )
        epsilon = torch.finfo(value.dtype).eps * 32.0
        target_direction = target_radial / target_radius.clamp_min(
            epsilon
        )[..., None]
        current_direction = radial / radius.clamp_min(epsilon)[..., None]
        # On the axis azimuth is undefined.  Falling back to the reference
        # direction makes a later non-zero radius deterministic and exact.
        current_direction = torch.where(
            (radius > epsilon)[..., None],
            current_direction,
            target_direction,
        )
        azimuth_mask = self.keep_mask[:, 1][None, :, None]
        direction = torch.where(
            azimuth_mask,
            target_direction,
            current_direction,
        )
        resolved_radius = torch.where(
            self.keep_mask[:, 0][None, :],
            target_radius,
            radius,
        )
        resolved_axial = torch.where(
            self.keep_mask[:, 2][None, :],
            target_axial,
            axial,
        )
        projected = (
            self.center
            + resolved_radius[..., None] * direction
            + resolved_axial[..., None] * self.axis
        )
        active = self.keep_mask.any(dim=1)[None, :, None]
        projected = torch.where(active, projected, value)
        anchors = self._transport_anchors
        if torch.any(anchors >= 0):
            anchor_indices = anchors.clamp_min(0)
            displacement = projected[:, anchor_indices] - value[:, anchor_indices]
            transported = value + displacement
            # Preserve the direct target calculation at constrained CA atoms.
            transported = torch.where(active, projected, transported)
            projected = torch.where((anchors >= 0)[None, :, None], transported, projected)
        return projected

    def maximum_error(self, coordinates: torch.Tensor) -> float:
        projected = self.project(coordinates)
        _, axial, radial, radius = self._parts(coordinates)
        _, target_axial, target_radial, target_radius = self._parts(
            self.reference
        )
        errors: list[torch.Tensor] = []
        if torch.any(self.keep_mask[:, 0]):
            mask = self.keep_mask[:, 0]
            errors.append(torch.abs(radius[:, mask] - target_radius[:, mask]))
        if torch.any(self.keep_mask[:, 2]):
            mask = self.keep_mask[:, 2]
            errors.append(torch.abs(axial[:, mask] - target_axial[:, mask]))
        if torch.any(self.keep_mask[:, 1]):
            mask = self.keep_mask[:, 1]
            epsilon = torch.finfo(projected.dtype).eps * 32.0
            direction = radial[:, mask] / radius[:, mask].clamp_min(
                epsilon
            )[..., None]
            target_direction = target_radial[:, mask] / target_radius[
                :, mask
            ].clamp_min(epsilon)[..., None]
            errors.append(torch.linalg.vector_norm(
                direction - target_direction,
                dim=-1,
            ))
        if not errors:
            return 0.0
        return float(max(error.max().item() for error in errors))


__all__ = ["CylindricalCoordinateProjector"]
