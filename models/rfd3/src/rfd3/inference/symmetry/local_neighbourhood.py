"""Bounded symmetry neighbourhoods for scalable high-order inference.

The current native sampler materializes every symmetry copy in the network
view.  For high-order groups this is unnecessarily expensive: an ASU usually
interacts only with nearby copies.  This module defines the exact mapping
between a bounded local network view and the complete symmetry orbit.

It deliberately contains no model-specific feature cropping.  The mapping can
therefore be tested independently before the initializer and denoiser are
switched from the explicit all-copy representation.
"""

import re
from dataclasses import dataclass
from typing import Any

import torch
from rfd3.inference.symmetry.atom_array import (
    FIXED_ENTITY_ID,
    FIXED_TRANSFORM_ID,
)
from rfd3.inference.symmetry.symmetry_utils import (
    SymmetryOrbitLayout,
    build_symmetry_orbit_layout,
)


@dataclass(frozen=True)
class LocalSymmetryNeighbourhood:
    """A deterministic local atom view of a complete symmetry assembly."""

    symmetry_id: str
    master_transform_id: int
    selected_transform_ids: tuple[int, ...]
    atom_indices: torch.Tensor
    global_to_local_atom: torch.Tensor
    full_atom_count: int

    @property
    def copy_count(self) -> int:
        return len(self.selected_transform_ids)


@dataclass(frozen=True)
class LocalFeatureView:
    """Cropped network features and their global token correspondence."""

    features: dict[str, Any]
    token_indices: torch.Tensor
    global_to_local_token: torch.Tensor


@dataclass(frozen=True)
class LocalSymmetryRuntimeContext:
    """Complete mapping shared by initializer, sampler and output heads."""

    layout: SymmetryOrbitLayout
    neighbourhood: LocalSymmetryNeighbourhood
    feature_view: LocalFeatureView


def _parse_finite_symmetry_id(symmetry_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([CcDd])(\d+)", str(symmetry_id).strip())
    if match is None:
        raise ValueError(
            "Local symmetry neighbourhoods currently support only Cn and "
            f"Dn identifiers, got {symmetry_id!r}"
        )
    kind = match.group(1).upper()
    order = int(match.group(2))
    minimum = 1 if kind == "C" else 2
    if order < minimum:
        raise ValueError(
            f"{kind} symmetry order must be at least {minimum}, got {order}"
        )
    return kind, order


def _cyclic_offsets(radius: int) -> tuple[int, ...]:
    if radius < 0:
        raise ValueError("neighbour_radius cannot be negative")
    offsets = [0]
    for distance in range(1, radius + 1):
        offsets.extend((-distance, distance))
    return tuple(offsets)


def select_local_transform_ids(
    symmetry_id: str,
    *,
    master_transform_id: int = 0,
    neighbour_radius: int = 1,
    include_dihedral_mate: bool = True,
) -> tuple[int, ...]:
    """Select master-adjacent transforms without scaling with group order.

    Foundry orders Cn transforms as one cyclic coset and Dn transforms as two
    cyclic cosets of size ``n``.  A Dn local view contains the master coset and,
    by default, the corresponding neighbourhood in the paired coset.
    """

    kind, order = _parse_finite_symmetry_id(symmetry_id)
    multiplicity = order if kind == "C" else 2 * order
    if not 0 <= int(master_transform_id) < multiplicity:
        raise ValueError(
            f"master_transform_id must be in [0, {multiplicity}), got "
            f"{master_transform_id}"
        )

    master_transform_id = int(master_transform_id)
    offsets = _cyclic_offsets(int(neighbour_radius))
    master_coset = master_transform_id // order
    phase = master_transform_id % order
    cosets = [master_coset]
    if kind == "D" and include_dihedral_mate:
        cosets.append(1 - master_coset)

    selected: list[int] = []
    for coset in cosets:
        for offset in offsets:
            transform_id = coset * order + (phase + offset) % order
            if transform_id not in selected:
                selected.append(transform_id)
    return tuple(selected)


def build_local_symmetry_neighbourhood(
    sym_feats: dict[str, Any],
    symmetry_id: str,
    *,
    like: torch.Tensor,
    neighbour_radius: int = 1,
    include_dihedral_mate: bool = True,
    layout: SymmetryOrbitLayout | None = None,
) -> LocalSymmetryNeighbourhood:
    """Build a complete-token local atom selection from runtime annotations."""

    if like.ndim != 3 or like.shape[-1] != 3:
        raise ValueError("like must have shape [batch, atoms, 3]")
    resolved_layout = (
        build_symmetry_orbit_layout(sym_feats, like=like)
        if layout is None
        else layout
    )
    asu_transform_ids = {
        asu_transform_id
        for _, asu_transform_id, _ in resolved_layout.entity_orbits
    }
    if len(asu_transform_ids) != 1:
        raise ValueError(
            "All symmetric entities must use the same ASU transform for a "
            "shared local network view"
        )
    master_transform_id = next(iter(asu_transform_ids))
    selected_transform_ids = select_local_transform_ids(
        symmetry_id,
        master_transform_id=master_transform_id,
        neighbour_radius=neighbour_radius,
        include_dihedral_mate=include_dihedral_mate,
    )

    available_transform_ids = set(resolved_layout.sym_transforms)
    missing = set(selected_transform_ids) - available_transform_ids
    if missing:
        raise ValueError(
            f"Runtime symmetry features lack selected transforms {sorted(missing)}"
        )

    ordered_indices = [
        torch.nonzero(
            resolved_layout.sym_transform_id == transform_id,
            as_tuple=False,
        ).flatten()
        for transform_id in selected_transform_ids
    ]
    # Unsymmetrized entities are global context and cannot be reconstructed
    # from the master orbit, so retain them after the ordered symmetry copies.
    fixed_indices = torch.nonzero(
        (resolved_layout.sym_entity_id == FIXED_ENTITY_ID)
        | (resolved_layout.sym_transform_id == FIXED_TRANSFORM_ID),
        as_tuple=False,
    ).flatten()
    if len(fixed_indices):
        ordered_indices.append(fixed_indices)
    atom_indices = torch.cat(ordered_indices)
    global_to_local = torch.full(
        (like.shape[-2],),
        -1,
        dtype=torch.long,
        device=like.device,
    )
    global_to_local[atom_indices] = torch.arange(
        len(atom_indices),
        dtype=torch.long,
        device=like.device,
    )
    return LocalSymmetryNeighbourhood(
        symmetry_id=str(symmetry_id).upper(),
        master_transform_id=master_transform_id,
        selected_transform_ids=selected_transform_ids,
        atom_indices=atom_indices,
        global_to_local_atom=global_to_local,
        full_atom_count=like.shape[-2],
    )


def crop_features_to_local_neighbourhood(
    features: dict[str, Any],
    neighbourhood: LocalSymmetryNeighbourhood,
) -> LocalFeatureView:
    """Crop atom/token features together and reindex atom-to-token mapping.

    The function fails closed if the atom selection splits a token.  Pair
    tensors are cropped on both token axes, while Mosaic motif-constraint
    tensors use their declared atom axis.  Unknown tensors are cropped only
    when their leading dimension unambiguously equals the full atom or token
    count; all metadata/scalars are retained unchanged.
    """

    if "atom_to_token_map" not in features:
        raise ValueError("Local feature cropping requires atom_to_token_map")
    atom_to_token = torch.as_tensor(
        features["atom_to_token_map"],
        dtype=torch.long,
        device=neighbourhood.atom_indices.device,
    )
    if atom_to_token.ndim != 1 or len(atom_to_token) != neighbourhood.full_atom_count:
        raise ValueError(
            "atom_to_token_map must be one-dimensional and match full atoms"
        )
    if len(atom_to_token) == 0:
        raise ValueError("Cannot construct a local view without atoms")
    token_count = int(atom_to_token.max().item()) + 1
    selected_global_tokens = atom_to_token[neighbourhood.atom_indices]

    token_order: list[int] = []
    for token_id in selected_global_tokens.tolist():
        if token_id not in token_order:
            token_order.append(token_id)
    token_indices = torch.tensor(
        token_order,
        dtype=torch.long,
        device=atom_to_token.device,
    )
    global_to_local_token = torch.full(
        (token_count,),
        -1,
        dtype=torch.long,
        device=atom_to_token.device,
    )
    global_to_local_token[token_indices] = torch.arange(
        len(token_indices),
        dtype=torch.long,
        device=atom_to_token.device,
    )

    full_counts = torch.bincount(atom_to_token, minlength=token_count)
    selected_counts = torch.bincount(
        selected_global_tokens,
        minlength=token_count,
    )
    if torch.any(
        (selected_counts > 0) & (selected_counts != full_counts)
    ):
        raise ValueError(
            "Local symmetry selection splits at least one atomized token"
        )

    local_features: dict[str, Any] = {}
    atom_count = neighbourhood.full_atom_count
    atom_indices = neighbourhood.atom_indices
    for key, raw_value in features.items():
        if key == "atom_to_token_map":
            local_features[key] = global_to_local_token[
                selected_global_tokens
            ]
            continue
        if key == "sym_transform" or not hasattr(raw_value, "shape"):
            local_features[key] = raw_value
            continue

        value = torch.as_tensor(raw_value, device=atom_indices.device)
        if key == "motif_constraint_group_membership":
            if value.ndim != 2 or value.shape[1] != atom_count:
                raise ValueError(
                    "motif_constraint_group_membership must have shape "
                    "[groups, atoms]"
                )
            local_features[key] = value[:, atom_indices]
        elif key == "motif_constraint_target_coordinates":
            if value.ndim != 4 or value.shape[2] != atom_count:
                raise ValueError(
                    "motif_constraint_target_coordinates must have shape "
                    "[batch, groups, atoms, 3]"
                )
            local_features[key] = value[:, :, atom_indices, :]
        elif value.ndim >= 2 and value.shape[:2] == (token_count, token_count):
            local_features[key] = value[
                token_indices[:, None],
                token_indices[None, :],
            ]
        elif value.ndim >= 2 and value.shape[:2] == (atom_count, atom_count):
            local_features[key] = value[
                atom_indices[:, None],
                atom_indices[None, :],
            ]
        elif value.ndim >= 1 and value.shape[0] == atom_count:
            local_features[key] = value[atom_indices]
        elif value.ndim >= 1 and value.shape[0] == token_count:
            local_features[key] = value[token_indices]
        else:
            local_features[key] = value

    return LocalFeatureView(
        features=local_features,
        token_indices=token_indices,
        global_to_local_token=global_to_local_token,
    )


def expand_local_prediction_to_full_orbit(
    local_prediction: torch.Tensor,
    full_template: torch.Tensor,
    neighbourhood: LocalSymmetryNeighbourhood,
    *,
    layout: SymmetryOrbitLayout,
) -> torch.Tensor:
    """Average selected predictions canonically and rebuild every copy.

    Omitted copies in ``full_template`` never enter the average.  Consequently
    a C200 update computed from three local copies has the same magnitude as a
    C12 update computed from the same neighbourhood instead of being diluted
    by the 197 copies that were not evaluated by the denoiser.
    """

    if full_template.ndim != 3 or full_template.shape[-1] != 3:
        raise ValueError("full_template must have shape [batch, atoms, 3]")
    if full_template.shape[-2] != neighbourhood.full_atom_count:
        raise ValueError("full_template atom dimension does not match neighbourhood")
    if (
        local_prediction.ndim != 3
        or local_prediction.shape[0] != full_template.shape[0]
        or local_prediction.shape[-1] != 3
        or local_prediction.shape[-2] != len(neighbourhood.atom_indices)
    ):
        raise ValueError(
            "local_prediction must have shape [batch, selected_atoms, 3]"
        )

    work_dtype = (
        torch.float64
        if full_template.dtype == torch.float64
        else torch.float32
    )
    local = local_prediction.to(dtype=work_dtype)
    expanded = full_template.to(dtype=work_dtype).clone()
    selected_ids = set(neighbourhood.selected_transform_ids)

    with torch.autocast(
        device_type=full_template.device.type,
        enabled=False,
    ):
        for entity_id, _, copies in layout.entity_orbits:
            canonical_predictions = []
            for transform_id, global_indices in copies:
                if transform_id not in selected_ids:
                    continue
                local_indices = neighbourhood.global_to_local_atom[
                    global_indices
                ]
                if torch.any(local_indices < 0):
                    raise ValueError(
                        "Selected transform has incomplete local atom coverage: "
                        f"entity={entity_id}, transform={transform_id}"
                    )
                rotation, translation = layout.sym_transforms[transform_id]
                canonical_predictions.append(
                    torch.matmul(
                        local[:, local_indices, :] - translation,
                        rotation,
                    )
                )
            if not canonical_predictions:
                raise ValueError(
                    f"No selected prediction covers symmetric entity {entity_id}"
                )
            canonical_mean = torch.stack(
                canonical_predictions,
                dim=0,
            ).mean(dim=0)
            for transform_id, global_indices in copies:
                rotation, translation = layout.sym_transforms[transform_id]
                expanded[:, global_indices, :] = (
                    torch.matmul(
                        canonical_mean,
                        rotation.transpose(-1, -2),
                    )
                    + translation
                )
    return expanded


def expand_local_token_prediction_to_full_orbit(
    local_prediction: torch.Tensor,
    full_features: dict[str, Any],
    feature_view: LocalFeatureView,
) -> torch.Tensor:
    """Average copy-equivalent token predictions and expand globally.

    Token correspondence is derived from the complete set of atom orbit slots
    belonging to each token, not from chain letters or incidental array order.
    This supports sequence logits and other per-token model outputs.
    """

    required = {
        "atom_to_token_map",
        "sym_entity_id",
        "sym_transform_id",
        "sym_orbit_slot",
    }
    missing = required - set(full_features)
    if missing:
        raise ValueError(
            "Token-orbit expansion requires features "
            f"{sorted(missing)}"
        )
    atom_to_token = torch.as_tensor(
        full_features["atom_to_token_map"],
        dtype=torch.long,
        device=local_prediction.device,
    )
    entity_ids = torch.as_tensor(
        full_features["sym_entity_id"],
        dtype=torch.long,
        device=local_prediction.device,
    )
    transform_ids = torch.as_tensor(
        full_features["sym_transform_id"],
        dtype=torch.long,
        device=local_prediction.device,
    )
    orbit_slots = torch.as_tensor(
        full_features["sym_orbit_slot"],
        dtype=torch.long,
        device=local_prediction.device,
    )
    if not (
        atom_to_token.ndim
        == entity_ids.ndim
        == transform_ids.ndim
        == orbit_slots.ndim
        == 1
        and len(atom_to_token)
        == len(entity_ids)
        == len(transform_ids)
        == len(orbit_slots)
    ):
        raise ValueError("Atomwise token-orbit features must have equal 1D shape")
    token_count = int(atom_to_token.max().item()) + 1
    if (
        local_prediction.ndim < 2
        or local_prediction.shape[-2] != len(feature_view.token_indices)
    ):
        raise ValueError(
            "local token prediction penultimate dimension must match local tokens"
        )

    token_keys: list[tuple[Any, ...]] = []
    for token_id in range(token_count):
        atom_mask = atom_to_token == token_id
        token_entities = torch.unique(entity_ids[atom_mask]).tolist()
        token_transforms = torch.unique(transform_ids[atom_mask]).tolist()
        if len(token_entities) != 1 or len(token_transforms) != 1:
            raise ValueError(
                f"Token {token_id} spans multiple symmetry entities/transforms"
            )
        entity_id = int(token_entities[0])
        transform_id = int(token_transforms[0])
        if entity_id == FIXED_ENTITY_ID or transform_id == FIXED_TRANSFORM_ID:
            token_keys.append(("fixed", token_id))
        else:
            slots = tuple(
                sorted(int(value) for value in orbit_slots[atom_mask].tolist())
            )
            token_keys.append(("orbit", entity_id, slots))

    local_groups: dict[tuple[Any, ...], list[int]] = {}
    for local_token_id, global_token_id in enumerate(
        feature_view.token_indices.tolist()
    ):
        local_groups.setdefault(
            token_keys[global_token_id],
            [],
        ).append(local_token_id)

    expanded_shape = list(local_prediction.shape)
    expanded_shape[-2] = token_count
    expanded = torch.empty(
        expanded_shape,
        dtype=local_prediction.dtype,
        device=local_prediction.device,
    )
    for global_token_id, key in enumerate(token_keys):
        local_token_ids = local_groups.get(key)
        if not local_token_ids:
            raise ValueError(
                "Local token view does not cover global token orbit "
                f"{key!r}"
            )
        index = torch.tensor(
            local_token_ids,
            dtype=torch.long,
            device=local_prediction.device,
        )
        expanded[..., global_token_id, :] = local_prediction.index_select(
            -2,
            index,
        ).mean(dim=-2)
    return expanded
