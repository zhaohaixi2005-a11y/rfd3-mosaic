from dataclasses import dataclass
from typing import Optional

import biotite.structure as struc
import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from rfd3.inference.symmetry.atom_array import (
    FIXED_ENTITY_ID,
    FIXED_TRANSFORM_ID,
    add_2d_entity_annotations,
    add_src_sym_component_annotations,
    add_sym_annotations,
    annotate_unsym_atom_array,
    fix_3D_sym_motif_annotations,
    get_symmetry_unit,
    reannotate_2d_conditions,
)
from rfd3.inference.symmetry.checks import (
    check_symmetry_config,
)
from rfd3.inference.symmetry.contigs import (
    expand_contig_unsym_motif,
    get_unsym_motif_mask,
)
from rfd3.inference.symmetry.frames import (
    decompose_symmetry_frame,
    get_symmetry_frames_from_atom_array,
    get_symmetry_frames_from_symmetry_id,
    get_symmetry_multiplicity_from_id,
)
from rfd3.transforms.conditioning_base import get_motif_features

from foundry.utils.components import fetch_mask_from_component
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(frozen=True)
class SymmetryOrbitLayout:
    """Validated runtime orbit indices and transforms for one coordinate batch."""

    sym_entity_id: torch.Tensor
    sym_transform_id: torch.Tensor
    is_sym_asu: torch.Tensor
    sym_orbit_slot: torch.Tensor | None
    sym_transforms: dict[int, tuple[torch.Tensor, torch.Tensor]]
    entity_orbits: tuple[
        tuple[int, int, tuple[tuple[int, torch.Tensor], ...]],
        ...,
    ]


class SymmetryConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )
    id: Optional[str] = Field(
        None,
        description=(
            "Symmetry group ID, for example C3 or D2. Compiler-declared "
            "frames additionally support the finite T, O and I groups."
        ),
    )
    is_unsym_motif: Optional[str] = Field(
        None,
        description="Comma separated list of contig/ligand names that should not be symmetrized such as DNA strands. \
         e.g. 'HEM' or 'Y1-11,Z16-25'",
    )
    is_symmetric_motif: bool = Field(
        True,
        description="If True, the input motifs are expected to be already symmetric and won't be symmetrized. \
        If False, the all input motifs are expected to be ASU and will be symmetrized.",
    )
    use_declared_frames: bool = Field(
        False,
        description=(
            "Use the frames declared by the symmetry ID instead of "
            "recovering frames from a pre-symmetrized motif. This is for "
            "compiler-owned inputs whose frame registry has already been "
            "validated, including multiple identical entities per ASU."
        ),
    )
    declared_action_is_quotient: bool = Field(
        False,
        description=(
            "The declared frames are physical coset representatives of a "
            "compiler-validated stabilizer quotient, rather than every "
            "element of the named full symmetry group."
        ),
    )
    declared_transform_order: Optional[list[str]] = Field(
        None,
        description=(
            "Ordered transform identifiers for compiler-declared symmetry "
            "frames. Required together with declared_transform_matrices "
            "when use_declared_frames is true."
        ),
    )
    declared_transform_matrices: Optional[dict[str, list[list[float]]]] = Field(
        None,
        description=(
            "Compiler-validated homogeneous 4x4 symmetry transforms keyed "
            "by declared transform identifier."
        ),
    )
    declared_preexpanded_chain_layout: Optional[list[dict]] = Field(
        None,
        description=(
            "Compiler-owned layout for an already expanded mixed-entity "
            "assembly. One record per input chain declares transform_index, "
            "entity_id and is_asu; coordinates are not expanded again."
        ),
    )


def convery_sym_conf_to_symmetry_config(sym_conf: dict) -> SymmetryConfig:
    return SymmetryConfig(**sym_conf)


def _resolve_symmetry_frames(
    sym_conf: SymmetryConfig,
    src_atom_array,
):
    """Resolve runtime frames without conflating entities with transforms."""

    if sym_conf.use_declared_frames:
        order = sym_conf.declared_transform_order
        matrices = sym_conf.declared_transform_matrices
        if not order or not matrices:
            raise ValueError(
                "use_declared_frames requires declared_transform_order and "
                "declared_transform_matrices"
            )
        if len(order) != len(matrices) or set(order) != set(matrices):
            raise ValueError(
                "Declared symmetry transform order does not cover the "
                "declared matrix set"
            )
        expected_count = get_symmetry_multiplicity_from_id(sym_conf)
        if (
            not sym_conf.declared_action_is_quotient
            and len(order) != expected_count
        ):
            raise ValueError(
                "Declared symmetry transform count does not match symmetry "
                f"ID {sym_conf.id}: {len(order)} != {expected_count}"
            )
        if sym_conf.declared_action_is_quotient and not (
            2 <= len(order) < expected_count
        ):
            raise ValueError(
                "A declared quotient action must contain at least two and "
                "fewer than the full-group number of physical frames: "
                f"{len(order)} vs full multiplicity {expected_count}"
            )
        frames = []
        for transform_index, transform_id in enumerate(order):
            matrix = np.asarray(matrices[transform_id], dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(
                    f"Declared symmetry transform {transform_id!r} must be "
                    "a finite 4x4 matrix"
                )
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
                raise ValueError(
                    f"Declared symmetry transform {transform_id!r} has an "
                    "invalid homogeneous row"
                )
            rotation = matrix[:3, :3]
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
                raise ValueError(
                    f"Declared symmetry transform {transform_id!r} is not "
                    "orthogonal"
                )
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
                raise ValueError(
                    f"Declared symmetry transform {transform_id!r} is not "
                    "a proper rotation"
                )
            if transform_index == 0 and not np.allclose(
                matrix,
                np.eye(4),
                atol=1e-6,
            ):
                raise ValueError(
                    "The first declared symmetry frame must be the identity "
                    "because runtime transform zero is the materialized ASU"
                )
            frames.append((rotation, matrix[:3, 3].copy()))
        ranked_logger.info(
            "Using compiler-declared symmetry matrices in declared order."
        )
        return frames

    frames = get_symmetry_frames_from_symmetry_id(sym_conf)
    if not sym_conf.is_symmetric_motif:
        return frames
    assert (
        src_atom_array is not None
    ), "Source atom array must be provided for symmetric motifs"
    return get_symmetry_frames_from_atom_array(src_atom_array, frames)


def make_symmetric_atom_array(
    asu_atom_array,
    sym_conf: SymmetryConfig | dict,
    sm=None,
    has_dist_cond=False,
    src_atom_array=None,
):
    """
    apply symmetry to an atom array.
    Arguments:
        asu_atom_array: atom array of the asymmetric unit
        sym_conf: symmetry configuration (dict, "id" key is required)
        sm: optional small molecule names (str, comma separated)
        has_dist_cond: whether to add 2d entity annotations
    Returns:
        new_asu_atom_array: atom array with symmetry applied
    """
    if not isinstance(sym_conf, SymmetryConfig):
        sym_conf = convery_sym_conf_to_symmetry_config(sym_conf)

    sym_conf = check_symmetry_config(
        asu_atom_array, sym_conf, sm, has_dist_cond, src_atom_array=src_atom_array
    )
    # `check_symmetry_config` is untyped (annotating its SymmetryConfig param/return
    # would be a circular import via checks.py), so mypy widens sym_conf back to the
    # declared `SymmetryConfig | dict`; it returns the same SymmetryConfig it was given.
    assert isinstance(sym_conf, SymmetryConfig)
    if sym_conf.declared_action_is_quotient and has_dist_cond:
        raise NotImplementedError(
            "Declared quotient actions are currently supported for exact "
            "3D motif conditioning, not 2D distance conditioning"
        )

    # Adding utility annotations to the asu atom array
    asu_atom_array = _add_util_annotations(asu_atom_array, sym_conf, sm)

    if has_dist_cond:  # NB: this will only work for asymmetric motifs at the moment - need to add functionality for symmetric motifs
        asu_atom_array = add_2d_entity_annotations(asu_atom_array)

    frames = _resolve_symmetry_frames(sym_conf, src_atom_array)

    preexpanded_layout = sym_conf.declared_preexpanded_chain_layout
    if preexpanded_layout is not None:
        if not sym_conf.use_declared_frames:
            raise ValueError(
                "declared_preexpanded_chain_layout requires "
                "use_declared_frames=true"
            )
        # Preserve compiler/contig chain order.  Lexicographic sorting would
        # place AA before B and silently corrupt layouts above 26 chains.
        chain_ids = list(
            dict.fromkeys(np.asarray(asu_atom_array.chain_id).tolist())
        )
        if len(preexpanded_layout) != len(chain_ids):
            raise ValueError(
                "Preexpanded chain layout length does not match parsed "
                f"input chains: {len(preexpanded_layout)} != "
                f"{len(chain_ids)}"
            )
        asu_atom_array = add_sym_annotations(asu_atom_array, sym_conf)
        transform_ids = np.full(asu_atom_array.shape[0], -1, dtype=np.int32)
        entity_ids = np.full(asu_atom_array.shape[0], -1, dtype=np.int32)
        is_asu = np.zeros(asu_atom_array.shape[0], dtype=np.bool_)
        # Biotite annotations are scalar arrays.  RFD3 therefore stores a
        # 3-vector in the structured scalar dtype returned by
        # ``decompose_symmetry_frame()``.  Using an ordinary ``(N, 3)`` array
        # here looks natural but is incompatible with the native fixed-motif
        # annotation path, which later replaces selected values with that
        # packed dtype.
        reference_origin, reference_x_axis, reference_y_axis = (
            decompose_symmetry_frame(frames[0])
        )
        origins = np.full(asu_atom_array.shape[0], reference_origin)
        x_axes = np.full(asu_atom_array.shape[0], reference_x_axis)
        y_axes = np.full(asu_atom_array.shape[0], reference_y_axis)
        entity_asu_counts: dict[int, int] = {}
        for chain_id, record in zip(
            chain_ids,
            preexpanded_layout,
            strict=True,
        ):
            transform_index = int(record["transform_index"])
            entity_id = int(record["entity_id"])
            chain_is_asu = bool(record.get("is_asu", False))
            if not 0 <= transform_index < len(frames):
                raise ValueError(
                    "Preexpanded chain layout transform_index is outside "
                    f"the declared frame set: {transform_index}"
                )
            if entity_id < 0:
                raise ValueError(
                    "Preexpanded chain layout entity_id must be nonnegative"
                )
            mask = asu_atom_array.chain_id == chain_id
            origin, x_axis, y_axis = decompose_symmetry_frame(
                frames[transform_index]
            )
            transform_ids[mask] = transform_index
            entity_ids[mask] = entity_id
            is_asu[mask] = chain_is_asu
            origins[mask] = origin
            x_axes[mask] = x_axis
            y_axes[mask] = y_axis
            if chain_is_asu:
                entity_asu_counts[entity_id] = (
                    entity_asu_counts.get(entity_id, 0) + 1
                )
        observed_entities = set(int(value) for value in np.unique(entity_ids))
        invalid_asu = {
            entity_id: entity_asu_counts.get(entity_id, 0)
            for entity_id in observed_entities
            if entity_asu_counts.get(entity_id, 0) != 1
        }
        if invalid_asu:
            raise ValueError(
                "Each preexpanded symmetry entity requires exactly one ASU "
                f"chain: {invalid_asu}"
            )
        asu_atom_array.set_annotation("sym_transform_id", transform_ids)
        asu_atom_array.set_annotation("sym_entity_id", entity_ids)
        asu_atom_array.set_annotation("is_sym_asu", is_asu)
        asu_atom_array.set_annotation("sym_transform_Ori", origins)
        asu_atom_array.set_annotation("sym_transform_X", x_axes)
        asu_atom_array.set_annotation("sym_transform_Y", y_axes)
        # Preexpanded chains were materialized independently by the compiler,
        # so their parser-level ``src_component`` labels are chain-specific.
        # Establish exact copy correspondence from local residue order and
        # atom identity instead of pretending those labels are shared.
        residue_ids = np.asarray(asu_atom_array.res_id)
        atom_names = np.asarray(asu_atom_array.atom_name)
        orbit_slots = np.full(asu_atom_array.shape[0], -1, dtype=np.int64)
        for entity_id in sorted(observed_entities):
            entity_mask = entity_ids == entity_id
            reference_keys = None
            for transform_index in sorted(
                int(value) for value in np.unique(transform_ids[entity_mask])
            ):
                indices = np.flatnonzero(
                    entity_mask & (transform_ids == transform_index)
                )
                local_residue_order = {
                    residue_id: index
                    for index, residue_id in enumerate(
                        dict.fromkeys(residue_ids[indices].tolist())
                    )
                }
                keys = tuple(
                    (
                        local_residue_order[residue_ids[index]],
                        str(atom_names[index]),
                    )
                    for index in indices
                )
                if len(keys) != len(set(keys)):
                    raise ValueError(
                        "Preexpanded symmetry chain contains duplicate "
                        f"local atom keys: entity={entity_id}, "
                        f"transform={transform_index}"
                    )
                ordered_keys = tuple(sorted(keys))
                if reference_keys is None:
                    reference_keys = ordered_keys
                elif ordered_keys != reference_keys:
                    raise ValueError(
                        "Preexpanded symmetry copies do not have identical "
                        f"local residue/atom keys: entity={entity_id}, "
                        f"transform={transform_index}"
                    )
                slot_by_key = {
                    key: slot for slot, key in enumerate(ordered_keys)
                }
                orbit_slots[indices] = np.asarray(
                    [slot_by_key[key] for key in keys],
                    dtype=np.int64,
                )
        if np.any(orbit_slots < 0):
            raise ValueError(
                "Preexpanded symmetry orbit-slot construction left "
                "unassigned atoms"
            )
        asu_atom_array.set_annotation(
            "mosaic_preexpanded_orbit_slot",
            orbit_slots,
        )
        asu_atom_array = fix_3D_sym_motif_annotations(asu_atom_array)
        asu_atom_array = add_src_sym_component_annotations(asu_atom_array)
        return _del_util_annotations(asu_atom_array)

    if not sym_conf.is_symmetric_motif:
        # At this point, asym case would have been caught by the check_symmetry_config function.
        ranked_logger.info(
            "No motifs found in atom array. Generating unconditional symmetric proteins."
        )

    # Add symmetry annotations to the asu atom array
    asu_atom_array = add_sym_annotations(asu_atom_array, sym_conf)

    # Extracting all things at this moment that we will not want to symmetrize.
    # This includes: 1) unsym motifs, 2) ligands
    unsym_atom_arrays = []
    if sym_conf.is_unsym_motif:
        # unsym_motif_atom_array = get_unsym_motif(asu_atom_array, asu_atom_array._is_unsym_motif)
        # Now remove the unsym motifs from the asu atom array
        unsym_atom_arrays.append(asu_atom_array[asu_atom_array._is_unsym_motif])
        asu_atom_array = asu_atom_array[~asu_atom_array._is_unsym_motif]
    if sm:
        unsym_atom_arrays.append(asu_atom_array[asu_atom_array._is_sm])
        asu_atom_array = asu_atom_array[~asu_atom_array._is_sm]
    unsym_atom_array = (
        struc.concatenate(unsym_atom_arrays) if len(unsym_atom_arrays) > 0 else None
    )

    # Annotate symmetric subunits
    symmetry_unit_list = []
    for transform_id, frame in enumerate(frames):
        # this is to build the fully symmetrized atom array containing all the symmetry subunits
        symmetry_unit = get_symmetry_unit(asu_atom_array, transform_id, frame)
        symmetry_unit_list.append(symmetry_unit)
    if unsym_atom_array:  # only if exists
        unsym_atom_array = annotate_unsym_atom_array(unsym_atom_array)
        symmetry_unit_list.append(
            unsym_atom_array
        )  # add the motifs to the end of the asu atom array list (motifs at end of atom array)
    # build the full symmetrized atom array
    symmetrized_atom_array = struc.concatenate(symmetry_unit_list)

    # add 2D conditioning annotations
    if has_dist_cond:
        symmetrized_atom_array = reannotate_2d_conditions(symmetrized_atom_array)

    # set all motifs to not have any symmetrization applied to them
    # TODO: this needs to be adapted to work with 2D cond (in 2D cond, we WANT to apply symmetry to the motifs since they move in space)
    symmetrized_atom_array = fix_3D_sym_motif_annotations(symmetrized_atom_array)

    # This is needed to output correct motif residue mappings in the output json
    symmetrized_atom_array = add_src_sym_component_annotations(symmetrized_atom_array)
    # remove utility annotations
    symmetrized_atom_array = _del_util_annotations(symmetrized_atom_array)
    return symmetrized_atom_array


def make_symmetric_atom_array_for_partial_diffusion(atom_array, sym_conf):
    """
    Apply symmetry to an atom array with partial diffusion.
    Arguments:
        atom_array: atom array of the asymmetric unit
        sym_conf: symmetry configuration (dict, "id" key is required)
    Returns:
        atom_array: atom array with symmetry applied
    """
    if not isinstance(sym_conf, SymmetryConfig):
        sym_conf = convery_sym_conf_to_symmetry_config(sym_conf)
    if sym_conf.declared_action_is_quotient:
        raise NotImplementedError(
            "Partial diffusion with a stabilizer quotient action is not "
            "implemented; use ordinary full-trajectory generation"
        )
    if (
        sym_conf.use_declared_frames
        and (sym_conf.id or "").strip().upper() in {"T", "O", "I"}
    ):
        raise NotImplementedError(
            "Partial diffusion for declared-frame T/O/I symmetry is not "
            "implemented; use ordinary generation or a Cn/Dn partial "
            "diffusion input"
        )

    # TODO: clean up this function

    # For partial diffusion with symmetric inputs, preserve exact positioning
    ranked_logger.info(
        "Partial diffusion with symmetry - preserving exact input coordinates"
    )
    ranked_logger.info("SKIPPING symmetry reconstruction to preserve input structure")
    # Add full symmetry annotations without changing coordinates
    from rfd3.inference.symmetry.checks import (
        check_atom_array_is_symmetric,
    )
    from rfd3.inference.symmetry.frames import (
        decompose_symmetry_frame,
    )

    check_symmetry_config(
        atom_array,
        sym_conf,
        sm=None,
        has_dist_cond=False,
        src_atom_array=None,
        partial=True,
    )

    atom_array = add_sym_annotations(atom_array, sym_conf)
    assert check_atom_array_is_symmetric(atom_array), "Atom array is not symmetric"

    n = atom_array.shape[0]
    chain_ids = np.unique(atom_array.chain_id)
    frames = get_symmetry_frames_from_symmetry_id(sym_conf)

    # Add symmetry ID
    symmetry_ids = np.full(n, sym_conf.id, dtype="U6")
    atom_array.set_annotation("symmetry_id", symmetry_ids)

    # Initialize transform annotations (use same format as original system)
    symmetry_transform_id = np.zeros(n, dtype=np.int32)
    symmetry_entity_id = np.zeros(n, dtype=np.int32)
    is_asu = np.zeros(n, dtype=bool)

    # Add transform annotations for each chain (same format as add_symmetry_transform_annotations)
    for i, chain_id in enumerate(chain_ids):
        chain_mask = atom_array.chain_id == chain_id
        transform_id = i % len(frames)  # Cycle through available frames
        frame = frames[transform_id]

        # Decompose frame to packed scalars
        Ori, X, Y = decompose_symmetry_frame(frame)

        # Set annotations for this chain (use np.full like original system)
        if i == 0:  # First chain - initialize arrays
            sym_transform_Ori = np.full(n, Ori)
            sym_transform_X = np.full(n, X)
            sym_transform_Y = np.full(n, Y)
            is_asu[chain_mask] = True
        else:  # Subsequent chains - update specific atoms
            sym_transform_Ori[chain_mask] = Ori
            sym_transform_X[chain_mask] = X
            sym_transform_Y[chain_mask] = Y

        symmetry_transform_id[chain_mask] = transform_id
        symmetry_entity_id[chain_mask] = 0  # All chains same entity for C9

    # Set all annotations
    atom_array.set_annotation("sym_transform_Ori", sym_transform_Ori)
    atom_array.set_annotation("sym_transform_X", sym_transform_X)
    atom_array.set_annotation("sym_transform_Y", sym_transform_Y)
    atom_array.set_annotation("sym_transform_id", symmetry_transform_id)
    atom_array.set_annotation("sym_entity_id", symmetry_entity_id)
    atom_array.set_annotation("is_sym_asu", is_asu)

    ranked_logger.info(
        f"Added full symmetry annotations to {len(chain_ids)} existing chains WITHOUT changing coordinates"
    )

    return atom_array


########################################################
# Private functions only used in make_symmetric_atom_array
########################################################


def _add_util_annotations(asu_atom_array, sym_conf, sm):
    """
    Add symmetry-specific utility annotations to the asu atom array.
    Arguments:
        asu_atom_array: atom array of the asymmetric unit
        sym_conf: symmetry configuration
        sm: small molecule names (str, comma separated)
    """
    n = asu_atom_array.shape[0]
    is_motif = get_motif_features(asu_atom_array)["is_motif_atom"].astype(np.bool_)
    is_sm = np.zeros(n, dtype=bool)
    is_asu = np.ones(n, dtype=bool)
    is_unsym_motif = np.zeros(n, dtype=bool)

    if sm:
        is_sm = np.logical_or.reduce(
            [
                fetch_mask_from_component(lig, atom_array=asu_atom_array)
                for lig in sm.split(",")
            ]
        )

    # assign unsym motifs
    if sym_conf.is_unsym_motif:
        unsym_motif_names = sym_conf.is_unsym_motif.split(",")
        unsym_motif_names = expand_contig_unsym_motif(unsym_motif_names)
        is_unsym_motif = get_unsym_motif_mask(asu_atom_array, unsym_motif_names)

    is_unindexed_motif = asu_atom_array.is_motif_atom_unindexed.astype(np.bool_)
    is_indexed_motif = ~is_sm & ~is_unindexed_motif & is_motif

    asu_atom_array.set_annotation(
        "_is_asu", is_asu
    )  # Currently not used but will needed for 2D cond
    asu_atom_array.set_annotation("_is_motif", is_motif)
    asu_atom_array.set_annotation("_is_sm", is_sm)
    asu_atom_array.set_annotation("_is_indexed_motif", is_indexed_motif)
    asu_atom_array.set_annotation("_is_unindexed_motif", is_unindexed_motif)
    asu_atom_array.set_annotation("_is_unsym_motif", is_unsym_motif)
    return asu_atom_array


def _del_util_annotations(aary):
    """
    Delete symmetry-specific utility annotations from the atom array.
    Arguments:
        aary: atom array
    """
    aary.del_annotation("_is_asu")  # Currently not used but will needed for 2D cond
    aary.del_annotation("_is_motif")
    aary.del_annotation("_is_sm")
    aary.del_annotation("_is_indexed_motif")
    aary.del_annotation("_is_unindexed_motif")
    aary.del_annotation("_is_unsym_motif")
    return aary


#########################
# Symmetrization functions
#########################


def center_symmetric_src_atom_array(src_atom_array):
    """
    Center the src atom array at the origin.
    Arguments:
        src_atom_array: atom array of the source
    Returns:
        src_atom_array: atom array of the source centered at the origin
    """
    # Compute COM of the src atom array (protein only elements)
    src_atom_array_com = np.mean(
        src_atom_array[src_atom_array.chain_type == 6].coord, axis=0
    )
    # center the src atom array
    src_atom_array.coord -= src_atom_array_com
    return src_atom_array


def _runtime_symmetry_features(X_L, sym_feats):
    """Normalize atomwise symmetry features onto the coordinate device.

    RFD3's data pipeline may leave atom annotations as NumPy arrays while the
    transforms are torch tensors.  Symmetry operations run in the coordinate
    dtype/device and use one convention throughout:

        x_copy = x_canonical @ R.T + T

    This is the same row-vector convention used when AtomArray copies are
    constructed in :mod:`rfd3.inference.symmetry.atom_array`.
    """

    device = X_L.device
    dtype = X_L.dtype
    sym_entity_id = torch.as_tensor(
        sym_feats["sym_entity_id"],
        dtype=torch.long,
        device=device,
    )
    sym_transform_id = torch.as_tensor(
        sym_feats["sym_transform_id"],
        dtype=torch.long,
        device=device,
    )
    is_sym_asu = torch.as_tensor(
        sym_feats["is_sym_asu"],
        dtype=torch.bool,
        device=device,
    )
    raw_orbit_slots = sym_feats.get("sym_orbit_slot")
    sym_orbit_slot = (
        None
        if raw_orbit_slots is None
        else torch.as_tensor(
            raw_orbit_slots,
            dtype=torch.long,
            device=device,
        )
    )
    if any(
        feature.ndim != 1 or feature.shape[0] != X_L.shape[-2]
        for feature in (
            sym_entity_id,
            sym_transform_id,
            is_sym_asu,
            *(() if sym_orbit_slot is None else (sym_orbit_slot,)),
        )
    ):
        raise ValueError(
            "Atomwise symmetry features must be one-dimensional and match "
            "the coordinate atom dimension"
        )

    sym_transforms = {}
    for raw_transform_id, raw_transform in sym_feats["sym_transform"].items():
        transform_id = int(raw_transform_id)
        if transform_id == FIXED_TRANSFORM_ID:
            continue
        if len(raw_transform) != 2:
            raise ValueError(
                f"Symmetry transform {transform_id} must contain (R, T)"
            )
        rotation = torch.as_tensor(
            raw_transform[0],
            dtype=dtype,
            device=device,
        )
        translation = torch.as_tensor(
            raw_transform[1],
            dtype=dtype,
            device=device,
        )
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                f"Symmetry transform {transform_id} must have shapes "
                "(3, 3) and (3,)"
            )
        sym_transforms[transform_id] = (rotation, translation)

    return (
        sym_entity_id,
        sym_transform_id,
        is_sym_asu,
        sym_orbit_slot,
        sym_transforms,
    )


def _symmetry_entity_orbits(
    sym_entity_id,
    sym_transform_id,
    is_sym_asu,
    sym_orbit_slot,
    sym_transforms,
):
    """Yield validated, position-corresponding indices for every orbit."""

    unique_entity_ids = torch.unique(sym_entity_id)
    unique_entity_ids = unique_entity_ids[
        unique_entity_ids != FIXED_ENTITY_ID
    ]
    for entity_id in unique_entity_ids.tolist():
        entity_mask = sym_entity_id == entity_id
        transform_ids = sorted(
            int(value)
            for value in torch.unique(
                sym_transform_id[entity_mask]
            ).tolist()
        )
        if not transform_ids:
            continue
        copy_indices = []
        expected_count = None
        for transform_id in transform_ids:
            if transform_id not in sym_transforms:
                raise ValueError(
                    f"Missing symmetry transform {transform_id} for "
                    f"entity {entity_id}"
                )
            transform_mask = entity_mask & (
                sym_transform_id == transform_id
            )
            indices = torch.nonzero(
                transform_mask,
                as_tuple=False,
            ).flatten()
            if sym_orbit_slot is not None:
                copy_slots = sym_orbit_slot[indices]
                order = torch.argsort(copy_slots)
                indices = indices[order]
                copy_slots = copy_slots[order]
                if not torch.equal(
                    copy_slots,
                    torch.arange(
                        len(indices),
                        dtype=torch.long,
                        device=copy_slots.device,
                    ),
                ):
                    raise ValueError(
                        "sym_orbit_slot must contain each integer from zero "
                        f"once per copy: entity={entity_id}, "
                        f"transform={transform_id}"
                    )
            count = len(indices)
            if expected_count is None:
                expected_count = count
            elif count != expected_count:
                raise ValueError(
                    "Symmetry entity subunits must contain the same number "
                    f"of atoms: entity={entity_id}, transform={transform_id}, "
                    f"expected={expected_count}, observed={count}"
                )
            copy_indices.append((transform_id, indices))

        asu_transform_ids = torch.unique(
            sym_transform_id[entity_mask & is_sym_asu]
        ).tolist()
        if len(asu_transform_ids) != 1:
            raise ValueError(
                f"Symmetry entity {entity_id} must have exactly one ASU "
                f"transform, observed {asu_transform_ids}"
            )
        asu_transform_id = int(asu_transform_ids[0])
        asu_indices = next(
            indices
            for transform_id, indices in copy_indices
            if transform_id == asu_transform_id
        )
        annotated_asu_indices = torch.nonzero(
            entity_mask & is_sym_asu,
            as_tuple=False,
        ).flatten()
        if not torch.equal(
            torch.sort(annotated_asu_indices).values,
            torch.sort(asu_indices).values,
        ):
            raise ValueError(
                f"ASU annotation does not cover the complete transform "
                f"{asu_transform_id} for entity {entity_id}"
            )
        yield entity_id, asu_transform_id, copy_indices


def _apply_frame(points, rotation, translation):
    """Apply a column-convention SE(3) frame to row-vector coordinates."""

    return torch.matmul(points, rotation.transpose(-1, -2)) + translation


def _invert_frame(points, rotation, translation):
    """Map row-vector coordinates from a symmetry copy to canonical space."""

    return torch.matmul(points - translation, rotation)


def _symmetry_work_dtype(coordinates):
    """Keep float64 diagnostics exact and promote lower precision to float32."""

    return (
        torch.float64
        if coordinates.dtype == torch.float64
        else torch.float32
    )


def symmetry_orbit_tolerance(
    coordinates: torch.Tensor,
    *,
    configured_tolerance: float,
) -> tuple[float, float]:
    """Return the absolute orbit gate and its float roundoff component.

    Exact orbit projection contains rotation/inverse-rotation round trips.
    For float32 coordinates at the large initial EDM noise scale, demanding a
    fixed sub-machine-precision residual makes tests and runtime checks depend
    on random draw order.  Keep the configured scientific gate, adding only a
    scale-aware numerical floor that becomes negligible at molecular scale.
    """

    configured = float(configured_tolerance)
    if configured <= 0.0:
        raise ValueError("configured_tolerance must be positive")
    if not torch.isfinite(coordinates).all():
        raise ValueError("Cannot derive symmetry tolerance from NaN or Inf")
    work_dtype = _symmetry_work_dtype(coordinates)
    coordinate_scale = torch.sqrt(
        torch.mean(
            torch.square(coordinates.detach().to(dtype=work_dtype))
        )
    )
    numerical_floor = (
        32.0
        * torch.finfo(work_dtype).eps
        * max(float(coordinate_scale.item()), 1.0)
    )
    return max(configured, numerical_floor), numerical_floor


def _nearest_proper_rotation(
    rotation: torch.Tensor,
    *,
    transform_id: int,
    maximum_correction: float = 1e-3,
) -> torch.Tensor:
    """Project a nearly rigid runtime frame onto SO(3).

    RFD3 stores a frame as three virtual points and reconstructs it with an
    epsilon-stabilized Gram-Schmidt pass.  Even an exact C3 frame therefore
    returns with a small scale/shear error.  Using ``R.T`` as its inverse then
    makes the orbit projector non-idempotent, and the error is amplified by
    the large initial diffusion noise.  Polar projection removes only that
    numerical frame-serialization error; transforms that require a material
    correction are rejected.
    """

    left, _, right_t = torch.linalg.svd(rotation)
    handedness = torch.linalg.det(left @ right_t)
    sign = torch.eye(
        3,
        dtype=rotation.dtype,
        device=rotation.device,
    )
    if float(handedness.item()) < 0.0:
        sign[-1, -1] = -1.0
    normalized = left @ sign @ right_t
    correction = torch.max(torch.abs(normalized - rotation))
    if float(correction.item()) > maximum_correction:
        raise ValueError(
            f"Symmetry transform {transform_id} requires an excessive "
            f"SO(3) correction ({float(correction.item()):.6g})"
        )
    return normalized


def build_symmetry_orbit_layout(
    sym_feats,
    *,
    like,
) -> SymmetryOrbitLayout:
    """Build one reusable orbit layout outside the denoising loop."""

    work_like = torch.empty(
        (1, like.shape[-2], 3),
        dtype=_symmetry_work_dtype(like),
        device=like.device,
    )
    (
        sym_entity_id,
        sym_transform_id,
        is_sym_asu,
        sym_orbit_slot,
        sym_transforms,
    ) = _runtime_symmetry_features(work_like, sym_feats)
    if sym_orbit_slot is None:
        raise ValueError(
            "Exact symmetry-orbit operations require the explicit "
            "sym_orbit_slot feature"
        )
    verified_slots = torch.as_tensor(
        sym_feats.get("sym_orbit_slot_verified", False),
        dtype=torch.bool,
        device=work_like.device,
    )
    if verified_slots.numel() != 1 or not bool(verified_slots.item()):
        raise ValueError(
            "Exact symmetry-orbit operations require atom-key-verified "
            "sym_orbit_slot correspondence"
        )
    identity = torch.eye(
        3,
        dtype=work_like.dtype,
        device=work_like.device,
    )
    # RFD3 calls the sampler from an outer bfloat16 autocast context.  The
    # C3 sine/cosine entries lose enough precision in bfloat16 to look
    # non-orthogonal at the strict runtime tolerance, so both validation and
    # projection must explicitly stay in float32/float64.
    with torch.autocast(
        device_type=work_like.device.type,
        enabled=False,
    ):
        normalized_transforms = {}
        for transform_id, (rotation, translation) in sym_transforms.items():
            if not (
                torch.isfinite(rotation).all()
                and torch.isfinite(translation).all()
            ):
                raise ValueError(
                    f"Symmetry transform {transform_id} contains NaN or Inf"
                )
            # Lightning may recursively cast feature tensors, including
            # ``sym_transform``, to bfloat16 before sampler entry.  A C3
            # rotation rounded that way has ~2e-3 orthogonality/determinant
            # error even though its nearest SO(3) correction is below 1e-3.
            # Prevalidation audits the original runtime frames strictly.
            # Here, use the bounded polar-correction test below as the single
            # acceptance gate instead of rejecting the lossy transport
            # representation before it can be normalized.
            normalized_rotation = _nearest_proper_rotation(
                rotation,
                transform_id=transform_id,
            )
            normalized_orthogonality_error = torch.max(
                torch.abs(
                    normalized_rotation @ normalized_rotation.T
                    - identity
                )
            )
            normalized_determinant_error = torch.abs(
                torch.linalg.det(normalized_rotation) - 1.0
            )
            if (
                float(normalized_orthogonality_error.item()) > 1e-5
                or float(normalized_determinant_error.item()) > 1e-5
            ):
                raise ValueError(
                    f"Symmetry transform {transform_id} could not be "
                    "normalized to a proper rotation"
                )
            normalized_transforms[transform_id] = (
                normalized_rotation,
                translation,
            )
    sym_transforms = normalized_transforms
    entity_orbits = tuple(
        (
            entity_id,
            asu_transform_id,
            tuple(copies),
        )
        for entity_id, asu_transform_id, copies
        in _symmetry_entity_orbits(
            sym_entity_id,
            sym_transform_id,
            is_sym_asu,
            sym_orbit_slot,
            sym_transforms,
        )
    )
    return SymmetryOrbitLayout(
        sym_entity_id=sym_entity_id,
        sym_transform_id=sym_transform_id,
        is_sym_asu=is_sym_asu,
        sym_orbit_slot=sym_orbit_slot,
        sym_transforms=sym_transforms,
        entity_orbits=entity_orbits,
    )


def _resolve_symmetry_orbit_layout(sym_feats, like, layout):
    if layout is None:
        return build_symmetry_orbit_layout(sym_feats, like=like)
    if layout.sym_entity_id.shape[0] != like.shape[-2]:
        raise ValueError(
            "Cached symmetry orbit layout does not match atom dimension"
        )
    if layout.sym_entity_id.device != like.device:
        raise ValueError(
            "Cached symmetry orbit layout is on a different device"
        )
    return layout


def expand_symmetry_coupled_displacements(
    displacements,
    sym_feats,
    *,
    layout: SymmetryOrbitLayout | None = None,
):
    """Copy one ASU displacement sample through each symmetry orbit.

    Translations are intentionally omitted for displacement vectors.  When
    the ASU frame is not identity, samples are first mapped into the canonical
    frame and then rotated into every target frame.  Fixed/unsymmetrized
    entities retain their input displacement and can be zeroed by the caller.
    """

    work = displacements.to(dtype=_symmetry_work_dtype(displacements))
    with torch.autocast(
        device_type=displacements.device.type,
        enabled=False,
    ):
        resolved_layout = _resolve_symmetry_orbit_layout(
            sym_feats,
            work,
            layout,
        )
        coupled = work.clone()
        for _, asu_transform_id, copies in (
            resolved_layout.entity_orbits
        ):
            asu_indices = next(
                indices
                for transform_id, indices in copies
                if transform_id == asu_transform_id
            )
            asu_rotation = resolved_layout.sym_transforms[
                asu_transform_id
            ][0]
            canonical = torch.matmul(
                work[:, asu_indices, :],
                asu_rotation,
            )
            for transform_id, target_indices in copies:
                target_rotation = resolved_layout.sym_transforms[
                    transform_id
                ][0]
                coupled[:, target_indices, :] = torch.matmul(
                    canonical,
                    target_rotation.transpose(-1, -2),
                )
    # Exact sampler states remain float32 when the model emits fp16/bfloat16;
    # casting back would quantize 20--30 A coordinates by much more than the
    # 1e-3 A symmetry-closure tolerance.
    return coupled


def project_symmetry_orbit_average(
    X_L,
    sym_feats,
    partial_diffusion=False,
    *,
    layout: SymmetryOrbitLayout | None = None,
):
    """Orthogonally average all copies in canonical orbit coordinates.

    Unlike the historical ASU-only projector, this Reynolds-style projection
    does not privilege chain/copy zero.  Every copy is inverse-transformed,
    averaged in the canonical frame, and expanded through the exact runtime
    transforms.
    """

    working_X_L = X_L.to(dtype=_symmetry_work_dtype(X_L))
    with torch.autocast(
        device_type=X_L.device.type,
        enabled=False,
    ):
        resolved_layout = _resolve_symmetry_orbit_layout(
            sym_feats,
            working_X_L,
            layout,
        )
        projected = working_X_L.clone()

        for _, _, copies in resolved_layout.entity_orbits:
            canonical_copies = []
            for transform_id, indices in copies:
                rotation, translation = resolved_layout.sym_transforms[
                    transform_id
                ]
                canonical_copies.append(
                    _invert_frame(
                        working_X_L[:, indices, :],
                        rotation,
                        translation,
                    )
                )
            canonical_mean = torch.stack(
                canonical_copies,
                dim=0,
            ).mean(dim=0)
            for transform_id, indices in copies:
                rotation, translation = resolved_layout.sym_transforms[
                    transform_id
                ]
                projected[:, indices, :] = _apply_frame(
                    canonical_mean,
                    rotation,
                    translation,
                )
    # Preserve the working precision for exact orbit states.  The legacy ASU
    # projector below retains its historical input-dtype behavior.
    return projected


def symmetry_orbit_residual(
    X_L,
    sym_feats,
    *,
    atom_mask=None,
    layout: SymmetryOrbitLayout | None = None,
):
    """Return per-batch RMS and maximum distance to the orbit-average space."""

    resolved_layout = _resolve_symmetry_orbit_layout(
        sym_feats,
        X_L.to(dtype=_symmetry_work_dtype(X_L)),
        layout,
    )
    projectable = (
        resolved_layout.sym_entity_id != FIXED_ENTITY_ID
    )
    if atom_mask is not None:
        normalized_mask = torch.as_tensor(
            atom_mask,
            dtype=torch.bool,
            device=X_L.device,
        )
        if (
            normalized_mask.ndim != 1
            or normalized_mask.shape[0] != X_L.shape[-2]
        ):
            raise ValueError(
                "symmetry residual atom_mask must have shape [L]"
            )
        projectable &= normalized_mask
    if not torch.any(projectable):
        zeros = torch.zeros(
            X_L.shape[0],
            dtype=_symmetry_work_dtype(X_L),
            device=X_L.device,
        )
        return zeros, zeros

    work = X_L.to(dtype=_symmetry_work_dtype(X_L))
    projected = project_symmetry_orbit_average(
        work,
        sym_feats,
        partial_diffusion=True,
        layout=resolved_layout,
    )
    with torch.autocast(
        device_type=X_L.device.type,
        enabled=False,
    ):
        error = torch.linalg.vector_norm(
            projected[:, projectable, :]
            - work[:, projectable, :],
            dim=-1,
        )
        rms = torch.sqrt(torch.mean(torch.square(error), dim=-1))
        maximum = torch.max(error, dim=-1).values
    return rms, maximum


def symmetry_orbit_mask_mismatch_count(
    atom_mask,
    sym_feats,
    *,
    layout: SymmetryOrbitLayout | None = None,
):
    """Count atom slots whose boolean mask is not repeated over an orbit."""

    mask = torch.as_tensor(atom_mask, dtype=torch.bool)
    dummy = torch.zeros(
        (1, mask.shape[0], 3),
        dtype=torch.float32,
        device=mask.device,
    )
    resolved_layout = _resolve_symmetry_orbit_layout(
        sym_feats,
        dummy,
        layout,
    )
    mismatches = 0
    for _, asu_transform_id, copies in resolved_layout.entity_orbits:
        reference = next(
            mask[indices]
            for transform_id, indices in copies
            if transform_id == asu_transform_id
        )
        for _, indices in copies:
            mismatches += int(
                torch.count_nonzero(mask[indices] != reference).item()
            )
    return mismatches


def apply_symmetry_to_xyz_atomwise(X_L, sym_feats, partial_diffusion=False):
    """
    Apply symmetry to the xyz coordinates.
    Arguments:
        X_L: [B, L, 3] xyz coordinates
        sym_feats: dictionary containing symmetry features (id, transform, entity_id, is_sym_asu)
    Returns:
        X_L: [B, L, 3] xyz coordinates with symmetry applied
    """
    # Work out-of-place: callers also retain X_L as the denoising reference,
    # so COM correction must not mutate it as a hidden side effect.
    output_dtype = X_L.dtype
    # ``Tensor.to()`` aliases its input when dtype/device already match.
    working_X_L = X_L.to(
        dtype=_symmetry_work_dtype(X_L)
    ).clone()
    with torch.autocast(
        device_type=X_L.device.type,
        enabled=False,
    ):
        (
            sym_entity_id,
            sym_transform_id,
            is_sym_asu,
            sym_orbit_slot,
            sym_transforms,
        ) = _runtime_symmetry_features(working_X_L, sym_feats)
        fixed_motif_mask = sym_entity_id == FIXED_ENTITY_ID
        # Preserve the historical COM correction in the legacy ASU
        # projector only.  The orbit-average projector intentionally has no
        # centering side effect.
        if not partial_diffusion and torch.any(~fixed_motif_mask):
            working_X_L[:, ~fixed_motif_mask, :] -= working_X_L[
                :, ~fixed_motif_mask, :
            ].mean(dim=1, keepdim=True)
        sym_X_L = working_X_L.clone()

        for _, asu_transform_id, copies in _symmetry_entity_orbits(
            sym_entity_id,
            sym_transform_id,
            is_sym_asu,
            sym_orbit_slot,
            sym_transforms,
        ):
            entity_asu_indices = next(
                indices
                for transform_id, indices in copies
                if transform_id == asu_transform_id
            )
            asu_xyz = working_X_L[:, entity_asu_indices, :]
            asu_rotation, asu_translation = sym_transforms[
                asu_transform_id
            ]
            canonical_asu = _invert_frame(
                asu_xyz,
                asu_rotation,
                asu_translation,
            )
            for target_id, target_indices in copies:
                rotation, translation = sym_transforms[target_id]
                sym_X_L[:, target_indices, :] = _apply_frame(
                    canonical_asu,
                    rotation,
                    translation,
                )

    return sym_X_L.to(dtype=output_dtype)
