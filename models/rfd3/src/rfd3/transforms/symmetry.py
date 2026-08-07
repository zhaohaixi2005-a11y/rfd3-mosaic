import re

import numpy as np
import torch
from atomworks.ml.transforms.base import Transform
from rfd3.inference.symmetry.frames import (
    framecoords_to_RTs,
    unpack_vector,
)


class AddSymmetryFeats(Transform):
    """
    Add atom_array symmetry features to the data features.
    Arguments:
        symmetry_features: The atom_array symmetry features to add to the data features.
    Returns:
        data: The data with the atom_array symmetry features added to the data features.
    """

    def __init__(
        self,
        symmetry_features=(
            "sym_transform_id",
            "sym_entity_id",
            "is_sym_asu",
        ),
        optional_symmetry_features=("motif_constraint_group_id",),
    ):
        self.symmetry_feats = tuple(symmetry_features)
        self.optional_symmetry_feats = tuple(optional_symmetry_features)

    def forward(self, data):
        atom_array = data["atom_array"]
        annotation_categories = set(
            atom_array.get_annotation_categories()
        )
        if "symmetry_id" in annotation_categories:
            symmetry_ids = np.unique(
                atom_array.get_annotation("symmetry_id")
            )
            if len(symmetry_ids) != 1:
                raise ValueError(
                    "Symmetric inference requires one symmetry_id, observed "
                    f"{symmetry_ids.tolist()}"
                )
            data["feats"]["symmetry_id"] = str(symmetry_ids[0])
        # Get frames from atom_array
        transforms_dict = self.make_transforms_dict(atom_array)
        data["feats"]["sym_transform"] = transforms_dict  # {str(id): tuple (R,T)}
        # Else, add symmetry features atomwise
        for feature_name in self.symmetry_feats:
            feature_array = atom_array.get_annotation(feature_name)
            data["feats"][feature_name] = feature_array
        (
            orbit_slots,
            orbit_slots_verified,
        ) = self.make_symmetry_orbit_slots(
            atom_array,
            return_verification=True,
        )
        data["feats"]["sym_orbit_slot"] = orbit_slots
        data["feats"]["sym_orbit_slot_verified"] = torch.tensor(
            orbit_slots_verified,
            dtype=torch.bool,
        )
        runtime_groups = (
            data.get("specification", {})
            .get("extra", {})
            .get("motif_constraint_groups")
        )
        runtime_orbits = (
            data.get("specification", {})
            .get("extra", {})
            .get("motif_constraint_orbits")
        )
        runtime_relations = (
            data.get("specification", {})
            .get("extra", {})
            .get("assembly_interface_relations")
        )
        if runtime_groups:
            membership = (
                self.make_motif_constraint_group_membership(
                    atom_array,
                    runtime_groups,
                )
            )
            data["feats"]["motif_constraint_group_membership"] = membership
            if runtime_orbits:
                data["feats"].update(
                    self.make_motif_constraint_orbit_features(
                        atom_array,
                        runtime_groups,
                        runtime_orbits,
                        membership,
                    )
                )
        else:
            for feature_name in self.optional_symmetry_feats:
                if feature_name not in annotation_categories:
                    continue
                feature_array = atom_array.get_annotation(feature_name)
                data["feats"][feature_name] = feature_array
                if feature_name == "motif_constraint_group_id":
                    group_ids = torch.as_tensor(
                        feature_array,
                        dtype=torch.long,
                    )
                    unique_group_ids = torch.unique(group_ids)
                    unique_group_ids = unique_group_ids[unique_group_ids >= 0]
                    data["feats"]["motif_constraint_group_membership"] = (
                        unique_group_ids[:, None] == group_ids[None, :]
                    )
        # Only output-stage geometric relations belong to diffusion-time
        # graph guidance.  Input-stage ``preserve_input`` relations are
        # already enforced by the exact motif projector and may refer to
        # legacy source-fragment identities which are intentionally merged
        # during RFD3 input materialization.  Trying to bind those identities
        # here both duplicates the hard-constraint path and breaks otherwise
        # valid legacy inputs such as LHD101.
        guidance_relations = tuple(
            relation
            for relation in (runtime_relations or ())
            if relation.get("satisfaction_stage", "input") == "output"
            and relation.get("target_geometry", {}).get("mode")
            == "geometric_constraints"
        )
        if guidance_relations:
            data["feats"].update(
                self.make_assembly_interface_relation_features(
                    atom_array,
                    guidance_relations,
                )
            )
        return data

    @staticmethod
    def make_symmetry_orbit_slots(
        atom_array,
        *,
        return_verification: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, bool]:
        """Assign atom-keyed within-copy slots for orbit correspondence.

        Native symmetry expansion normally preserves atom order, but exact
        projection must not rely on that incidental ordering.  The
        copy-invariant ``(src_component, res_id, insertion, atom_name)`` key
        identifies each atom within an entity; every transform copy must
        contain the same unique key set.  ``res_id`` is essential for generated
        contig blocks because all residues in one ``70-100`` component share
        the same ``src_component``.
        """

        transform_ids = np.asarray(
            atom_array.get_annotation("sym_transform_id")
        )
        entity_ids = np.asarray(
            atom_array.get_annotation("sym_entity_id")
        )
        categories = set(atom_array.get_annotation_categories())
        required_key_annotations = {"src_component", "atom_name"}
        missing = required_key_annotations - categories
        if missing:
            # Preserve the historical transform for generic Foundry inputs,
            # but mark the positional correspondence as unverified.  Exact
            # orbit sampling rejects that flag before diffusion.
            slots = np.full(atom_array.shape[0], -1, dtype=np.int64)
            for entity_id in sorted(
                int(value)
                for value in np.unique(entity_ids)
                if int(value) >= 0
            ):
                entity_mask = entity_ids == entity_id
                expected_count: int | None = None
                for transform_id in sorted(
                    int(value)
                    for value in np.unique(transform_ids[entity_mask])
                    if int(value) >= 0
                ):
                    indices = np.flatnonzero(
                        entity_mask & (transform_ids == transform_id)
                    )
                    if expected_count is None:
                        expected_count = len(indices)
                    elif len(indices) != expected_count:
                        raise ValueError(
                            "Symmetry copies must have equal atom counts"
                        )
                    slots[indices] = np.arange(
                        len(indices),
                        dtype=np.int64,
                    )
            result = torch.from_numpy(slots)
            return (
                (result, False)
                if return_verification
                else result
            )
        source_components = np.asarray(
            atom_array.get_annotation("src_component")
        )
        residue_ids = (
            np.asarray(atom_array.get_annotation("res_id"))
            if "res_id" in categories
            else None
        )
        insertion_codes = (
            np.asarray(atom_array.get_annotation("ins_code"))
            if "ins_code" in categories
            else None
        )
        atom_names = np.asarray(
            atom_array.get_annotation("atom_name")
        )
        slots = np.full(atom_array.shape[0], -1, dtype=np.int64)
        for entity_id in sorted(
            int(value)
            for value in np.unique(entity_ids)
            if int(value) >= 0
        ):
            entity_mask = entity_ids == entity_id
            reference_keys: tuple[tuple[str, ...], ...] | None = None
            for transform_id in sorted(
                int(value)
                for value in np.unique(transform_ids[entity_mask])
                if int(value) >= 0
            ):
                indices = np.flatnonzero(
                    entity_mask & (transform_ids == transform_id)
                )
                keys = []
                for index in indices:
                    key_parts = [str(source_components[index])]
                    if residue_ids is not None:
                        key_parts.append(str(residue_ids[index]))
                        key_parts.append(
                            (
                                str(insertion_codes[index])
                                if insertion_codes is not None
                                else ""
                            )
                        )
                    key_parts.append(str(atom_names[index]))
                    keys.append(tuple(key_parts))
                if len(keys) != len(set(keys)):
                    raise ValueError(
                        "Symmetry copy contains duplicate atom correspondence "
                        f"keys: entity={entity_id}, transform={transform_id}"
                    )
                ordered_keys = tuple(sorted(keys))
                if reference_keys is None:
                    reference_keys = ordered_keys
                elif ordered_keys != reference_keys:
                    raise ValueError(
                        "Symmetry copies do not contain the same atom "
                        f"correspondence keys: entity={entity_id}, "
                        f"transform={transform_id}"
                    )
                slot_by_key = {
                    key: slot for slot, key in enumerate(reference_keys)
                }
                slots[indices] = np.asarray(
                    [slot_by_key[key] for key in keys],
                    dtype=np.int64,
                )
        result = torch.from_numpy(slots)
        return (result, True) if return_verification else result

    @staticmethod
    def make_motif_constraint_group_membership(
        atom_array,
        groups,
    ) -> torch.Tensor:
        """Resolve hard motif groups after native symmetry expansion.

        Interface groups contain the two roles ``left`` and ``right``.  A
        conventional central motif is a single-protomer constraint and uses
        ``constraint_kind=fixed_motif`` with the sole role ``motif``.  Keeping
        these schemas explicit prevents a central motif from being disguised
        as a synthetic interface merely to reach the runtime projector.
        """

        categories = set(atom_array.get_annotation_categories())
        required = {
            "src_component",
            "sym_transform_id",
            "is_motif_atom_with_fixed_coord",
        }
        missing = required - categories
        if missing:
            raise ValueError(
                "Runtime motif constraint groups require AtomArray "
                f"annotations {sorted(missing)}"
            )

        source_components = np.asarray(
            atom_array.get_annotation("src_component")
        )
        transform_ids = np.asarray(
            atom_array.get_annotation("sym_transform_id")
        )
        fixed = np.asarray(
            atom_array.get_annotation(
                "is_motif_atom_with_fixed_coord"
            ),
            dtype=bool,
        )
        membership = np.zeros(
            (len(groups), atom_array.shape[0]),
            dtype=bool,
        )
        for group_index, group in enumerate(groups):
            constraint_kind = str(
                group.get("constraint_kind", "interface")
            )
            expected_roles = {
                "interface": {"left", "right"},
                "fixed_motif": {"motif"},
            }.get(constraint_kind)
            if expected_roles is None:
                raise ValueError(
                    f"Motif constraint group {group.get('group_id')!r} "
                    f"has unsupported constraint_kind {constraint_kind!r}"
                )
            roles_with_atoms: set[str] = set()
            for member in group.get("members", ()):
                member_mask = (
                    np.isin(
                        source_components,
                        member["src_components"],
                    )
                    & (
                        transform_ids
                        == int(member["sym_transform_id"])
                    )
                    & fixed
                )
                if not np.any(member_mask):
                    raise ValueError(
                        f"Motif constraint group {group.get('group_id')!r} "
                        "member matched no fixed atoms: "
                        f"src_components={member['src_components']!r}, "
                        f"sym_transform_id={member['sym_transform_id']}"
                    )
                membership[group_index] |= member_mask
                roles_with_atoms.add(member["role"])
            if roles_with_atoms != expected_roles:
                raise ValueError(
                    f"Motif constraint group {group.get('group_id')!r} "
                    f"with constraint_kind={constraint_kind!r} must resolve "
                    f"exactly the roles {sorted(expected_roles)!r}; observed "
                    f"{sorted(roles_with_atoms)!r}"
                )

        assigned = membership.any(axis=0)
        if np.any(assigned & ~fixed):
            raise ValueError(
                "Motif constraint groups resolved non-fixed atoms"
            )
        if np.any(fixed & ~assigned):
            raise ValueError(
                "Runtime motif constraint groups do not cover every fixed "
                "motif atom"
            )
        return torch.from_numpy(membership)

    @staticmethod
    def make_motif_constraint_orbit_features(
        atom_array,
        groups,
        orbits,
        membership: torch.Tensor,
    ) -> dict[str, object]:
        """Build stable atom slots and numeric orbit-control metadata."""

        source_components = np.asarray(
            atom_array.get_annotation("src_component")
        )
        atom_names = np.asarray(
            atom_array.get_annotation("atom_name")
        )
        transform_ids = np.asarray(
            atom_array.get_annotation("sym_transform_id")
        )
        fixed = np.asarray(
            atom_array.get_annotation(
                "is_motif_atom_with_fixed_coord"
            ),
            dtype=bool,
        )

        keys_by_group: list[tuple[tuple[str, ...], ...]] = []
        indices_by_group: list[list[int]] = []
        group_id_to_index: dict[str, int] = {}
        for group_index, group in enumerate(groups):
            group_id = str(group["group_id"])
            if group_id in group_id_to_index:
                raise ValueError(
                    f"Duplicate motif constraint group ID {group_id!r}"
                )
            group_id_to_index[group_id] = group_index
            key_to_index: dict[tuple[str, ...], int] = {}
            for member in group.get("members", ()):
                member_mask = (
                    np.isin(
                        source_components,
                        member["src_components"],
                    )
                    & (
                        transform_ids
                        == int(member["sym_transform_id"])
                    )
                    & fixed
                )
                for atom_index in np.flatnonzero(member_mask):
                    key = (
                        str(member["role"]),
                        str(member["source_fragment_id"]),
                        str(source_components[atom_index]),
                        str(atom_names[atom_index]),
                    )
                    if key in key_to_index:
                        raise ValueError(
                            f"Constraint group {group_id!r} has duplicate "
                            f"atom correspondence key {key!r}"
                        )
                    key_to_index[key] = int(atom_index)
            ordered_keys = tuple(sorted(key_to_index))
            ordered_indices = [
                key_to_index[key] for key in ordered_keys
            ]
            if set(ordered_indices) != set(
                torch.nonzero(
                    membership[group_index],
                    as_tuple=False,
                ).flatten().tolist()
            ):
                raise ValueError(
                    f"Stable atom slots for group {group_id!r} do not "
                    "match its membership mask"
                )
            keys_by_group.append(ordered_keys)
            indices_by_group.append(ordered_indices)

        maximum_group_size = max(
            (len(indices) for indices in indices_by_group),
            default=0,
        )
        group_atom_indices = torch.full(
            (len(groups), maximum_group_size),
            -1,
            dtype=torch.long,
        )
        group_atom_mask = torch.zeros_like(
            group_atom_indices,
            dtype=torch.bool,
        )
        for group_index, indices in enumerate(indices_by_group):
            group_atom_indices[
                group_index,
                : len(indices),
            ] = torch.tensor(indices, dtype=torch.long)
            group_atom_mask[group_index, : len(indices)] = True

        group_orbit_index = torch.full(
            (len(groups),),
            -1,
            dtype=torch.long,
        )
        group_orbit_transform_id = torch.full_like(
            group_orbit_index,
            -1,
        )
        master_group_indices = []
        mobility_modes = []
        orbit_bounds = []
        mobility_subspaces = []
        mobility_proposals = []
        mobility_schedules = []
        mobility_objectives = []
        constraint_orbit_ids = []
        coupling_group_ids = []
        atoms_by_orbit: list[set[int]] = []
        mobile_orbit_indices: set[int] = set()
        for orbit_index, orbit in enumerate(orbits):
            constraint_orbit_id = str(
                orbit.get("constraint_orbit_id") or f"orbit_{orbit_index}"
            )
            coupling_group_id = str(
                orbit.get("coupling_group_id")
                or constraint_orbit_id
            )
            if not constraint_orbit_id or not coupling_group_id:
                raise ValueError(
                    "Constraint orbit and coupling-group identifiers cannot "
                    "be empty"
                )
            constraint_orbit_ids.append(constraint_orbit_id)
            coupling_group_ids.append(coupling_group_id)
            orbit_group_ids = [
                str(group_id) for group_id in orbit["group_ids"]
            ]
            try:
                orbit_group_indices = [
                    group_id_to_index[group_id]
                    for group_id in orbit_group_ids
                ]
                master_group_index = group_id_to_index[
                    str(orbit["master_group_id"])
                ]
            except KeyError as error:
                raise ValueError(
                    "Motif constraint orbit references an unknown group"
                ) from error
            reference_keys = keys_by_group[master_group_index]
            for group_index in orbit_group_indices:
                if keys_by_group[group_index] != reference_keys:
                    raise ValueError(
                        "All motif groups in one constraint orbit must "
                        "have identical stable atom keys"
                    )
            transform_values = orbit["group_transform_ids"]
            if len(transform_values) != len(orbit_group_indices):
                raise ValueError(
                    "Constraint orbit group/action counts do not match"
                )
            for group_index, transform_id in zip(
                orbit_group_indices,
                transform_values,
            ):
                if group_orbit_index[group_index] >= 0:
                    raise ValueError(
                        "One motif constraint group cannot belong to "
                        "multiple constraint orbits"
                    )
                group_orbit_index[group_index] = orbit_index
                group_orbit_transform_id[group_index] = int(transform_id)
            master_group_indices.append(master_group_index)
            is_mobile = orbit.get("mobility_mode") == "orbit_rigid"
            mobility_modes.append(1 if is_mobile else 0)
            orbit_bounds.append(
                [
                    float(orbit.get("max_translation") or 0.0),
                    float(orbit.get("max_rotation_deg") or 0.0),
                ]
            )
            subspace_codes = {
                None: 0,
                "radial": 1,
                "radial_axial": 2,
                "tilt_only": 3,
                "bounded_se3": 4,
            }
            proposal_codes = {
                None: 0,
                "denoiser_fit": 1,
                "scaffold_objectives": 2,
                "hoyeung_drag_compat": 3,
            }
            subspace = orbit.get(
                "mobility_subspace",
                "bounded_se3" if is_mobile else None,
            )
            proposal = orbit.get(
                "mobility_proposal",
                "denoiser_fit" if is_mobile else None,
            )
            if subspace not in subspace_codes:
                raise ValueError(
                    f"Unknown mobility subspace {subspace!r}"
                )
            if proposal not in proposal_codes:
                raise ValueError(
                    f"Unknown mobility proposal {proposal!r}"
                )
            mobility_subspaces.append(subspace_codes[subspace])
            mobility_proposals.append(proposal_codes[proposal])
            schedule = orbit.get("mobility_schedule") or {}
            if is_mobile and schedule:
                mobility_schedules.append([
                    float(schedule.get("start_fraction", 0.10)),
                    float(schedule.get("end_fraction", 0.85)),
                    float(schedule.get("response", 0.25)),
                    float(schedule.get("max_step_translation", 0.25)),
                    float(schedule.get("max_step_rotation_deg", 1.0)),
                ])
            elif is_mobile:
                # Negative sentinel means this legacy input inherits the
                # sampler-level schedule rather than silently changing it.
                mobility_schedules.append([-1.0] * 5)
            else:
                mobility_schedules.append([0.0] * 5)
            mobility_objectives.append(
                tuple(str(value) for value in orbit.get(
                    "mobility_objectives",
                    (),
                ))
            )
            if is_mobile:
                mobile_orbit_indices.add(orbit_index)
            atoms_by_orbit.append(
                set(
                    group_atom_indices[
                        orbit_group_indices
                    ][
                        group_atom_mask[orbit_group_indices]
                    ].tolist()
                )
            )

        if torch.any(group_orbit_index < 0):
            raise ValueError(
                "Every motif constraint group must belong to an orbit"
            )
        for mobile_index in mobile_orbit_indices:
            for other_index, other_atoms in enumerate(atoms_by_orbit):
                if (
                    other_index != mobile_index
                    and atoms_by_orbit[mobile_index] & other_atoms
                ):
                    raise ValueError(
                        "A mobile motif constraint orbit cannot overlap "
                        "any other constraint orbit"
                    )

        return {
            "motif_constraint_group_atom_indices": group_atom_indices,
            "motif_constraint_group_atom_mask": group_atom_mask,
            "motif_constraint_group_orbit_index": group_orbit_index,
            "motif_constraint_group_orbit_transform_id": (
                group_orbit_transform_id
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor(
                master_group_indices,
                dtype=torch.long,
            ),
            "motif_constraint_orbit_mobility_mode": torch.tensor(
                mobility_modes,
                dtype=torch.long,
            ),
            "motif_constraint_orbit_bounds": torch.tensor(
                orbit_bounds,
                dtype=torch.float32,
            ),
            "motif_constraint_orbit_subspace": torch.tensor(
                mobility_subspaces,
                dtype=torch.long,
            ),
            "motif_constraint_orbit_proposal": torch.tensor(
                mobility_proposals,
                dtype=torch.long,
            ),
            "motif_constraint_orbit_schedule": torch.tensor(
                mobility_schedules,
                dtype=torch.float32,
            ),
            "motif_constraint_orbit_objective_ids": tuple(
                mobility_objectives
            ),
            "motif_constraint_orbit_ids": tuple(constraint_orbit_ids),
            "motif_constraint_orbit_component_ids": tuple(
                coupling_group_ids
            ),
        }

    @staticmethod
    def make_assembly_interface_relation_features(
        atom_array,
        relations,
    ) -> dict[str, object]:
        """Resolve compiler-declared graph edges to atomwise runtime masks.

        The compiler already expands every edge through the complete group
        action.  Runtime guidance therefore consumes concrete left/right
        memberships instead of reinterpreting YAML selectors or inventing a
        second symmetry-neighbour convention.
        """

        categories = set(atom_array.get_annotation_categories())
        required_annotations = {"src_component", "sym_transform_id"}
        missing = required_annotations - categories
        if missing:
            raise ValueError(
                "Assembly interface guidance requires AtomArray annotations "
                f"{sorted(missing)}"
            )
        source_components = np.asarray(
            atom_array.get_annotation("src_component")
        )
        transform_ids = np.asarray(
            atom_array.get_annotation("sym_transform_id")
        )
        left_masks = []
        right_masks = []
        edge_ids = []
        source_interface_ids = []
        modes = []
        required_flags = []
        contact_minima = []
        contact_cutoffs = []
        coverage_minima = []
        contiguous_minima = []
        automatic_quality_flags = []
        distance_targets = []
        distance_tolerances = []
        satisfaction_stages = []

        def expanded_source_components(values) -> list[str]:
            expanded: list[str] = []
            for value in values:
                text = str(value)
                match = re.fullmatch(r"([^0-9]+)(\d+)-(\d+)", text)
                if match is None:
                    expanded.append(text)
                    continue
                chain, start_text, end_text = match.groups()
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(
                        "Assembly interface relation contains a reversed "
                        f"source selector: {text!r}"
                    )
                expanded.extend(
                    f"{chain}{residue}"
                    for residue in range(start, end + 1)
                )
            return expanded

        for relation in relations:
            left = np.isin(
                source_components,
                expanded_source_components(
                    relation["left_source_components"]
                ),
            ) & (transform_ids == int(relation["source_copy_index"]))
            right = np.isin(
                source_components,
                expanded_source_components(
                    relation["right_source_components"]
                ),
            ) & (transform_ids == int(relation["target_copy_index"]))
            if not np.any(left) or not np.any(right):
                raise ValueError(
                    "Assembly interface relation matched no runtime atoms: "
                    f"{relation.get('edge_instance_id')!r}"
                )
            if np.any(left & right):
                raise ValueError(
                    "Assembly interface relation sides overlap: "
                    f"{relation.get('edge_instance_id')!r}"
                )
            geometry = relation["target_geometry"]
            mode = str(geometry["mode"])
            if mode == "reference_transform":
                minimum_contacts = int(
                    geometry.get("minimum_heavy_atom_contacts", 0)
                )
                cutoff = float(geometry.get("contact_cutoff", 4.5))
                distance_target = float("nan")
                distance_tolerance = float("nan")
                minimum_coverage = 0
                minimum_contiguous = 0
                automatic_quality = False
                mode_index = 0
            elif mode == "geometric_constraints":
                contacts = geometry.get("contacts") or {}
                coverage = geometry.get("coverage") or {}
                distance = geometry.get("distance") or {}
                minimum_contacts = int(
                    contacts.get("min_heavy_atom_contacts", 0)
                )
                cutoff = float(contacts.get("cutoff", 8.0))
                distance_target = float(distance.get("target", float("nan")))
                distance_tolerance = float(
                    distance.get("tolerance", float("nan"))
                )
                minimum_coverage = int(
                    coverage.get("minimum_contact_residues_per_side") or 0
                )
                minimum_contiguous = int(
                    coverage.get(
                        "minimum_contiguous_contact_residues_per_side"
                    )
                    or 0
                )
                automatic_quality = coverage.get("mode") == "auto"
                mode_index = 1
            else:
                raise ValueError(
                    f"Unsupported runtime interface mode {mode!r}"
                )
            left_masks.append(left)
            right_masks.append(right)
            edge_ids.append(str(relation["edge_instance_id"]))
            source_interface_ids.append(
                str(
                    relation.get("source_interface_id")
                    or str(relation["edge_instance_id"]).split("@", 1)[0]
                )
            )
            modes.append(mode_index)
            required_flags.append(bool(relation.get("required", True)))
            contact_minima.append(minimum_contacts)
            contact_cutoffs.append(cutoff)
            coverage_minima.append(minimum_coverage)
            contiguous_minima.append(minimum_contiguous)
            automatic_quality_flags.append(automatic_quality)
            distance_targets.append(distance_target)
            distance_tolerances.append(distance_tolerance)
            satisfaction_stages.append(
                str(relation.get("satisfaction_stage", "input"))
            )

        return {
            "assembly_interface_left_membership": torch.from_numpy(
                np.stack(left_masks).astype(bool)
            ),
            "assembly_interface_right_membership": torch.from_numpy(
                np.stack(right_masks).astype(bool)
            ),
            "assembly_interface_mode": torch.tensor(
                modes,
                dtype=torch.long,
            ),
            "assembly_interface_required": torch.tensor(
                required_flags,
                dtype=torch.bool,
            ),
            "assembly_interface_minimum_contacts": torch.tensor(
                contact_minima,
                dtype=torch.long,
            ),
            "assembly_interface_contact_cutoff": torch.tensor(
                contact_cutoffs,
                dtype=torch.float32,
            ),
            "assembly_interface_minimum_residues_per_side": torch.tensor(
                coverage_minima,
                dtype=torch.long,
            ),
            "assembly_interface_minimum_contiguous_residues_per_side": (
                torch.tensor(contiguous_minima, dtype=torch.long)
            ),
            "assembly_interface_automatic_quality": torch.tensor(
                automatic_quality_flags,
                dtype=torch.bool,
            ),
            "assembly_interface_distance_target": torch.tensor(
                distance_targets,
                dtype=torch.float32,
            ),
            "assembly_interface_distance_tolerance": torch.tensor(
                distance_tolerances,
                dtype=torch.float32,
            ),
            "assembly_interface_ids": tuple(edge_ids),
            "assembly_interface_source_ids": tuple(source_interface_ids),
            "assembly_interface_satisfaction_stages": tuple(
                satisfaction_stages
            ),
        }

    def make_transforms_dict(self, atom_array):
        transform_ids = np.asarray(
            atom_array.get_annotation("sym_transform_id")
        )
        origins = atom_array.get_annotation("sym_transform_Ori")
        x_axes = atom_array.get_annotation("sym_transform_X")
        y_axes = atom_array.get_annotation("sym_transform_Y")

        unique_transform_ids = sorted(
            int(value)
            for value in np.unique(transform_ids)
            if int(value) != -1
        )
        frame_origins: list[list[float]] = []
        frame_x_axes: list[list[float]] = []
        frame_y_axes: list[list[float]] = []
        for transform_id in unique_transform_ids:
            indices = np.flatnonzero(transform_ids == transform_id)
            unpacked_origins = np.asarray(
                [unpack_vector(origins[index]) for index in indices]
            )
            unpacked_x_axes = np.asarray(
                [unpack_vector(x_axes[index]) for index in indices]
            )
            unpacked_y_axes = np.asarray(
                [unpack_vector(y_axes[index]) for index in indices]
            )
            for name, values in (
                ("origin", unpacked_origins),
                ("x-axis", unpacked_x_axes),
                ("y-axis", unpacked_y_axes),
            ):
                if not np.allclose(values, values[0], atol=1e-6):
                    raise ValueError(
                        f"Symmetry transform {transform_id} has inconsistent "
                        f"{name} annotations"
                    )
            frame_origins.append(unpacked_origins[0].tolist())
            frame_x_axes.append(unpacked_x_axes[0].tolist())
            frame_y_axes.append(unpacked_y_axes[0].tolist())

        Oris = torch.tensor(frame_origins)
        Xs = torch.tensor(frame_x_axes)
        Ys = torch.tensor(frame_y_axes)
        Rs, Ts = framecoords_to_RTs(Oris, Xs, Ys)

        return {
            str(transform_id): (R, T)
            for transform_id, R, T in zip(unique_transform_ids, Rs, Ts)
        }
