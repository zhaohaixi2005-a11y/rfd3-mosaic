"""Transactional reference transport for full-scaffold rigid seed proposals."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from rfd3_mosaic.seed_stabilizer import _fit_transform
from rfd3_mosaic.validation.reference_transport import (
    certify_reference_transition,
    transport_fingerprint,
    transported_reference,
)
from rfd3_mosaic.validation.scaffold_contract import _segment_distances, contract_arrays

from .differentiable_transport import (
    interpolate_se3_batch,
    rigid_landmarks,
    rigid_transform,
)
from .exact_reference_trace import ExactReferenceTrace
from .reference_scaffold import reference_scaffold_guidance_deficits
from .scaffold_core_guidance import route_nonregression_check


class ScaffoldReferenceTransport:
    """Prepare a coupled proposal without mutation; commit only after all guards."""

    def __init__(self, features, topology):
        self.plan = features["mosaic_reference_transport"]
        self.base = topology.scaffold_contract.contract
        if self.base["schema_version"] != 2:
            raise ValueError("Scaffold mobility requires backbone contract version 2")
        raw = features.get("mosaic_transport_fixed_atom_indices")
        if raw is None:
            raise ValueError(
                "Reference transport lacks validated fixed atom identity binding"
            )
        self.fixed_indices = torch.as_tensor(
            raw, device=topology.atom_to_token.device, dtype=torch.long
        )
        if len(self.fixed_indices) != len(self.plan["fixed_atoms"]):
            raise ValueError("Reference transport fixed atom count mismatch")
        self.reference = topology.scaffold_contract
        self.transforms = np.repeat(np.eye(4)[None], len(self.base["residues"]), axis=0)
        self.motions = {name: np.eye(4).tolist() for name in self.plan["groups"]}
        self.accepted, self.rejected = [], []
        self.proposal_attempts = []
        self._exact_reference_trace = ExactReferenceTrace.from_environment()
        self._trace_plan_sha256 = (
            transport_fingerprint(self.plan)
            if self._exact_reference_trace is not None else None
        )
        initial = torch.tensor(
            [record["coordinate"] for record in self.plan["fixed_atoms"]],
            dtype=torch.float64,
        )
        self._group_landmarks = {}
        for name, group in self.plan["groups"].items():
            indices = torch.tensor(group["atom_indices"], dtype=torch.long)
            self._group_landmarks[name] = indices[rigid_landmarks(initial[indices])]
        self._initial_fixed = initial
        self._binding_groups = {}
        for index, binding in enumerate(self.plan["residue_transforms"]):
            key = binding["left"], binding["right"]
            indices, fractions = self._binding_groups.setdefault(key, ([], []))
            indices.append(index)
            fractions.append(binding["fraction"])
        # Every generated atom inherits its residue's one proper transform,
        # including atoms beyond N/CA/C/O if present in the native representation.
        tokens = topology.atom_to_token[self.reference.ca_atom_indices]
        token_to_residue = torch.full(
            (int(topology.atom_to_token.max()) + 1,),
            -1,
            dtype=torch.long,
            device=tokens.device,
        )
        token_to_residue[tokens] = torch.arange(len(tokens), device=tokens.device)
        self.atom_residue = token_to_residue[topology.atom_to_token]
        self.generated = topology.generated_atom_mask
        if bool(torch.any(self.atom_residue[self.generated] < 0)):
            raise ValueError(
                "Reference transport cannot identify a generated atom's residue"
            )

    def candidate_coordinates(self, target, scaffold, *, projector=None, before=None):
        """Score the commit path with a continuous rigid-pose surrogate derivative.

        Inputs and output are [L, 3]. ``scaffold`` is in the CURRENT accepted
        reference frame, including any tentative packing update. The returned
        ``before`` supplies the accepted baseline when ``scaffold`` contains a
        tentative packing update. The returned state is for scoring only;
        prepare() independently validates and
        transports the unmodified proposal once before committing it.
        """
        if target.ndim != 2 or target.shape != scaffold.shape:
            raise ValueError("Transport objective expects matching [L, 3] coordinates")
        scaffold = scaffold.to(target)
        source = self._initial_fixed.to(target)
        selected = target[self.fixed_indices.to(target.device)]
        motions = {
            name: rigid_transform(source[indices], selected[indices])
            for name, raw_indices in self._group_landmarks.items()
            for indices in [raw_indices.to(target.device)]
        }
        transforms = target.new_empty((len(self.plan["residue_transforms"]), 4, 4))
        for (left, right), (indices, fractions) in self._binding_groups.items():
            transforms[indices] = (
                motions[left]
                if left == right
                else interpolate_se3_batch(
                    motions[left], motions[right], target.new_tensor(fractions)
                )
            )
        increments = transforms @ torch.as_tensor(
            np.linalg.inv(self.transforms), dtype=target.dtype, device=target.device
        )
        generated = self.generated.to(target.device)
        ids = self.atom_residue.to(target.device)[generated]
        moved = (
            torch.einsum("nij,nj->ni", increments[ids, :3, :3], scaffold[generated])
            + increments[ids, :3, 3]
        )
        candidate = target.clone()
        candidate[generated] = moved
        if projector is not None:
            candidate = projector(candidate[None], target[None])[0]
        # The three-landmark map supplies a smooth rigid-pose derivative.
        # Its float32 primal differs from the all-atom fit used by prepare().
        # Score the commit path's exact primal, retaining only that derivative;
        # no guard or transported reference is changed by this correction.
        with torch.no_grad():
            baseline = scaffold if before is None else before.to(scaffold)
            prepared = self._construct_candidate(
                baseline.detach()[None],
                SimpleNamespace(target=target.detach()[None], coordinates=scaffold.detach()[None]),
                projector=projector if projector is not None else lambda x, t: x,
                allow_noop=before is not None,
            )
            exact, reference = prepared[:2]
            exact, _, _, _ = self._restore_geometry(
                baseline.detach(), exact[0], reference, target.detach(), projector,
            )
        return exact + (candidate - candidate.detach())

    def _fit_motions(self, target):
        """Use the commit path's all-atom, float64 pose fit."""
        target = target[self.fixed_indices].detach().cpu().double().numpy()
        initial = np.asarray([r["coordinate"] for r in self.plan["fixed_atoms"]])
        motions = {}
        for name, group in self.plan["groups"].items():
            ids = group["atom_indices"]
            rotation, translation, _ = _fit_transform(initial[ids], target[ids])
            error = np.linalg.norm(
                initial[ids] @ rotation.T + translation - target[ids], axis=-1
            ).max()
            if error > self.base["limits"]["fixed_ca_tolerance"]:
                raise ValueError(
                    "A proposed joint seed is not one preserved rigid body"
                )
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = rotation, translation
            motions[name] = matrix.tolist()
        return motions

    def _reference_from_contract(self, contract, coordinates):
        xyz, _, chains, _ = contract_arrays(contract)
        separations = {
            (i, j): float(
                _segment_distances(
                    xyz[a[:-1]], xyz[a[1:]], xyz[b[:-1]], xyz[b[1:]]
                ).min()
            )
            for i, a in enumerate(chains)
            for j, b in enumerate(chains)
            if j > i
        }
        return replace(
            self.reference,
            contract=contract,
            reference_ca=torch.as_tensor(
                xyz, device=coordinates.device, dtype=coordinates.dtype
            ),
            reference_backbone=torch.as_tensor(
                [r["reference_backbone"] for r in contract["residues"]],
                device=coordinates.device,
                dtype=coordinates.dtype,
            ),
            reference_separations=separations,
        )

    def _restore_geometry(self, before, candidate, reference, target, projector):
        """Try a tiny coordinate trust region, preserving reference and guards.

        This is a real generated-coordinate update, not a residual correction.
        Project every trial, then require the original complete-scaffold guard.
        The trust region is numerical in size; it is not an error certificate.
        """
        def deficits(value, prior):
            # Evaluate the actual stored state in double; this does not undo
            # quantization or substitute an intended pre-cast state.
            with torch.autocast(device_type=value.device.type, enabled=False):
                return reference_scaffold_guidance_deficits(value.double(), prior)

        residual_before = deficits(before, self.reference)
        residual_after = deficits(candidate, reference)
        tolerance = self.base["limits"]["geometry_tolerance"]
        def assess(value, residual):
            report = route_nonregression_check(residual_before, residual, tolerance=tolerance)
            if value.dtype != torch.float64:
                with torch.autocast(device_type=value.device.type, enabled=False):
                    native = route_nonregression_check(
                        reference_scaffold_guidance_deficits(before, self.reference),
                        reference_scaffold_guidance_deficits(value, reference),
                        tolerance=tolerance,
                    )
                report["native_stored_dtype_guard"] = native
                report["passed"] = report["passed"] and native["passed"]
            return report

        guard = assess(candidate, residual_after)
        if guard["passed"] or projector is None:
            return candidate, residual_before, residual_after, guard
        indices = reference.backbone_atom_indices.reshape(-1)
        selected = self.generated[indices]
        indices = indices[selected]
        desired = reference.reference_backbone.reshape(-1, 3)[selected].to(candidate)
        budget = min(
            8 * torch.finfo(candidate.dtype).eps * max(float(candidate.abs().max()), 1.0),
            2e-5,
        )
        for fraction in (1e-6, 3e-6, 1e-5):
            trial = candidate.clone()
            trial[indices] = torch.lerp(candidate[indices], desired, fraction)
            trial = projector(trial[None], target[None])[0]
            if (
                not bool(torch.isfinite(trial).all())
                or float(torch.linalg.vector_norm(trial - candidate, dim=-1).max()) > budget
                or not torch.equal(trial[self.fixed_indices], target[self.fixed_indices])
            ):
                continue
            residual = deficits(trial, reference)
            check = assess(trial, residual)
            if check["passed"]:
                return trial, residual_before, residual, check
        return candidate, residual_before, residual_after, guard

    def _construct_candidate(self, coordinates, proposed, *, projector, allow_noop=True):
        """Construct one coupled stored state, with no mutations or guard waiver."""
        # An unchanged accepted state has no reference-frame transition.
        # Refitting it against serialized, unrounded landmarks otherwise
        # invents a tiny motion and can reject even an identity proposal.
        source = proposed.coordinates if proposed.coordinates is not None else coordinates
        if (
            allow_noop
            and source.shape == coordinates.shape
            and torch.equal(source, coordinates)
            and torch.equal(
                proposed.target[:, self.fixed_indices], coordinates[:, self.fixed_indices]
            )
            and bool(torch.isfinite(coordinates).all())
        ):
            certificate = certify_reference_transition(
                self.reference.contract, self.reference.contract
            )
            if not certificate["passed"]:
                raise ValueError("Reference motion cannot certify interchain segment clearance")
            return (
                coordinates.clone(), self.reference, self.transforms.copy(),
                {name: np.asarray(matrix).tolist() for name, matrix in self.motions.items()},
                certificate,
            )
        motions = self._fit_motions(proposed.target[0])
        contract, transforms = transported_reference(self.base, self.plan, motions)
        certificate = certify_reference_transition(self.reference.contract, contract)
        if not certificate["passed"]:
            raise ValueError(
                "Reference motion cannot certify interchain segment clearance"
            )
        reference = self._reference_from_contract(contract, coordinates)
        increments = transforms @ np.linalg.inv(self.transforms)
        field = torch.as_tensor(
            increments, device=coordinates.device, dtype=coordinates.dtype
        )
        candidate = torch.as_tensor(
            proposed.coordinates if proposed.coordinates is not None else coordinates,
            dtype=coordinates.dtype,
            device=coordinates.device,
        ).clone()
        ids = self.atom_residue[self.generated]
        candidate[0, self.generated] = (
            torch.einsum("nij,nj->ni", field[ids, :3, :3], candidate[0, self.generated])
            + field[ids, :3, 3]
        )
        candidate = torch.where(
            self.generated[None, :, None], candidate, proposed.target.to(candidate)
        )
        candidate = projector(candidate, proposed.target)
        return candidate, reference, transforms, motions, certificate

    def prepare(self, coordinates, proposed, *, projector):
        """Validate the exact same stored candidate used by scoring."""
        prepared = self._construct_candidate(coordinates, proposed, projector=projector)
        candidate, reference, transforms, motions, certificate = prepared
        if (
            reference is self.reference
            and torch.equal(candidate, coordinates)
        ):
            return prepared
        restored, before, after, guard = self._restore_geometry(
            coordinates[0], candidate[0], reference, proposed.target[0], projector,
        )
        repair_delta = float((restored - candidate[0]).abs().max())
        candidate = restored[None]
        certificate["candidate_geometry_guard"] = guard
        certificate["candidate_coordinate_restoration"] = {
            "applied": repair_delta > 0,
            "maximum_component_change_angstrom": repair_delta,
            "maximum_atom_displacement_angstrom": float(
                torch.linalg.vector_norm(restored - prepared[0][0], dim=-1).max()
            ),
            "absolute_trust_region_angstrom": 2e-5,
            "semantics": "actual_generated_coordinate_trust_region; reference_and_guard_unchanged",
        }
        if not guard["passed"]:
            if self._exact_reference_trace is not None:
                self._exact_reference_trace.record_rejection(
                    coordinates_before=coordinates[0], coordinates_after=candidate[0],
                    residual_before=before, residual_after=after,
                    reference_before=self.reference, reference_after=reference,
                    guard=guard,
                    geometry_tolerance=self.base["limits"]["geometry_tolerance"],
                    plan_sha256=self._trace_plan_sha256,
                )
            raise ValueError(
                f"Transported clean candidate regresses the complete-scaffold contract: {guard}"
            )
        return candidate, reference, transforms, motions, certificate

    def commit(self, prepared, *, progress):
        _, self.reference, self.transforms, self.motions, certificate = prepared
        self.accepted.append(
            {
                "progress": float(progress),
                "group_transforms": self.motions,
                "reference_transition": certificate,
            }
        )

    def candidate_validator(self, coordinates, *, projector, geometry_guard):
        """Check the state that would actually be committed by transport.

        A seed-only trial can stretch a junction that stays intact when its
        generated scaffold follows the accepted reference motion. Validating
        that frozen-scaffold intermediate incorrectly rejects feasible moves.
        Preparation is read-only, so line-search trials cannot update the
        reference or leak tentative poses into subsequent candidates.
        """

        def validate(candidate):
            try:
                prepared = self.prepare(
                    coordinates,
                    SimpleNamespace(target=candidate, coordinates=candidate),
                    projector=projector,
                )
            except ValueError as error:
                return {
                    "accepted": False,
                    "evaluation_state": "transported_seed_reference_scaffold",
                    "reason": str(error),
                }
            report = geometry_guard(prepared[0])
            return {
                **report,
                "evaluation_state": "transported_seed_reference_scaffold",
            }

        return validate

    def record_proposal_attempt(
        self, *, progress, before, after, proposed, committed, reason=None
    ):
        """Keep rejected proposal evidence separate from committed pose state."""
        previous = before.diagnostics()
        current = after.diagnostics()
        self.proposal_attempts.append(
            {
                "progress": float(progress),
                "proposed": bool(proposed),
                "committed": bool(committed),
                "reason": reason,
                "controller_update_calls": (
                    current.get("update_calls", 0) - previous.get("update_calls", 0)
                ),
                "attempted_controller_trajectory": current.get("trajectory", [])[
                    len(previous.get("trajectory", [])) :
                ],
                "semantics": "tentative proposal diagnostics; committed flag is authoritative",
            }
        )

    def diagnostics(self):
        return {
            "schema_version": 1,
            "plan_sha256": transport_fingerprint(self.plan),
            "accepted_transitions": self.accepted,
            "rejected_proposals": self.rejected,
            "proposal_attempts": self.proposal_attempts,
            "exact_reference_trace": (
                self._exact_reference_trace.diagnostics()
                if self._exact_reference_trace is not None else {"enabled": False}
            ),
            "final_group_transforms": self.motions,
            "semantics": "atomic_seed_reference_generated_clean_candidate; noisy states remain unconstrained",
            "proposal_objective_semantics": {
                "scaffold_boundary": "differentiable_coupled_seed_reference_scaffold",
                "denoiser": "rigid_fit_to_denoiser_prediction",
                "transport_map": "same principal screw map as independently validated commit",
            },
            "acceptance_evaluation_state": "coupled_transported_seed_reference_generated_coordinates",
            "acceptance_semantics": (
                "rigid seed and bounded symmetric poses; certified reference transition; "
                "reference/backbone residual nonregression; physical geometry guards"
            ),
        }
