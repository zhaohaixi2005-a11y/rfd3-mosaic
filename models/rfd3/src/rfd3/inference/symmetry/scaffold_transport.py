"""Transactional reference transport for full-scaffold rigid seed proposals."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from rfd3_mosaic.seed_stabilizer import _fit_transform
from rfd3_mosaic.validation.reference_transport import (
    certify_reference_transition,
    transport_fingerprint,
    transported_reference,
)
from rfd3_mosaic.validation.scaffold_contract import _segment_distances, contract_arrays

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

    def prepare(self, coordinates, proposed, *, projector):
        """Validate rigid poses, transport the prior and candidate, then compare."""
        target = proposed.target[0, self.fixed_indices].detach().cpu().double().numpy()
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
        contract, transforms = transported_reference(self.base, self.plan, motions)
        certificate = certify_reference_transition(self.reference.contract, contract)
        if not certificate["passed"]:
            raise ValueError(
                "Reference motion cannot certify interchain segment clearance"
            )
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
        reference = replace(
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
        increments = transforms @ np.linalg.inv(self.transforms)
        field = torch.as_tensor(
            increments, device=coordinates.device, dtype=coordinates.dtype
        )
        candidate = (
            proposed.coordinates if proposed.coordinates is not None else coordinates
        ).clone()
        ids = self.atom_residue[self.generated]
        candidate[0, self.generated] = (
            torch.einsum("nij,nj->ni", field[ids, :3, :3], candidate[0, self.generated])
            + field[ids, :3, 3]
        )
        candidate = torch.where(
            self.generated[None, :, None], candidate, proposed.target
        )
        candidate = projector(candidate, proposed.target)
        before = reference_scaffold_guidance_deficits(coordinates[0], self.reference)
        after = reference_scaffold_guidance_deficits(candidate[0], reference)
        guard = route_nonregression_check(
            before, after, tolerance=self.base["limits"]["geometry_tolerance"]
        )
        if not guard["passed"]:
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

    def diagnostics(self):
        return {
            "schema_version": 1,
            "plan_sha256": transport_fingerprint(self.plan),
            "accepted_transitions": self.accepted,
            "rejected_proposals": self.rejected,
            "final_group_transforms": self.motions,
            "semantics": "atomic_seed_reference_generated_clean_candidate; noisy states remain unconstrained",
        }
