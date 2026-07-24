"""Model-independent geometry checks for generated protein scaffolds."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from rfd3_mosaic.structure import AtomRecord


def audit_scaffold_geometry(
    atoms: tuple[AtomRecord, ...],
    *,
    min_cn_distance: float = 1.0,
    max_cn_distance: float = 2.0,
    max_ca_step: float = 4.5,
    max_chain_ca_rg: float = 25.0,
    ca_clash_distance: float = 3.0,
) -> dict[str, Any]:
    """Audit continuity, compactness and coarse CA clashes."""

    residues: dict[
        tuple[str, int, str], dict[str, np.ndarray]
    ] = defaultdict(dict)
    for atom in atoms:
        if atom.record_type != "ATOM":
            continue
        name = atom.atom_name.upper()
        if name in {"N", "CA", "C"}:
            residues[atom.residue_id][name] = np.asarray(
                atom.coordinate, dtype=float
            )

    by_chain: dict[str, list[tuple[tuple[str, int, str], dict[str, np.ndarray]]]] = (
        defaultdict(list)
    )
    for residue_id, backbone in residues.items():
        by_chain[residue_id[0]].append((residue_id, backbone))
    for chain_residues in by_chain.values():
        chain_residues.sort(key=lambda item: (item[0][1], item[0][2]))

    chains: list[dict[str, Any]] = []
    ca_records: list[tuple[str, int, np.ndarray]] = []
    for chain_id, chain_residues in sorted(by_chain.items()):
        cn_distances: list[float] = []
        ca_steps: list[float] = []
        breaks: list[dict[str, Any]] = []
        ca_coordinates: list[np.ndarray] = []
        for index, (residue_id, backbone) in enumerate(chain_residues):
            if "CA" in backbone:
                ca_coordinates.append(backbone["CA"])
                ca_records.append((chain_id, index, backbone["CA"]))
            if index == 0:
                continue
            previous_id, previous = chain_residues[index - 1]
            cn = (
                float(np.linalg.norm(previous["C"] - backbone["N"]))
                if "C" in previous and "N" in backbone
                else None
            )
            ca = (
                float(np.linalg.norm(previous["CA"] - backbone["CA"]))
                if "CA" in previous and "CA" in backbone
                else None
            )
            if cn is not None:
                cn_distances.append(cn)
            if ca is not None:
                ca_steps.append(ca)
            failed = (
                cn is None
                or ca is None
                or not min_cn_distance <= cn <= max_cn_distance
                or ca > max_ca_step
            )
            if failed:
                breaks.append(
                    {
                        "previous_residue": previous_id[1],
                        "next_residue": residue_id[1],
                        "cn_distance": cn,
                        "ca_distance": ca,
                    }
                )
        ca_array = np.asarray(ca_coordinates, dtype=float)
        ca_rg = (
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (ca_array - ca_array.mean(axis=0)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if len(ca_array)
            else float("inf")
        )
        chains.append(
            {
                "chain_id": chain_id,
                "residue_count": len(chain_residues),
                "ca_count": len(ca_coordinates),
                "ca_radius_of_gyration": ca_rg,
                "maximum_cn_distance": max(cn_distances, default=None),
                "maximum_ca_step": max(ca_steps, default=None),
                "chain_breaks": breaks,
                "passed_continuity": not breaks,
                "passed_compactness": ca_rg <= max_chain_ca_rg,
            }
        )

    ca_clashes: list[dict[str, Any]] = []
    for left_index, (left_chain, left_seq, left_coord) in enumerate(ca_records):
        for right_chain, right_seq, right_coord in ca_records[left_index + 1 :]:
            if left_chain == right_chain and abs(left_seq - right_seq) <= 2:
                continue
            distance = float(np.linalg.norm(left_coord - right_coord))
            if distance < ca_clash_distance:
                ca_clashes.append(
                    {
                        "left_chain": left_chain,
                        "left_index": left_seq,
                        "right_chain": right_chain,
                        "right_index": right_seq,
                        "distance": distance,
                    }
                )

    passed_continuity = bool(chains) and all(
        chain["passed_continuity"] for chain in chains
    )
    passed_compactness = bool(chains) and all(
        chain["passed_compactness"] for chain in chains
    )
    passed_clashes = not ca_clashes
    return {
        "audit": "rfd3_mosaic.scaffold_geometry",
        "schema_version": 1,
        "passed": (
            passed_continuity and passed_compactness and passed_clashes
        ),
        "summary": {
            "chain_count": len(chains),
            "chain_break_count": sum(
                len(chain["chain_breaks"]) for chain in chains
            ),
            "maximum_chain_ca_radius_of_gyration": max(
                (
                    chain["ca_radius_of_gyration"]
                    for chain in chains
                ),
                default=float("inf"),
            ),
            "ca_clash_count": len(ca_clashes),
            "passed_continuity": passed_continuity,
            "passed_compactness": passed_compactness,
            "passed_clashes": passed_clashes,
        },
        "thresholds": {
            "min_cn_distance": min_cn_distance,
            "max_cn_distance": max_cn_distance,
            "max_ca_step": max_ca_step,
            "max_chain_ca_radius_of_gyration": max_chain_ca_rg,
            "ca_clash_distance": ca_clash_distance,
        },
        "chains": chains,
        "ca_clashes": ca_clashes,
    }
