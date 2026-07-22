"""Compile standalone Interface-Seed artifacts into an RFD3 input JSON."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rfd3_mosaic.compile import (
    expand_symmetry_instances,
    load_interface_seed_config,
)
from rfd3_mosaic.geometry import (
    build_transform_registry,
    validate_group_closure,
)
from rfd3_mosaic.output.standalone import compile_standalone
from rfd3_mosaic.schema.specs import SymmetryType


@dataclass(frozen=True)
class RFD3AdapterOutputs:
    """Artifacts needed to prevalidate and run one native RFD3 design."""

    input_path: Path
    structure_path: Path
    mapping_path: Path
    manifest_path: Path
    example_id: str
    contig: str


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_sample_overrides(
    manifest_path: Path,
    *,
    config_path: Path,
) -> tuple[dict[str, dict[str, Any]], int | None, str]:
    """Recover exact joint unit samples and validate config provenance."""

    manifest = _load_json(manifest_path)
    candidate_config = manifest.get("config", {})
    expected_config_sha = candidate_config.get("sha256")
    actual_config_sha = _sha256(config_path)
    if expected_config_sha != actual_config_sha:
        raise ValueError(
            "Pose candidate config SHA256 does not match the requested "
            f"config: {expected_config_sha!r} != {actual_config_sha!r}"
        )
    initialization_samples = manifest.get("initialization_samples", {})
    if not initialization_samples:
        raise ValueError("Pose candidate manifest has no initialization samples")
    overrides: dict[str, dict[str, Any]] = {}
    for group_id, sample in initialization_samples.items():
        unit_samples = sample.get("unit_samples")
        if not isinstance(unit_samples, dict):
            raise ValueError(
                f"Pose candidate group {group_id!r} lacks unit samples"
            )
        so3_unit = unit_samples.get("so3")
        if so3_unit is None:
            raise ValueError(
                f"Pose candidate group {group_id!r} lacks SO(3) unit samples"
            )
        overrides[group_id] = {
            "radius_unit": unit_samples.get("radius"),
            "axial_offset_unit": unit_samples.get("axial_offset"),
            "so3_unit": so3_unit,
        }
    expected_structure_sha = manifest.get("outputs", {}).get(
        "structure", {}
    ).get("sha256")
    if not expected_structure_sha:
        raise ValueError("Pose candidate manifest lacks structure SHA256")
    effective_seed = candidate_config.get("effective_random_seed")
    return overrides, effective_seed, str(expected_structure_sha)


def _fragment_selector(
    mapping: dict[str, Any],
    fragment_instance_id: str,
) -> str:
    """Return an RFD3 label-chain residue selector for one fragment copy.

    AtomWorks indexes mmCIF selections with ``label_seq_id``.  Original PDB
    residue numbers remain available under the mapping's ``source`` records
    for provenance, but using those author numbers in an RFD3 contig makes the
    motif unresolvable after CIF parsing.
    """

    records = [
        record
        for record in mapping["atom_mappings"]
        if record["instance"]["fragment_instance_id"]
        == fragment_instance_id
    ]
    if not records:
        raise ValueError(
            f"No atom mappings found for fragment {fragment_instance_id!r}"
        )

    compiled_chains = {
        record["compiled"]["chain_id"] for record in records
    }
    if len(compiled_chains) != 1:
        raise ValueError(
            f"Fragment {fragment_instance_id!r} spans multiple output chains"
        )
    chain_id = next(iter(compiled_chains))
    residue_numbers = sorted(
        {
            int(record["compiled"]["label_seq_id"])
            for record in records
        }
    )
    expected = list(range(residue_numbers[0], residue_numbers[-1] + 1))
    if residue_numbers != expected:
        raise ValueError(
            f"RFD3 indexed motifs must be residue-contiguous; "
            f"{fragment_instance_id!r} is not"
        )
    return f"{chain_id}{residue_numbers[0]}-{residue_numbers[-1]}"


def _rfd3_atom_selection(fixed_atoms: str | list[str] | None) -> str:
    if fixed_atoms is None:
        return "ALL"
    if isinstance(fixed_atoms, list):
        if not fixed_atoms:
            raise ValueError("fixed_atoms lists cannot be empty")
        return ",".join(fixed_atoms)
    aliases = {
        "all": "ALL",
        "backbone": "BKBN",
        "bkbn": "BKBN",
        "tip": "TIP",
    }
    return aliases.get(fixed_atoms.lower(), fixed_atoms)


def _native_symmetry_id_and_multiplicity(transform_set: Any) -> tuple[str, int]:
    if transform_set.type == SymmetryType.CYCLIC:
        symmetry_id = f"C{transform_set.order}"
        multiplicity = transform_set.order
    elif transform_set.type == SymmetryType.DIHEDRAL:
        symmetry_id = f"D{transform_set.order}"
        multiplicity = 2 * transform_set.order
    else:
        raise NotImplementedError(
            f"Native RFD3 symmetry does not support "
            f"{transform_set.type.value!r}"
        )
    if multiplicity > 10:
        raise ValueError(
            f"Native RFD3 symmetric-motif inference currently allows at "
            f"most 10 transforms, but {symmetry_id} requires {multiplicity}"
        )
    return symmetry_id, multiplicity


def _preflight_native_transform_registry(
    transform_set: Any,
    expected_multiplicity: int,
) -> list[str]:
    """Reject incomplete, duplicated, improper, or non-closed Cn/Dn sets."""

    registry = build_transform_registry(transform_set)
    if registry.order != expected_multiplicity:
        raise ValueError(
            f"Transform registry {registry.group_name} contains "
            f"{registry.order} transforms; expected {expected_multiplicity}"
        )
    validate_group_closure(registry)
    matrices = [registry.transform(item) for item in registry.transform_ids]
    for index, matrix in enumerate(matrices):
        if not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6):
            raise ValueError(
                f"Transform {registry.transform_ids[index]} is not a "
                "proper rotation"
            )
        for previous in matrices[:index]:
            if np.allclose(matrix, previous, atol=1e-6):
                raise ValueError(
                    f"Transform registry {registry.group_name} contains "
                    "duplicate frames"
                )
    return list(registry.transform_ids)


def compile_rfd3_input(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    base_directory: str | Path = ".",
    example_id: str = "lhd101_c3_interface_seed",
    is_non_loopy: bool = True,
    pose_seed: int | None = None,
    pose_candidate_manifest: str | Path | None = None,
) -> RFD3AdapterOutputs:
    """Emit a native-symmetry RFD3 task from one orbit of scaffold links.

    RFD3 constructs one asymmetric unit from the contig and then applies its
    symmetry sampler.  Consequently, only the copy-zero scaffold path belongs
    in the contig; the pre-symmetrized CIF supplies the complete motif geometry
    used by RFD3 to infer the symmetry frames.
    """

    config = Path(config_path).resolve()
    base = Path(base_directory).resolve()
    output = Path(output_directory)
    candidate_manifest_path: Path | None = None
    sample_overrides: dict[str, dict[str, Any]] | None = None
    candidate_structure_sha: str | None = None
    if pose_candidate_manifest is not None:
        if pose_seed is not None:
            raise ValueError(
                "pose_seed and pose_candidate_manifest are mutually exclusive"
            )
        candidate_manifest_path = Path(pose_candidate_manifest).resolve()
        (
            sample_overrides,
            pose_seed,
            candidate_structure_sha,
        ) = _candidate_sample_overrides(
            candidate_manifest_path,
            config_path=config,
        )
    standalone = compile_standalone(
        config,
        output,
        base_directory=base,
        random_seed=pose_seed,
        sample_overrides=sample_overrides,
    )
    rebuilt_structure_sha = _sha256(standalone.structure_path)
    if (
        candidate_structure_sha is not None
        and rebuilt_structure_sha != candidate_structure_sha
    ):
        raise ValueError(
            "Rebuilt RFD3 adapter structure does not exactly reproduce the "
            f"pose candidate: {rebuilt_structure_sha} != "
            f"{candidate_structure_sha}"
        )
    spec = load_interface_seed_config(config)
    instances = expand_symmetry_instances(
        spec,
        master_transforms=None,
    )
    mapping = _load_json(standalone.mapping_path)

    orbit_links = [
        link
        for link in instances.scaffold_links.values()
        if link.copy_index == 0
    ]
    if len(orbit_links) != 1:
        raise NotImplementedError(
            "The static RFD3 adapter currently requires exactly one "
            "copy-zero scaffold relation"
        )
    link = orbit_links[0]
    if link.orbit_id is None:
        raise ValueError("Native symmetry requires an orbit-expanded link")

    orbit = spec.symmetry.orbits[link.orbit_id]
    transform_set = spec.symmetry.transform_sets[orbit.transform_set]
    symmetry_id, symmetry_multiplicity = (
        _native_symmetry_id_and_multiplicity(transform_set)
    )
    registry_transform_order = _preflight_native_transform_registry(
        transform_set,
        symmetry_multiplicity,
    )

    from_selector = _fragment_selector(
        mapping, link.from_fragment_instance_id
    )
    to_selector = _fragment_selector(mapping, link.to_fragment_instance_id)
    if link.chain_break:
        contig = f"{from_selector},/0,{to_selector}"
        scaffold_mode = "independent_chains"
        asu_chain_count = 2
    else:
        linker = f"{link.minimum_length}-{link.maximum_length}"
        contig = f"{from_selector},{linker},{to_selector}"
        scaffold_mode = "continuous_linker"
        asu_chain_count = 1

    from_fragment = instances.fragments[link.from_fragment_instance_id]
    to_fragment = instances.fragments[link.to_fragment_instance_id]
    fixed_atoms = {
        from_selector: _rfd3_atom_selection(
            spec.fragments[from_fragment.source_id].fixed_atoms
        ),
        to_selector: _rfd3_atom_selection(
            spec.fragments[to_fragment.source_id].fixed_atoms
        ),
    }
    payload = {
        example_id: {
            "dialect": 2,
            "input": standalone.structure_path.name,
            "contig": contig,
            "select_fixed_atoms": fixed_atoms,
            "redesign_motif_sidechains": False,
            "is_non_loopy": is_non_loopy,
            "symmetry": {
                "id": symmetry_id,
                "is_symmetric_motif": True,
            },
            "extra": {
                "compiler": "rfd3_mosaic.static_adapter",
                "pose_seed": (
                    spec.random_seed if pose_seed is None else pose_seed
                ),
                "pose_source": (
                    "candidate_manifest"
                    if candidate_manifest_path is not None
                    else "random_seed"
                ),
                "pose_candidate_manifest": (
                    str(candidate_manifest_path)
                    if candidate_manifest_path is not None
                    else None
                ),
                "pose_candidate_structure_sha256": candidate_structure_sha,
                "adapter_structure_sha256": rebuilt_structure_sha,
                "interface_seed_config": str(config),
                "asu_scaffold_link_instance": link.id,
                "asu_source_copy_index": link.copy_index,
                "asu_target_copy_index": link.target_copy_index,
                "symmetry_multiplicity": symmetry_multiplicity,
                "asu_chain_count": asu_chain_count,
                "scaffold_mode": scaffold_mode,
                "registry_preflight": "passed",
                "registry_transform_order": registry_transform_order,
                "mosaic_transform_order": list(
                    dict.fromkeys(
                        group.transform_id
                        for group in instances.motion_groups.values()
                        if group.orbit_id == link.orbit_id
                    )
                ),
            },
        }
    }
    input_path = output / "rfd3_input.json"
    input_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RFD3AdapterOutputs(
        input_path=input_path,
        structure_path=standalone.structure_path,
        mapping_path=standalone.mapping_path,
        manifest_path=standalone.manifest_path,
        example_id=example_id,
        contig=contig,
    )
