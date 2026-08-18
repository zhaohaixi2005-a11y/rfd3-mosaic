"""Ordinary-user intent schema for input-driven cage design."""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AliasChoices, Field, model_validator

from rfd3_mosaic.schema.design import (
    RequestedLength,
    UserDesignPreferences,
    UserInterfaceUsageSpec,
    UserOutputSpec,
    UserResourceSpec,
)
from rfd3_mosaic.schema.specs import CopyRelationSpec, Identifier, StrictModel

SimpleSymmetryName = Annotated[
    str,
    Field(pattern=r"^(?:C[1-9][0-9]*|D[2-9][0-9]*|T|O|I)$"),
]


class SimpleIntegerRange(StrictModel):
    minimum: Annotated[int, Field(ge=1)]
    maximum: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_order(self) -> "SimpleIntegerRange":
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class SimpleFloatRange(StrictModel):
    minimum: Annotated[float, Field(gt=0.0)]
    maximum: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def validate_order(self) -> "SimpleFloatRange":
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class SimpleCageGoalSpec(StrictModel):
    """Scientific intent retained while Mosaic chooses the geometry."""

    architecture: Literal["auto", "ring", "cage"] = "auto"
    composition: Literal["auto", "homomer", "heteromer"] = "auto"
    symmetry: Literal["auto"] | tuple[SimpleSymmetryName, ...] = "auto"
    subunits: SimpleIntegerRange | None = None
    diameter_angstrom: SimpleFloatRange | None = None
    cavity_diameter_angstrom: SimpleFloatRange | None = None

    @model_validator(mode="after")
    def validate_symmetry_choices(self) -> "SimpleCageGoalSpec":
        if isinstance(self.symmetry, tuple):
            if not self.symmetry:
                raise ValueError("goal.symmetry choices cannot be empty")
            if len(self.symmetry) != len(set(self.symmetry)):
                raise ValueError("goal.symmetry choices must be unique")
        return self


class SimpleInterfaceSeedSpec(StrictModel):
    """One interface detected in or selected from the input structure."""

    source: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("source", "structure"),
    )
    participants: tuple[str, ...] = Field(
        min_length=2,
        validation_alias=AliasChoices("participants", "between_chains"),
    )
    selectors: dict[str, Annotated[str, Field(min_length=1)]]
    use: UserInterfaceUsageSpec = Field(
        default_factory=UserInterfaceUsageSpec
    )
    # A supplied natural interface is one atomic rigid hyperedge.  Ordinary
    # mode may move that complete seed while solving the cage, but it never
    # deforms the geometry between its participants.  Component-level
    # bounded mobility remains available through the expert public graph.
    geometry: Literal["preserve_exact"] = "preserve_exact"

    @model_validator(mode="after")
    def validate_sides(self) -> "SimpleInterfaceSeedSpec":
        if len(self.participants) != len(set(self.participants)):
            raise ValueError("interface seed participants must be unique")
        if set(self.selectors) != set(self.participants):
            raise ValueError(
                "interface seed selectors must contain exactly its "
                "participants"
            )
        return self


class SimpleGenerationIntentSpec(StrictModel):
    length: RequestedLength


class SimplePolymerEndpointSpec(StrictModel):
    """One supplied interface participant used as a polymer-path node."""

    interface: Identifier
    participant: Annotated[str, Field(min_length=1)]

    @model_validator(mode="before")
    @classmethod
    def parse_compact_endpoint(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        interface_id, separator, participant = value.rpartition(".")
        if not separator or not interface_id or not participant:
            raise ValueError(
                "Compact polymer endpoints must use interface.participant"
            )
        return {
            "interface": interface_id,
            "participant": participant,
        }


class SimplePolymerConnectionSpec(StrictModel):
    """A user-authoritative C-to-N connection between supplied seeds."""

    id: Identifier | None = None
    from_endpoint: SimplePolymerEndpointSpec = Field(
        validation_alias=AliasChoices("from", "from_endpoint"),
        serialization_alias="from",
    )
    to_endpoint: SimplePolymerEndpointSpec = Field(
        validation_alias=AliasChoices("to", "to_endpoint"),
        serialization_alias="to",
    )
    copy_relation: CopyRelationSpec | None = None


class SimpleInspectionSpec(StrictModel):
    """Frozen input-analysis parameters used to create this intent."""

    contact_cutoff: Annotated[float, Field(gt=0.0)] = 4.5
    minimum_atom_contacts: Annotated[int, Field(ge=1)] = 4
    minimum_contact_residues_per_side: Annotated[int, Field(ge=1)] = 2


class SimpleCageIntentSpec(StrictModel):
    """Short input-driven request before architecture resolution."""

    schema_version: Literal[1] = 1
    kind: Literal["simple_cage_intent"] = "simple_cage_intent"
    name: Identifier
    input: Path
    seed_layout: Literal["auto", "preserve_input", "solve"] = "auto"
    goal: SimpleCageGoalSpec = Field(default_factory=SimpleCageGoalSpec)
    interface_seeds: dict[Identifier, SimpleInterfaceSeedSpec]
    polymer_connections: tuple[SimplePolymerConnectionSpec, ...] = ()
    generation: SimpleGenerationIntentSpec
    preferences: UserDesignPreferences = Field(
        default_factory=UserDesignPreferences
    )
    inspection: SimpleInspectionSpec = Field(
        default_factory=SimpleInspectionSpec
    )
    resources: UserResourceSpec = Field(default_factory=UserResourceSpec)
    output: UserOutputSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_input_from_seed_library(
        cls,
        value: Any,
    ) -> Any:
        """Let an ordinary seed library omit the redundant top-level input.

        The canonical execution path still consumes one structure.  The
        resolver materializes that structure from all independently supplied
        seed files.  Using the first seed source here only gives the typed
        intent a real path before that materialization step; it never implies
        that the other seeds share its coordinate frame.
        """

        if not isinstance(value, dict) or value.get("input") is not None:
            return value
        seeds = value.get("interface_seeds")
        if not isinstance(seeds, dict):
            return value
        for seed in seeds.values():
            if not isinstance(seed, dict):
                continue
            source = seed.get("source", seed.get("structure"))
            if source is not None:
                updated = dict(value)
                updated["input"] = source
                return updated
        return value

    @model_validator(mode="after")
    def require_interfaces(self) -> "SimpleCageIntentSpec":
        if not self.interface_seeds:
            raise ValueError(
                "simple cage intent requires at least one interface seed"
            )
        if self.seed_layout == "solve" and len(self.interface_seeds) < 2:
            raise ValueError(
                "seed_layout=solve requires at least two supplied interface "
                "seeds whose relative pose can be optimized"
            )
        missing_sources = [
            seed_id
            for seed_id, seed in self.interface_seeds.items()
            if seed.source is None and self.input is None
        ]
        if missing_sources:
            raise ValueError(
                "interface seeds require either their own source/structure "
                "or one top-level input: " + ", ".join(missing_sources)
            )
        connection_ids = [
            connection.id
            for connection in self.polymer_connections
            if connection.id is not None
        ]
        if len(connection_ids) != len(set(connection_ids)):
            raise ValueError("polymer connection IDs must be unique")
        outgoing: set[tuple[str, str]] = set()
        incoming: set[tuple[str, str]] = set()
        used: set[tuple[str, str]] = set()
        for connection in self.polymer_connections:
            source = connection.from_endpoint
            target = connection.to_endpoint
            for endpoint in (source, target):
                seed = self.interface_seeds.get(endpoint.interface)
                if seed is None:
                    raise ValueError(
                        "polymer connection references unknown interface "
                        f"{endpoint.interface!r}"
                    )
                if endpoint.participant not in seed.participants:
                    raise ValueError(
                        "polymer connection references unknown participant "
                        f"{endpoint.interface}.{endpoint.participant}"
                    )
            source_key = (source.interface, source.participant)
            target_key = (target.interface, target.participant)
            if source.interface == target.interface:
                raise ValueError(
                    "polymer connections cannot join participants of the "
                    "same supplied interface seed"
                )
            if source_key in outgoing:
                raise ValueError(
                    "a supplied participant can have at most one outgoing "
                    f"polymer connection: {source.interface}."
                    f"{source.participant}"
                )
            if target_key in incoming:
                raise ValueError(
                    "a supplied participant can have at most one incoming "
                    f"polymer connection: {target.interface}."
                    f"{target.participant}"
                )
            outgoing.add(source_key)
            incoming.add(target_key)
            used.update((source_key, target_key))
        if self.polymer_connections:
            expected = {
                (interface_id, participant)
                for interface_id, seed in self.interface_seeds.items()
                for participant in seed.participants
            }
            missing = sorted(expected - used)
            if missing:
                rendered = ", ".join(
                    f"{interface_id}.{participant}"
                    for interface_id, participant in missing
                )
                raise ValueError(
                    "user-declared polymer connections must assign every "
                    f"supplied interface participant; missing: {rendered}"
                )
        return self


def load_simple_cage_intent(path: str | Path) -> SimpleCageIntentSpec:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Simple cage intent does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Simple cage intent must contain a YAML mapping")
    intent = SimpleCageIntentSpec.model_validate(payload)
    structure = intent.input.expanduser()
    if not structure.is_absolute():
        structure = source.parent / structure
    structure = structure.resolve()
    if not structure.is_file():
        raise FileNotFoundError(f"Design input does not exist: {structure}")
    resolved_seeds = {}
    for seed_id, seed in intent.interface_seeds.items():
        seed_source = seed.source
        if seed_source is not None:
            seed_source = seed_source.expanduser()
            if not seed_source.is_absolute():
                seed_source = source.parent / seed_source
            seed_source = seed_source.resolve()
            if not seed_source.is_file():
                raise FileNotFoundError(
                    f"Interface seed {seed_id!r} source does not exist: "
                    f"{seed_source}"
                )
            seed = seed.model_copy(update={"source": seed_source})
        resolved_seeds[seed_id] = seed
    output = intent.output
    if output is not None:
        root = output.root.expanduser()
        if not root.is_absolute():
            root = source.parent / root
        output = output.model_copy(update={"root": root.resolve()})
    return intent.model_copy(
        update={
            "input": structure,
            "interface_seeds": resolved_seeds,
            "output": output,
        }
    )


__all__ = [
    "SimpleCageGoalSpec",
    "SimpleCageIntentSpec",
    "SimpleFloatRange",
    "SimpleGenerationIntentSpec",
    "SimpleInspectionSpec",
    "SimpleIntegerRange",
    "SimpleInterfaceSeedSpec",
    "SimplePolymerConnectionSpec",
    "SimplePolymerEndpointSpec",
    "load_simple_cage_intent",
]
