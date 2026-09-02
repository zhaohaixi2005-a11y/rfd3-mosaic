"""Resolve interface intent once for plans, runtime metadata and reports.

Supplied-interface preservation and generated-interface creation are
orthogonal contracts.  Keeping their derivation here prevents frontends,
capability reporting and runtime metadata from silently assigning different
meanings to the same public design.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from rfd3_mosaic.schema.design import (
    TerminalGeneration,
    UserDesignSpec,
    UserDesignTask,
)
from rfd3_mosaic.schema.specs import StrictModel


class InterfaceContractMode(str, Enum):
    NONE = "none"
    PRESERVE_SUPPLIED_ONLY = "preserve_supplied_only"
    CREATE_GENERATED_ONLY = "create_generated_only"
    PRESERVE_SUPPLIED_AND_CREATE_GENERATED = "preserve_supplied_and_create_generated"


GeneratedInterfaceSource = Literal[
    "task_terminal_inference",
    "graph_declared",
    "automatic_symmetric_generated",
]


class ResolvedInterfaceContract(StrictModel):
    """Backend-neutral statement of which interface geometry is protected."""

    mode: InterfaceContractMode
    preserves_supplied_geometry: bool
    creates_generated_interface: bool
    generated_interface_sources: tuple[GeneratedInterfaceSource, ...] = ()


def resolve_interface_contract(design: UserDesignSpec) -> ResolvedInterfaceContract:
    """Derive interface intent without inferring it from task names downstream."""

    preserves_supplied = bool(
        design.task == UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY
        or any(
            interface.relation.mode == "preserve_input"
            for interface in design.interfaces
        )
    )
    generated_sources: list[GeneratedInterfaceSource] = []
    if design.task == UserDesignTask.CREATE_SYMMETRIC_INTERFACE and any(
        isinstance(item, TerminalGeneration) for item in design.generation
    ):
        generated_sources.append("task_terminal_inference")
    if any(interface.relation.mode == "contact" for interface in design.interfaces):
        generated_sources.append("graph_declared")
    if design.sampling.scaffold_packing == "symmetric_generated":
        generated_sources.append("automatic_symmetric_generated")

    creates_generated = bool(generated_sources)
    if preserves_supplied and creates_generated:
        mode = InterfaceContractMode.PRESERVE_SUPPLIED_AND_CREATE_GENERATED
    elif preserves_supplied:
        mode = InterfaceContractMode.PRESERVE_SUPPLIED_ONLY
    elif creates_generated:
        mode = InterfaceContractMode.CREATE_GENERATED_ONLY
    else:
        mode = InterfaceContractMode.NONE
    return ResolvedInterfaceContract(
        mode=mode,
        preserves_supplied_geometry=preserves_supplied,
        creates_generated_interface=creates_generated,
        generated_interface_sources=tuple(generated_sources),
    )


__all__ = [
    "InterfaceContractMode",
    "ResolvedInterfaceContract",
    "resolve_interface_contract",
]
