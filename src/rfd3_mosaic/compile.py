from pathlib import Path

import yaml

from rfd3_mosaic.schema import InterfaceSeedSpec


def load_interface_seed_config(
    config_path: str | Path,
) -> InterfaceSeedSpec:
    """Load and validate an Interface-Seed configuration."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Interface-Seed config does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Interface-Seed config must contain a YAML mapping"
        )

    payload = raw_config.get("interface_seed", raw_config)

    if not isinstance(payload, dict):
        raise ValueError(
            "The interface_seed field must contain a YAML mapping"
        )

    return InterfaceSeedSpec.model_validate(payload)