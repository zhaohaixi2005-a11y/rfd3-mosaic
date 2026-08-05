import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from rfd3.model.cfg_utils import (
    strip_f,
)
from rfd3.model.inference_sampler import ConditionalDiffusionSampler
from rfd3.model.layers.encoders import TokenInitializer
from torch import nn

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class RFD3(nn.Module):
    """
    Simplified model for generation
    This module level serves to wrap the diffusion module of AF3
    to be roughly equivalent to the AF3 model w/o trunk processing.

    Allows the same sampler to be used
    """

    def __init__(
        self,
        *,
        # Channel dimensions ('global' features)
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atompair: int,
        # Arguments for modules that will be instantiated
        token_initializer: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        **_: Any,
    ):
        super().__init__()
        # Check for chunked P_LL mode via environment variable
        use_chunked_pll = os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"
        ranked_logger.info(f"RFD3 initialized with chunked_pll={use_chunked_pll}")

        # Simple constant-feature initializer.
        # `**token_initializer` / `**inference_sampler` below: omegaconf's DictConfig
        # supports the mapping protocol at runtime but its stubs don't satisfy
        # SupportsKeysAndGetItem, so mypy rejects `**(DictConfig | dict)`. Hydra sub-configs.
        self.token_initializer = TokenInitializer(  # type: ignore[arg-type]
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            use_chunked_pll=use_chunked_pll,
            **token_initializer,
        )

        # Diffusion module instantiated to allow for config scripting
        self.diffusion_module = hydra.utils.instantiate(
            diffusion_module, c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z
        )

        self.use_classifier_free_guidance = (
            inference_sampler["use_classifier_free_guidance"]
            and inference_sampler["cfg_scale"] != 1.0
        )
        self.cfg_features = inference_sampler.pop("cfg_features", [])

        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = ConditionalDiffusionSampler(**inference_sampler)  # type: ignore[arg-type]

    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor | None = None,
        n_cycle: int | None = None,
        **_: Any,
    ) -> dict:
        full_f = input["f"]
        network_f = full_f
        local_symmetry_context = None
        if not self.training:
            # The local backend must crop before TokenInitializer; cropping
            # only X_noisy_L inside the sampler would retain the full pair
            # representation and misalign atom/token indices.
            assert coord_atom_lvl_to_be_noised is not None
            prepare_local_view = getattr(
                self.inference_sampler.sampler,
                "prepare_local_network_view",
                None,
            )
            if prepare_local_view is not None:
                local_symmetry_context = prepare_local_view(
                    full_f,
                    coord_atom_lvl_to_be_noised,
                )
                if local_symmetry_context is not None:
                    if not self.token_initializer.use_chunked_pll:
                        raise ValueError(
                            "local_neighbourhood requires low_memory_mode=True"
                        )
                    network_f = (
                        local_symmetry_context.feature_view.features
                    )

        initializer_outputs = self.token_initializer(network_f)

        if self.training:
            # Single denoising step
            return self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                n_recycle=n_cycle,
                **initializer_outputs,
            )  # [D, L, 3]
        else:
            # Inference always provides the coordinates to be noised.
            assert coord_atom_lvl_to_be_noised is not None
            if self.use_classifier_free_guidance:
                f_ref = strip_f(network_f, self.cfg_features)
                ref_initializer_outputs = self.token_initializer(f_ref)
            else:
                f_ref = None
                ref_initializer_outputs = None

            local_kwargs = {}
            if local_symmetry_context is not None:
                local_kwargs = {
                    "network_f": network_f,
                    "local_symmetry_context": local_symmetry_context,
                }
            return self.inference_sampler.sample_diffusion_like_af3(
                f=full_f,
                f_ref=f_ref,  # for cfg
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=coord_atom_lvl_to_be_noised.shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                # Forwarded as **kwargs:
                initializer_outputs=initializer_outputs,
                ref_initializer_outputs=ref_initializer_outputs,  # for cfg
                **local_kwargs,
            )
