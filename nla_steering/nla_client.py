"""
Thin wrapper around the nla-inference NLAClient that handles SGLang server
lifecycle and provides a clean interface for verbalization and reconstruction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLAMeta:
    """Parsed nla_meta.yaml sidecar for a checkpoint."""

    model_name: str
    d_model: int
    layer_idx: int
    injection_scale: float
    embed_scale: Optional[float]
    template_prompt: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "NLAMeta":
        path = Path(checkpoint_dir) / "nla_meta.yaml"
        if not path.exists():
            raise FileNotFoundError(f"nla_meta.yaml not found in {checkpoint_dir}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


class NLAVerbalizer:
    """
    Wraps the NLAClient inference-only package to verbalize activation vectors.

    Expects an SGLang server to already be running for the AV model.
    See scripts/launch_sglang.sh for server setup.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        sglang_url: str = "http://localhost:30000",
        device: str = "cpu",
    ) -> None:
        try:
            from nla_inference import NLAClient  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "nla_inference not found. Clone https://github.com/kitft/nla-inference "
                "and install it: pip install -e ."
            ) from e

        self.meta = NLAMeta.from_checkpoint(checkpoint_dir)
        self._client = NLAClient(
            checkpoint_dir=str(checkpoint_dir),
            sglang_url=sglang_url,
            device=device,
        )
        logger.info(
            "NLAVerbalizer ready — model=%s layer=%d d_model=%d",
            self.meta.model_name,
            self.meta.layer_idx,
            self.meta.d_model,
        )

    def verbalize(
        self,
        activation: np.ndarray | torch.Tensor,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> str:
        """Convert a single activation vector to a natural language description."""
        if isinstance(activation, torch.Tensor):
            activation = activation.float().cpu().numpy()
        activation = activation.squeeze()
        if activation.shape != (self.meta.d_model,):
            raise ValueError(
                f"Expected activation of shape ({self.meta.d_model},), got {activation.shape}"
            )
        return self._client.generate(
            activation,
            extract_explanation=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def verbalize_batch(
        self,
        activations: np.ndarray | torch.Tensor,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Verbalize a batch of activations. activations shape: (N, d_model)."""
        if isinstance(activations, torch.Tensor):
            activations = activations.float().cpu().numpy()
        return self._client.generate_batch(
            activations,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def health_check(self) -> bool:
        """Return True if the SGLang server is reachable."""
        try:
            resp = httpx.get(self._client.sglang_url + "/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.RequestError:
            return False
