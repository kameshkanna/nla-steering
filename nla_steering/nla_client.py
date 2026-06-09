"""
NLA verbalizer that calls the SGLang server directly.

Bypasses NLAClient.__init__ tokenizer validation (which tries to load a
tokenizer from the checkpoint dir — but NLA checkpoints only have weights,
not a tokenizer). Instead we accept the base model's tokenizer externally
and build input_embeds ourselves following the same protocol as nla_inference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import orjson
import torch
import yaml
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLAMeta:
    """Parsed nla_meta.yaml sidecar for a checkpoint (schema_version=2)."""

    role: str
    d_model: int
    layer_idx: int
    injection_scale: float
    injection_char: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int
    av_prompt_template: str
    ar_prompt_template: str

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "NLAMeta":
        path = Path(checkpoint_dir) / "nla_meta.yaml"
        if not path.exists():
            raise FileNotFoundError(f"nla_meta.yaml not found in {checkpoint_dir}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            role=raw["role"],
            d_model=raw["d_model"],
            layer_idx=raw["extraction_layer_index"],
            injection_scale=raw["extraction"]["injection_scale"],
            injection_char=raw["tokens"]["injection_char"],
            injection_token_id=raw["tokens"]["injection_token_id"],
            left_neighbor_id=raw["tokens"]["injection_left_neighbor_id"],
            right_neighbor_id=raw["tokens"]["injection_right_neighbor_id"],
            av_prompt_template=raw["prompt_templates"]["av"],
            ar_prompt_template=raw["prompt_templates"]["ar"],
        )


class NLAVerbalizer:
    """
    Verbalizes residual stream activations by calling a running SGLang AV server.

    Builds input_embeds directly: tokenizes the AV prompt, finds the injection
    position, replaces its embedding with the scaled activation vector, and
    sends to SGLang /generate. The base model's tokenizer must be passed in
    because NLA checkpoints don't ship with their own tokenizer files.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        model: torch.nn.Module,
        sglang_url: str = "http://localhost:30000",
    ) -> None:
        self.meta = NLAMeta.from_checkpoint(checkpoint_dir)
        self._tokenizer = tokenizer
        self._sglang_url = sglang_url.rstrip("/")

        # Load embedding table from the base model (CPU, float32)
        embed_module = model.get_input_embeddings()
        self._embed_weight: torch.Tensor = embed_module.weight.detach().float().cpu()

        logger.info(
            "NLAVerbalizer ready — role=%s layer=%d d_model=%d",
            self.meta.role,
            self.meta.layer_idx,
            self.meta.d_model,
        )

    def _build_embeds(self, activation: np.ndarray) -> np.ndarray:
        """
        Build the input_embeds array for one activation vector.
        Follows nla_inference._build_embeds protocol exactly.
        """
        prompt_content = self.meta.av_prompt_template.format(
            injection_char=self.meta.injection_char
        )
        ids: list[int] = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            tokenize=True,
            add_generation_prompt=True,
        )

        # Find injection position (with neighbor validation)
        inject_pos: Optional[int] = None
        for i in range(1, len(ids) - 1):
            if (
                ids[i] == self.meta.injection_token_id
                and ids[i - 1] == self.meta.left_neighbor_id
                and ids[i + 1] == self.meta.right_neighbor_id
            ):
                inject_pos = i
                break

        if inject_pos is None:
            # Fallback: find by token id alone
            for i, tok in enumerate(ids):
                if tok == self.meta.injection_token_id:
                    inject_pos = i
                    break

        if inject_pos is None:
            raise RuntimeError(
                f"Injection token {self.meta.injection_token_id} not found in "
                f"tokenized prompt. ids={ids[:20]}..."
            )

        # Build embedding matrix
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        embeds = self._embed_weight[ids_tensor].clone()  # (T, d_model)

        # Normalize and inject activation
        act_tensor = torch.from_numpy(activation.astype(np.float32))
        norm = act_tensor.norm()
        if norm > 1e-8:
            act_tensor = act_tensor / norm
        act_tensor = act_tensor * self.meta.injection_scale
        embeds[inject_pos] = act_tensor

        return embeds.numpy().astype(np.float32)

    def _sglang_generate(
        self,
        embeds: np.ndarray,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        payload = {
            "input_embeds": embeds.tolist(),
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        }
        resp = httpx.post(
            self._sglang_url + "/generate",
            content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY),
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        resp.raise_for_status()
        text: str = resp.json()["text"]
        # Extract content inside <explanation>...</explanation> if present
        m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

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
                f"Expected shape ({self.meta.d_model},), got {activation.shape}"
            )
        embeds = self._build_embeds(activation)
        return self._sglang_generate(embeds, temperature, max_new_tokens)

    def verbalize_batch(
        self,
        activations: np.ndarray | torch.Tensor,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Verbalize a batch of activations. activations shape: (N, d_model)."""
        if isinstance(activations, torch.Tensor):
            activations = activations.float().cpu().numpy()
        return [
            self.verbalize(activations[i], temperature, max_new_tokens)
            for i in range(len(activations))
        ]

    def health_check(self) -> bool:
        """Return True if the SGLang server is reachable."""
        try:
            resp = httpx.get(self._sglang_url + "/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.RequestError:
            return False
