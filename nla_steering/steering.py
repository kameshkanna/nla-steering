"""
Steering vector generation (CAA / PCA-diff) and injection utilities.

Provides:
  - SteeringVector: a layer-indexed mapping of direction vectors
  - generate_caa_vector: build a steering direction from contrast pairs
  - SteeringHook: inject a vector at a specific layer during a forward pass
  - extract_post_steering_activation: single call that steers + captures AV input
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from nla_steering.activation_extractor import (
    ActivationStore,
    _get_decoder_layers,
    _last_non_padding_position,
    extraction_hooks,
)

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """
    Layer-indexed steering directions produced by CAA / PCA-diff.

    directions: dict[layer_idx → np.ndarray of shape (d_model,)]
    """

    model_name: str
    directions: dict[int, np.ndarray] = field(default_factory=dict)

    def to_tensor(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.directions[layer_idx]).float().to(device)

    def save(self, path: str) -> None:
        np.savez(path, **{str(k): v for k, v in self.directions.items()})

    @classmethod
    def load(cls, path: str, model_name: str) -> "SteeringVector":
        data = np.load(path)
        return cls(
            model_name=model_name,
            directions={int(k): v for k, v in data.items()},
        )


def generate_caa_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    positive_texts: list[str],
    negative_texts: list[str],
    layer_indices: list[int],
    batch_size: int = 8,
    method: str = "mean_diff",
) -> SteeringVector:
    """
    Compute a CAA steering vector from contrastive text pairs.

    Args:
        model: HuggingFace causal LM (eval mode expected).
        tokenizer: Corresponding tokenizer.
        positive_texts: Texts representing the desired direction.
        negative_texts: Texts representing the undesired direction.
        layer_indices: Layers at which to compute directions.
        batch_size: Inference batch size.
        method: "mean_diff" (default) or "pca_diff".

    Returns:
        SteeringVector with one direction per requested layer.
    """
    assert len(positive_texts) == len(negative_texts), (
        "positive_texts and negative_texts must have equal length"
    )

    model.eval()
    pos_accum: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}
    neg_accum: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}

    def _run_batch(texts: list[str], accum: dict[int, list[np.ndarray]]) -> None:
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(next(model.parameters()).device)
        store = ActivationStore()
        with torch.no_grad():
            with extraction_hooks(
                model,
                layer_indices,
                store,
                position="last",
                attention_mask=enc.get("attention_mask"),
            ):
                model(**enc)
        for l in layer_indices:
            act = store.get(l)
            if act is not None:
                accum[l].append(act.cpu().numpy())

    for i in range(0, len(positive_texts), batch_size):
        _run_batch(positive_texts[i : i + batch_size], pos_accum)
        _run_batch(negative_texts[i : i + batch_size], neg_accum)
        logger.debug("Processed pairs %d-%d", i, i + batch_size)

    directions: dict[int, np.ndarray] = {}

    for l in layer_indices:
        pos_mat = np.concatenate(pos_accum[l], axis=0)  # (N, d_model)
        neg_mat = np.concatenate(neg_accum[l], axis=0)

        if method == "mean_diff":
            directions[l] = (pos_mat.mean(0) - neg_mat.mean(0)).astype(np.float32)

        elif method == "pca_diff":
            diff = pos_mat - neg_mat  # (N, d_model)
            # PCA via SVD — top singular vector captures dominant direction
            _, _, vh = np.linalg.svd(diff - diff.mean(0), full_matrices=False)
            direction = vh[0].astype(np.float32)
            # Sign: align with mean difference
            mean_diff = pos_mat.mean(0) - neg_mat.mean(0)
            if np.dot(direction, mean_diff) < 0:
                direction = -direction
            directions[l] = direction

        else:
            raise ValueError(f"Unknown method: {method}")

        # Unit-normalize
        norm = np.linalg.norm(directions[l])
        if norm > 1e-8:
            directions[l] /= norm

    model_name = type(model).__name__
    logger.info("Generated %s steering vector at %d layers", method, len(directions))
    return SteeringVector(model_name=model_name, directions=directions)


@contextmanager
def steering_hook(
    model: PreTrainedModel,
    layer_idx: int,
    direction: torch.Tensor,
    coefficient: float,
) -> Generator[None, None, None]:
    """
    Context manager that adds `coefficient * direction` to the residual stream
    output of `layer_idx` during a forward pass.
    """
    decoder_layers = _get_decoder_layers(model)
    n_layers = len(decoder_layers)
    real_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx

    def _hook(module: nn.Module, inp: tuple, out: tuple | torch.Tensor) -> tuple | torch.Tensor:
        hidden = out[0] if isinstance(out, tuple) else out
        hidden = hidden + coefficient * direction.to(hidden.device, hidden.dtype)
        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden

    handle = decoder_layers[real_idx].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def extract_post_steering_activation(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    steer_layer: int,
    capture_layer: int,
    direction: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """
    Apply a steering vector at `steer_layer`, then capture the residual stream
    at `capture_layer` (typically the same layer, for NLA verbalization).

    Returns:
        Tensor of shape (batch, d_model) — the post-steering activation.
    """
    store = ActivationStore()
    with torch.no_grad():
        with steering_hook(model, steer_layer, direction, coefficient):
            with extraction_hooks(
                model,
                [capture_layer],
                store,
                position="last",
                attention_mask=attention_mask,
            ):
                model(input_ids=input_ids, attention_mask=attention_mask)
    result = store.get(capture_layer)
    if result is None:
        raise RuntimeError(f"No activation captured at layer {capture_layer}")
    return result
