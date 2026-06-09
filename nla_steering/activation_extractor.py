"""
Forward-hook based activation extractor for HuggingFace decoder models.

Captures residual stream activations at specified layer indices, optionally
before or after the MLP/attention sublayers, for use with NLA verbalizers.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class ActivationStore:
    """Thread-unsafe store for activations captured by forward hooks."""

    def __init__(self) -> None:
        self._store: dict[int, torch.Tensor] = {}

    def record(self, layer_idx: int, activation: torch.Tensor) -> None:
        self._store[layer_idx] = activation.detach().float()

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._store.get(layer_idx)

    def all(self) -> dict[int, torch.Tensor]:
        return dict(self._store)

    def clear(self) -> None:
        self._store.clear()


def _get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Resolve the decoder layer list for standard HF model families."""
    # Qwen2, Llama, Mistral, Gemma all use model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 / Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        f"Cannot resolve decoder layers for {type(model).__name__}. "
        "Set model.repeng_layers or update _get_decoder_layers."
    )


def _last_non_padding_position(
    hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Return the last non-padded token's hidden state for each batch element.
    hidden: (batch, seq, d_model)
    returns: (batch, d_model)
    """
    if attention_mask is None:
        return hidden[:, -1, :]
    lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    idx = lengths.clamp(min=0).long()
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx, :]


@contextmanager
def extraction_hooks(
    model: PreTrainedModel,
    layer_indices: list[int],
    store: ActivationStore,
    position: str = "last",
    attention_mask: Optional[torch.Tensor] = None,
) -> Generator[ActivationStore, None, None]:
    """
    Context manager that installs forward hooks to capture activations.

    Args:
        model: HuggingFace causal LM.
        layer_indices: Which decoder layers to hook (0-indexed).
        store: ActivationStore to write captured activations into.
        position: "last" (last non-padding token) or "all" (full sequence tensor).
        attention_mask: Used when position="last" to find the real last token.

    Yields:
        The same ActivationStore (populated after forward pass inside context).
    """
    decoder_layers = _get_decoder_layers(model)
    n_layers = len(decoder_layers)
    handles: list[torch.utils.hooks.RemovableHook] = []

    store.clear()

    def _make_hook(layer_idx: int) -> callable:
        def _hook(module: nn.Module, inp: tuple, out: tuple | torch.Tensor) -> None:
            # Most decoder blocks return (hidden_state, ...) tuples
            hidden = out[0] if isinstance(out, tuple) else out
            if position == "last":
                captured = _last_non_padding_position(hidden, attention_mask)
            else:
                captured = hidden.float()
            store.record(layer_idx, captured)

        return _hook

    for idx in layer_indices:
        if idx < 0:
            idx = n_layers + idx
        if not (0 <= idx < n_layers):
            raise IndexError(f"Layer index {idx} out of range [0, {n_layers})")
        handle = decoder_layers[idx].register_forward_hook(_make_hook(idx))
        handles.append(handle)
        logger.debug("Installed hook at layer %d", idx)

    try:
        yield store
    finally:
        for h in handles:
            h.remove()
        logger.debug("Removed %d hooks", len(handles))


def extract_activations(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    layer_indices: list[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[int, torch.Tensor]:
    """
    Run a forward pass and return residual stream activations at given layers.

    Returns:
        dict mapping layer_idx → tensor of shape (batch, d_model).
    """
    store = ActivationStore()
    with torch.no_grad():
        with extraction_hooks(
            model,
            layer_indices,
            store,
            position="last",
            attention_mask=attention_mask,
        ):
            model(input_ids=input_ids, attention_mask=attention_mask)
    return store.all()
