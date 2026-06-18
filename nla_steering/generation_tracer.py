"""
Generation Tracer: verbalize the residual stream at every autoregressive step.

During steered generation we want to know:
  - Does the model's internal representation shift with the steering vector?
  - Does it shift back (self-correction) at some point?
  - How does magnitude affect the trajectory?

We implement this by hooking into the generate() loop via a custom
`LogitsProcessor` that fires on every forward step, captures the residual
stream at the target layer, and queues the activation for async verbalization
after generation completes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizerBase

from nla_steering.activation_extractor import (
    ActivationStore,
    _get_decoder_layers,
    _last_non_padding_position,
    extraction_hooks,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationStep:
    """Single step captured during autoregressive generation."""

    step: int
    token_id: int
    token_text: str
    activation: np.ndarray  # (d_model,)
    description: Optional[str] = None  # filled in after verbalization
    steering_norm: Optional[float] = None  # ||delta|| at this step if steered


class ActivationCapturingProcessor(LogitsProcessor):
    """
    LogitsProcessor that captures the residual stream at `capture_layer`
    on every forward step inside model.generate().

    HuggingFace calls LogitsProcessor.__call__(input_ids, scores) after each
    forward pass — at that point the model's last forward pass is complete and
    hooks have already fired. We install hooks externally and this processor
    reads from the shared ActivationStore.

    Args:
        store: Shared ActivationStore populated by extraction_hooks.
        capture_layer: Layer index to read from.
        steps: Output list — one GenerationStep appended per step.
    """

    def __init__(
        self,
        store: ActivationStore,
        capture_layer: int,
        steps: list[GenerationStep],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._store = store
        self._capture_layer = capture_layer
        self._steps = steps
        self._tokenizer = tokenizer
        self._step_count = 0

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        act = self._store.get(self._capture_layer)
        if act is not None:
            next_token_id = int(scores.argmax(dim=-1)[0].item())
            next_token_text = self._tokenizer.decode([next_token_id])
            self._steps.append(
                GenerationStep(
                    step=self._step_count,
                    token_id=next_token_id,
                    token_text=next_token_text,
                    activation=act[0].cpu().float().numpy(),
                )
            )
        self._step_count += 1
        # Return scores unchanged — we're only observing
        return scores


def trace_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    capture_layer: int,
    verbalizer,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    steering_hook_ctx=None,
    verbalize_every_n: int = 1,
    verbalize_max_tokens: int = 96,
) -> list[GenerationStep]:
    """
    Run autoregressive generation and capture + verbalize the residual stream
    at `capture_layer` on every step (or every `verbalize_every_n` steps).

    Args:
        model: Causal LM in eval mode.
        tokenizer: Matching tokenizer.
        input_ids: Prompt token ids, shape (1, prompt_len).
        capture_layer: Layer to read activations from.
        verbalizer: Verbalizer instance with verbalize_batch() — called after generation to label steps.
        attention_mask: Optional attention mask.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        steering_hook_ctx: Optional context manager (e.g. last_token_steering_hook)
            to wrap around the generate() call. None for unsteered tracing.
        verbalize_every_n: Only verbalize every Nth step to reduce SGLang calls.
        verbalize_max_tokens: max_new_tokens budget per verbalization.

    Returns:
        List of GenerationStep objects with .description populated for verbalized steps.
    """
    steps: list[GenerationStep] = []
    store = ActivationStore()
    processor = ActivationCapturingProcessor(store, capture_layer, steps, tokenizer)

    gen_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "logits_processor": [processor],
        "return_dict_in_generate": False,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        with extraction_hooks(
            model,
            [capture_layer],
            store,
            position="last",
            attention_mask=attention_mask,
        ):
            if steering_hook_ctx is not None:
                with steering_hook_ctx:
                    output_ids = model.generate(**gen_kwargs)
            else:
                output_ids = model.generate(**gen_kwargs)

    # Decode full generated text (prompt stripped)
    prompt_len = input_ids.shape[1]
    full_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    logger.info("Generated %d tokens: %s...", len(steps), full_text[:80])

    # Verbalize captured activations (only every N steps)
    steps_to_verbalize = [s for i, s in enumerate(steps) if i % verbalize_every_n == 0]
    if steps_to_verbalize:
        acts = np.stack([s.activation for s in steps_to_verbalize])  # (K, d_model)
        logger.info("Verbalizing %d/%d generation steps...", len(steps_to_verbalize), len(steps))
        descriptions = verbalizer.verbalize_batch(
            acts,
            temperature=0.3,
            max_new_tokens=verbalize_max_tokens,
        )
        for step, desc in zip(steps_to_verbalize, descriptions):
            step.description = desc

    # Attach generated text to the step list as a convenience attribute
    for step in steps:
        step.__dict__["_full_generated_text"] = full_text

    return steps, full_text


def compute_steering_norm_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    capture_layer: int,
    steering_direction: torch.Tensor,
    coefficient: float,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
) -> list[float]:
    """
    Measure how much the steering vector's component in the residual stream
    changes at each generation step. Returns a list of cosine similarities
    between the residual stream and the steering direction at each step.

    This answers: "does the model 'notice' the injected vector and attenuate it?"

    Returns:
        List of cosine similarities (step i → cos_sim[i]) — one per generated token.
    """
    steps: list[GenerationStep] = []
    store = ActivationStore()
    processor = ActivationCapturingProcessor(store, capture_layer, steps, tokenizer)

    gen_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "logits_processor": [processor],
        "return_dict_in_generate": False,
        "do_sample": False,
    }

    d = steering_direction.float().cpu().numpy()
    d = d / (np.linalg.norm(d) + 1e-8)

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    cos_sims: list[float] = []
    for step in steps:
        act = step.activation
        act_norm = act / (np.linalg.norm(act) + 1e-8)
        cos_sims.append(float(np.dot(act_norm, d)))

    return cos_sims
