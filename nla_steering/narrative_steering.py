"""
Experiment 3: Narrative-Driven Steering.

Pipeline:
  1. User writes a desired internal state in natural language
  2. NLA AR (Activation Reconstructor) converts the text → target activation
  3. steering_vec = AR(text) - original_activation
  4. Inject at layer L during generation

This gives a fully natural-language interface to the residual stream.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from nla_steering.activation_extractor import extract_activations
from nla_steering.steering import steering_hook

logger = logging.getLogger(__name__)


class NarrativeSteerer:
    """
    Steers a model's internal representation toward a text description of
    a desired activation state, using the NLA Activation Reconstructor.

    Requires both AV and AR checkpoints (same directory in the NLA release).
    """

    def __init__(
        self,
        ar_checkpoint_dir: str,
        sglang_url: str = "http://localhost:30000",
        device: str = "cpu",
    ) -> None:
        try:
            from nla_inference import NLAClient  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "nla_inference not found. Install from https://github.com/kitft/nla-inference"
            ) from e

        self._ar_client = NLAClient(
            checkpoint_dir=ar_checkpoint_dir,
            sglang_url=sglang_url,
            device=device,
        )
        self._layer_idx: int = self._ar_client.meta.layer_idx
        self._d_model: int = self._ar_client.meta.d_model
        logger.info(
            "NarrativeSteerer ready — layer=%d d_model=%d",
            self._layer_idx,
            self._d_model,
        )

    def text_to_activation(self, description: str) -> np.ndarray:
        """
        Run the AR to reconstruct an activation from a natural language description.

        Returns:
            numpy array of shape (d_model,).
        """
        return self._ar_client.reconstruct(description)

    def compute_delta(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        target_description: str,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Compute the steering delta between the model's current internal state
        and the desired description.

        Returns:
            (delta_tensor, original_activation, target_activation)
            delta_tensor: shape (d_model,), to be injected at self._layer_idx
        """
        enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        activations = extract_activations(
            model,
            enc["input_ids"],
            [self._layer_idx],
            attention_mask=enc.get("attention_mask"),
        )
        original = activations[self._layer_idx][0].cpu().numpy()  # (d_model,)
        target = self.text_to_activation(target_description)      # (d_model,)
        delta = torch.from_numpy((target - original).astype(np.float32))
        return delta, original, target

    def steered_generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        target_description: str,
        coefficient: float = 1.0,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> dict[str, str]:
        """
        Generate text with the model steered toward `target_description`.

        Returns dict with keys:
            - "baseline": generation without steering
            - "steered": generation with narrative steering
            - "original_description": (if AV provided, else empty)
            - "target_description": the input description
        """
        enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        delta, _, _ = self.compute_delta(model, tokenizer, prompt, target_description)

        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        with torch.no_grad():
            baseline_ids = model.generate(**gen_kwargs)

        with torch.no_grad():
            with steering_hook(model, self._layer_idx, delta, coefficient):
                steered_ids = model.generate(**gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        baseline_text = tokenizer.decode(
            baseline_ids[0][prompt_len:], skip_special_tokens=True
        )
        steered_text = tokenizer.decode(
            steered_ids[0][prompt_len:], skip_special_tokens=True
        )

        return {
            "baseline": baseline_text,
            "steered": steered_text,
            "target_description": target_description,
        }
