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
        tokenizer: PreTrainedTokenizerBase,
        model: torch.nn.Module,
        sglang_url: str = "http://localhost:30001",
    ) -> None:
        from nla_steering.nla_client import NLAMeta, NLAVerbalizer

        self._meta = NLAMeta.from_checkpoint(ar_checkpoint_dir)
        self._layer_idx: int = self._meta.layer_idx
        self._d_model: int = self._meta.d_model
        self._sglang_url = sglang_url.rstrip("/")
        self._tokenizer = tokenizer
        self._model = model
        logger.info(
            "NarrativeSteerer ready — layer=%d d_model=%d",
            self._layer_idx,
            self._d_model,
        )

    def text_to_activation(self, description: str) -> np.ndarray:
        """
        Call the AR SGLang server to reconstruct an activation from a description.
        Uses the AR prompt template with the description injected.

        Returns:
            numpy array of shape (d_model,).
        """
        import httpx, orjson, re as _re

        prompt_content = self._meta.ar_prompt_template.format(explanation=description)
        ids: list[int] = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        embed_module = self._model.get_input_embeddings()
        embed_weight = embed_module.weight.detach().float().cpu()
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        embeds = embed_weight[ids_tensor].numpy().astype(np.float32)

        payload = {
            "input_embeds": embeds.tolist(),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_hidden_states": True,
        }
        resp = httpx.post(
            self._sglang_url + "/generate",
            content=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # AR returns the activation at the last token position as hidden state
        hidden = np.array(data["hidden_states"][-1], dtype=np.float32)
        return hidden.squeeze()

    def compute_delta(
        self,
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
        enc = self._tokenizer(prompt, return_tensors="pt").to(
            next(self._model.parameters()).device
        )
        activations = extract_activations(
            self._model,
            enc["input_ids"],
            [self._layer_idx],
            attention_mask=enc.get("attention_mask"),
        )
        original = activations[self._layer_idx][0].cpu().numpy()
        target = self.text_to_activation(target_description)
        delta = torch.from_numpy((target - original).astype(np.float32))
        return delta, original, target

    def steered_generate(
        self,
        prompt: str,
        target_description: str,
        coefficient: float = 1.0,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> dict[str, str]:
        """
        Generate text with the model steered toward `target_description`.

        Returns dict with keys: "baseline", "steered", "target_description".
        """
        enc = self._tokenizer(prompt, return_tensors="pt").to(
            next(self._model.parameters()).device
        )
        delta, _, _ = self.compute_delta(prompt, target_description)

        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        with torch.no_grad():
            baseline_ids = self._model.generate(**gen_kwargs)

        with torch.no_grad():
            with steering_hook(self._model, self._layer_idx, delta, coefficient):
                steered_ids = self._model.generate(**gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        baseline_text = self._tokenizer.decode(
            baseline_ids[0][prompt_len:], skip_special_tokens=True
        )
        steered_text = self._tokenizer.decode(
            steered_ids[0][prompt_len:], skip_special_tokens=True
        )

        return {
            "baseline": baseline_text,
            "steered": steered_text,
            "target_description": target_description,
        }
