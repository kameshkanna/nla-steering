"""
Experiment 4: Looping Verbalizer & Last-Token Ablation

Three sub-experiments that address the layer-20 paper's methodological gaps:

Sub-experiment A — Cross-layer looping (layers 19 → 20 → 21):
  Extract residual stream at layers 19, 20, 21 in one forward pass. Verbalize
  each layer's activation, conditioning each call on the prior layer's verbalization
  via prefix injection. This builds a depth-aware narrative of how concepts evolve
  across the three layers and tests whether verbalization quality improves when the
  NLA receives context about the preceding computational state.

Sub-experiment B — Across-token looping:
  At a fixed layer (20), verbalize the last-token activation at positions t-2, t-1,
  and t sequentially. Each verbalization is conditioned on the previous one. Tests
  whether sequential context makes verbalizations more coherent and whether there
  is token-to-token semantic drift or continuity.

Sub-experiment C — Last-token ablation:
  For each prompt:
    1. Verbalize the last-token activation at layer 20 → V_last
    2. Decode the model's greedy next-token prediction → token_pred
    3. Verbalize the embedding of token_pred → V_next
    4. Compute:
       a. Cosine similarity between AR(V_last) and AR(V_next) (vector space)
       b. Token-level overlap between V_last and V_next text
    This measures how committed the model's internal state is to the next token:
    high similarity = committed, low similarity = still in superposition.

Key design decision: we reuse the existing HFVerbalizer from exp2 rather than
reimplementing verbalization. Cross-layer looping is achieved by prepending prior
verbalizations as plain-text conditioning in the AV prompt (the injection character
still carries the activation — the conditioning is additive context before <concept>).

Usage:
    python experiments/exp4_looping_verbalizer.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --av-checkpoint checkpoints/grpo/final_av \\
        --nla-meta data/labeled/nla_meta_av.yaml \\
        --sub-experiments cross_layer across_token last_token \\
        --output results/exp4_looping.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from peft import PeftModel
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_steering.activation_extractor import (
    ActivationStore,
    extract_activations,
    extraction_hooks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Eval prompts — mix of factual recall, reasoning, and open-ended
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    "What is the derivative of x^2 sin(x)?",
    "Explain why gradient descent can get stuck in local minima.",
    "What were the main causes of World War I?",
    "Is the number 97 prime?",
    "Describe how attention mechanisms work in transformers.",
    "What is 17 multiplied by 23?",
    "Why is the sky blue?",
    "What is the capital of Australia?",
]


# ---------------------------------------------------------------------------
# Conditioned verbalizer
# ---------------------------------------------------------------------------

@dataclass
class VerbalizationResult:
    """Result from one verbalization call."""

    layer_idx: int
    token_position: int  # -1 for "last token"
    activation: np.ndarray  # (d_model,) float32
    verbalization: str
    conditioning_context: str  # prior verbalization text used as prefix (empty if none)


class ConditionedHFVerbalizer:
    """
    Verbalizer that optionally prepends prior verbalization text as conditioning
    context before the <concept> injection in the AV prompt.

    This implements the "looping" protocol: verbalization at layer L is conditioned
    on the verbalization produced at layer L-1 (or token t-1), so the model builds
    a running narrative rather than isolated snapshots.

    The conditioning is injected into the user-turn content before the <concept>
    block:

        "...Prior context: <prior verbalization here>...
         Here is the vector: <concept>{injection_char}</concept>"

    This keeps the injection position unambiguous (still the same CJK token) while
    giving the verbalizer semantic context about what came before.
    """

    def __init__(
        self,
        base_model_name: str,
        av_checkpoint: str,
        tokenizer: AutoTokenizer,
        device: torch.device,
        nla_meta_path: Optional[str] = None,
    ) -> None:
        # Resolve meta path:
        #   1. explicit --nla-meta arg
        #   2. <checkpoint>/nla_meta.yaml  (kitft released checkpoints)
        #   3. data/labeled/nla_meta_av.yaml  (nla-train pipeline default)
        if nla_meta_path is None:
            candidates = [
                Path(av_checkpoint) / "nla_meta.yaml",
                Path("data/labeled/nla_meta_av.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    nla_meta_path = str(candidate)
                    break
        if nla_meta_path is None or not Path(nla_meta_path).exists():
            raise FileNotFoundError(
                "nla_meta.yaml not found. Tried: "
                f"{Path(av_checkpoint) / 'nla_meta.yaml'}, data/labeled/nla_meta_av.yaml. "
                "Pass --nla-meta explicitly."
            )
        with open(nla_meta_path) as f:
            self._meta = yaml.safe_load(f)

        self._tokenizer = tokenizer
        self._device = device
        self._injection_token_id: int = self._meta["tokens"]["injection_token_id"]
        self._left_neighbor_id: int = self._meta["tokens"]["injection_left_neighbor_id"]
        self._right_neighbor_id: int = self._meta["tokens"]["injection_right_neighbor_id"]
        self._injection_scale: float = self._meta["extraction"]["injection_scale"]
        self._layer_idx: int = self._meta["extraction_layer_index"]
        self._d_model: int = self._meta["d_model"]
        self._injection_char: str = self._meta["tokens"]["injection_char"]

        logger.info("Loading AV model from %s", av_checkpoint)
        av_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map={"": str(device)},
            attn_implementation="eager",
        )
        self._av_model = PeftModel.from_pretrained(av_base, av_checkpoint, is_trainable=False)
        self._av_model.eval()
        self._embed_layer = self._av_model.get_input_embeddings()

        logger.info(
            "ConditionedHFVerbalizer ready — layer=%d d_model=%d",
            self._layer_idx,
            self._d_model,
        )

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    def _build_conditioned_prompt(self, prior_context: str) -> str:
        """
        Build the AV prompt string with optional prior-context conditioning.

        If prior_context is non-empty, it is injected before the <concept> block
        as: "Prior layer context: <prior_context>\n\n"
        """
        base_template = self._meta["prompt_templates"]["av"]
        if prior_context:
            conditioned_template = base_template.replace(
                "Here is the vector:",
                f"Prior layer context: {prior_context.strip()}\n\nHere is the vector:",
            )
        else:
            conditioned_template = base_template
        return conditioned_template.format(injection_char=self._injection_char)

    def _build_embeds(
        self,
        activations: np.ndarray,
        prior_contexts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (input_embeds, attention_mask) for a batch where each item may have
        a different prior context string.

        Because contexts differ per item, we must build each sequence individually
        and then left-pad to a common length.

        Returns:
            embeds: (N, max_seq_len, d_model) bfloat16
            attn_mask: (N, max_seq_len) long
        """
        N = len(activations)
        all_ids: list[torch.Tensor] = []

        for i in range(N):
            prompt_content = self._build_conditioned_prompt(prior_contexts[i])
            formatted: str = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids: list[int] = self._tokenizer.encode(formatted, add_special_tokens=False)
            all_ids.append(torch.tensor(ids, dtype=torch.long))

        max_len = max(t.shape[0] for t in all_ids)
        pad_id = self._tokenizer.eos_token_id

        padded_ids = torch.full((N, max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(N, max_len, dtype=torch.long)

        for i, ids_t in enumerate(all_ids):
            seq_len = ids_t.shape[0]
            padded_ids[i, max_len - seq_len :] = ids_t
            attn_mask[i, max_len - seq_len :] = 1

        padded_ids = padded_ids.to(self._device)
        attn_mask = attn_mask.to(self._device)

        embeds = self._embed_layer(padded_ids).clone()

        # Inject activation vectors at the marked injection position for each batch item.
        # Scans for [left_neighbor, injection_token, right_neighbor] triplet; falls back
        # to first occurrence of injection_token_id if neighbors don't match.
        ids_np = padded_ids.cpu().numpy()
        act_tensor = torch.tensor(activations, dtype=embeds.dtype, device=self._device)
        for b in range(N):
            inject_pos: Optional[int] = None
            seq = ids_np[b]
            for j in range(1, len(seq) - 1):
                if (
                    seq[j] == self._injection_token_id
                    and seq[j - 1] == self._left_neighbor_id
                    and seq[j + 1] == self._right_neighbor_id
                ):
                    inject_pos = j
                    break
            if inject_pos is None:
                candidates = (seq == self._injection_token_id).nonzero()[0]
                if len(candidates) == 0:
                    raise RuntimeError(
                        f"Injection token {self._injection_token_id} not found in prompt "
                        f"for batch item {b}."
                    )
                inject_pos = int(candidates[0])
            act = act_tensor[b].float()
            norm = act.norm()
            if norm > 1e-8:
                act = act / norm
            embeds[b, inject_pos] = (act * self._injection_scale).to(embeds.dtype)

        return embeds, attn_mask

    @torch.no_grad()
    def verbalize_batch(
        self,
        activations: np.ndarray | torch.Tensor,
        prior_contexts: Optional[list[str]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 96,
    ) -> list[str]:
        """
        Verbalize a batch of activations with optional per-item conditioning.

        Args:
            activations: (N, d_model) float32.
            prior_contexts: List of N conditioning strings (empty string = no conditioning).
            temperature: Sampling temperature (0 = greedy).
            max_new_tokens: Max tokens per description.

        Returns:
            List of N verbalization strings.
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.float().cpu().numpy()
        activations = activations.astype(np.float32)
        N = len(activations)

        if prior_contexts is None:
            prior_contexts = [""] * N
        if len(prior_contexts) != N:
            raise ValueError(f"prior_contexts length {len(prior_contexts)} != activations length {N}")

        embeds, attn_mask = self._build_embeds(activations, prior_contexts)

        do_sample = temperature > 0.0
        out_ids = self._av_model.generate(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        descriptions: list[str] = []
        for ids in out_ids:
            raw = self._tokenizer.decode(ids, skip_special_tokens=True)
            m = re.search(r"<explanation>(.*?)</explanation>", raw, re.DOTALL)
            descriptions.append(m.group(1).strip() if m else raw.strip())

        del embeds, out_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return descriptions

    def verbalize(
        self,
        activation: np.ndarray | torch.Tensor,
        prior_context: str = "",
        temperature: float = 0.0,
        max_new_tokens: int = 96,
    ) -> str:
        """Verbalize a single activation with optional conditioning."""
        if isinstance(activation, torch.Tensor):
            activation = activation.float().cpu().numpy()
        activation = activation.squeeze()
        return self.verbalize_batch(
            activation[None], [prior_context], temperature, max_new_tokens
        )[0]


# ---------------------------------------------------------------------------
# Sub-experiment A: Cross-layer looping (layers 19 → 20 → 21)
# ---------------------------------------------------------------------------

def run_cross_layer_looping(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer: ConditionedHFVerbalizer,
    prompts: list[str],
    layers: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[dict]:
    """
    For each prompt, extract residual stream at layers 19, 20, 21 simultaneously
    and verbalize each layer conditioned on the previous layer's verbalization.

    Returns a list of result dicts, one per prompt.
    """
    results: list[dict] = []

    for prompt in tqdm(prompts, desc="[A] Cross-layer looping", dynamic_ncols=True):
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(chat_ids, "input_ids"):
            chat_ids = chat_ids.input_ids
        chat_ids = chat_ids.to(device)
        attn_mask = torch.ones_like(chat_ids)

        # Single forward pass captures all three layers
        layer_acts = extract_activations(model, chat_ids, layers, attn_mask)

        chain: list[dict] = []
        prior_context = ""

        for layer_idx in layers:
            act = layer_acts[layer_idx][0].cpu().float().numpy()  # (d_model,)
            verbalization = verbalizer.verbalize(
                act,
                prior_context=prior_context,
                max_new_tokens=max_new_tokens,
            )
            chain.append({
                "layer": layer_idx,
                "verbalization": verbalization,
                "conditioning_context": prior_context,
                "activation_norm": float(np.linalg.norm(act)),
            })
            prior_context = verbalization  # chain for next layer

        # Compute cosine similarities between adjacent layers
        acts_np = np.stack([layer_acts[l][0].cpu().float().numpy() for l in layers])
        cosine_adjacent = []
        for i in range(len(layers) - 1):
            a = acts_np[i]
            b = acts_np[i + 1]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            cosine_adjacent.append({"layers": [layers[i], layers[i + 1]], "cosine": cos})

        results.append({
            "prompt": prompt,
            "layer_chain": chain,
            "activation_cosine_adjacent": cosine_adjacent,
        })

        # Rich display
        console.rule(f"[bold cyan][A] Cross-layer: {prompt[:60]}[/bold cyan]")
        t = Table(box=box.SIMPLE_HEAD, show_lines=False)
        t.add_column("Layer", style="cyan", width=7)
        t.add_column("Verbalization", style="white", max_width=80)
        t.add_column("Conditioned?", style="magenta", width=12)
        for entry in chain:
            t.add_row(
                str(entry["layer"]),
                entry["verbalization"][:80],
                "yes" if entry["conditioning_context"] else "no",
            )
        console.print(t)
        for c in cosine_adjacent:
            console.print(
                f"  cos({c['layers'][0]},{c['layers'][1]}) = {c['cosine']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Sub-experiment B: Across-token looping
# ---------------------------------------------------------------------------

def _extract_last_token_at_prefix(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    layer_idx: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run a forward pass on input_ids and return the last-token activation
    at layer_idx as a float32 numpy array of shape (d_model,).
    """
    attn_mask = torch.ones_like(input_ids)
    acts = extract_activations(model, input_ids, [layer_idx], attn_mask)
    return acts[layer_idx][0].cpu().float().numpy()


def run_across_token_looping(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer: ConditionedHFVerbalizer,
    prompts: list[str],
    layer_idx: int,
    lookback: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict]:
    """
    At layer_idx, verbalize the last-token activation at positions t-lookback+1
    through t sequentially, conditioning each verbalization on the previous one.

    For a prompt tokenized to length T, we run forward passes on prefixes of
    lengths [T-lookback+1, ..., T] and capture the last-token activation at
    layer_idx for each. This simulates "reading" the residual stream token-by-token
    and building a running narrative.

    Args:
        lookback: Number of sequential token positions to verbalize (default 3
                  gives positions t-2, t-1, t).

    Returns:
        List of result dicts, one per prompt.
    """
    results: list[dict] = []

    for prompt in tqdm(prompts, desc="[B] Across-token looping", dynamic_ncols=True):
        chat_ids_list: list[int] = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        total_len = len(chat_ids_list)

        # We need at least `lookback` tokens available
        start = max(1, total_len - lookback + 1)
        token_positions = list(range(start, total_len + 1))  # prefix lengths

        chain: list[dict] = []
        prior_context = ""

        for prefix_len in token_positions:
            prefix_ids = torch.tensor(
                chat_ids_list[:prefix_len], dtype=torch.long, device=device
            ).unsqueeze(0)

            act = _extract_last_token_at_prefix(model, prefix_ids, layer_idx, device)
            last_token_text = tokenizer.decode([chat_ids_list[prefix_len - 1]])

            verbalization = verbalizer.verbalize(
                act,
                prior_context=prior_context,
                max_new_tokens=max_new_tokens,
            )

            chain.append({
                "prefix_length": prefix_len,
                "last_token": last_token_text,
                "verbalization": verbalization,
                "conditioning_context": prior_context,
                "activation_norm": float(np.linalg.norm(act)),
            })
            prior_context = verbalization

        # Cosine similarity between adjacent token positions
        cosine_adjacent: list[dict] = []
        for i in range(len(chain) - 1):
            a_ids = torch.tensor(
                chat_ids_list[: token_positions[i]], dtype=torch.long, device=device
            ).unsqueeze(0)
            b_ids = torch.tensor(
                chat_ids_list[: token_positions[i + 1]], dtype=torch.long, device=device
            ).unsqueeze(0)
            a_act = _extract_last_token_at_prefix(model, a_ids, layer_idx, device)
            b_act = _extract_last_token_at_prefix(model, b_ids, layer_idx, device)
            cos = float(
                np.dot(a_act, b_act) / (np.linalg.norm(a_act) * np.linalg.norm(b_act) + 1e-8)
            )
            cosine_adjacent.append({
                "prefix_lengths": [token_positions[i], token_positions[i + 1]],
                "cosine": cos,
            })

        results.append({
            "prompt": prompt,
            "layer": layer_idx,
            "token_chain": chain,
            "activation_cosine_adjacent": cosine_adjacent,
        })

        # Rich display
        console.rule(f"[bold green][B] Across-token: {prompt[:60]}[/bold green]")
        t = Table(box=box.SIMPLE_HEAD, show_lines=False)
        t.add_column("Prefix len", style="cyan", width=11)
        t.add_column("Last token", style="yellow", width=14)
        t.add_column("Verbalization", style="white", max_width=75)
        for entry in chain:
            t.add_row(
                str(entry["prefix_length"]),
                repr(entry["last_token"]),
                entry["verbalization"][:75],
            )
        console.print(t)
        for c in cosine_adjacent:
            console.print(
                f"  cos(len={c['prefix_lengths'][0]}, len={c['prefix_lengths'][1]}) "
                f"= {c['cosine']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Sub-experiment C: Last-token ablation
# ---------------------------------------------------------------------------

def _token_overlap(text_a: str, text_b: str) -> float:
    """Unigram token overlap (Jaccard) between two verbalization strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def run_last_token_ablation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer: ConditionedHFVerbalizer,
    prompts: list[str],
    layer_idx: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict]:
    """
    For each prompt:
      1. Verbalize the last-token residual stream at layer_idx → V_last
      2. Decode the model's greedy next-token prediction → token_pred
      3. Verbalize the embedding of token_pred (injected as if it were an activation) → V_next
      4. Compute:
           - Cosine similarity between the raw activation vectors
           - Token-level Jaccard overlap between V_last and V_next text
      5. Also verbalize at each of layers [layer_idx-1, layer_idx, layer_idx+1]
         without conditioning, to measure layer-wise convergence toward next token.

    The key metric: if the residual stream at layer L is "committed" to the next
    token, cos(act_last, embed(next_token)) should be high and the verbalizations
    should share vocabulary. If it is still in superposition, both metrics are low.

    Returns:
        List of result dicts, one per prompt.
    """
    embed_layer = model.get_input_embeddings()
    results: list[dict] = []
    sweep_layers = [layer_idx - 1, layer_idx, layer_idx + 1]

    for prompt in tqdm(prompts, desc="[C] Last-token ablation", dynamic_ncols=True):
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(chat_ids, "input_ids"):
            chat_ids = chat_ids.input_ids
        chat_ids = chat_ids.to(device)
        attn_mask = torch.ones_like(chat_ids)

        # Extract last-token activation at the target layer and sweep layers
        all_layers = list(set(sweep_layers))
        layer_acts = extract_activations(model, chat_ids, all_layers, attn_mask)
        act_last = layer_acts[layer_idx][0].cpu().float().numpy()  # (d_model,)

        # Greedy next-token prediction
        with torch.no_grad():
            logits = model(input_ids=chat_ids, attention_mask=attn_mask).logits
        next_token_id = int(logits[0, -1, :].argmax(dim=-1).item())
        next_token_text = tokenizer.decode([next_token_id])

        # Embedding of the predicted next token (d_model,) float32
        next_token_embed = (
            embed_layer(torch.tensor([next_token_id], device=device))
            .squeeze(0)
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        # Verbalize last-token activation (no conditioning)
        v_last = verbalizer.verbalize(act_last, prior_context="", max_new_tokens=max_new_tokens)

        # Verbalize next-token embedding — inject the embedding directly as if
        # it were a residual stream activation. Scale to match the NLA's expected
        # injection_scale so the injection magnitude is comparable.
        embed_norm = float(np.linalg.norm(next_token_embed))
        if embed_norm > 1e-8:
            next_token_embed_scaled = next_token_embed * (
                verbalizer._injection_scale / embed_norm
            )
        else:
            next_token_embed_scaled = next_token_embed
        v_next = verbalizer.verbalize(
            next_token_embed_scaled, prior_context="", max_new_tokens=max_new_tokens
        )

        # Cosine similarity between act_last and next_token_embed (raw, unnormalized)
        cos_act_embed = float(
            np.dot(act_last, next_token_embed)
            / (np.linalg.norm(act_last) * np.linalg.norm(next_token_embed) + 1e-8)
        )

        # Jaccard unigram overlap between the two verbalization texts
        text_overlap = _token_overlap(v_last, v_next)

        # Layer-sweep verbalizations (no conditioning, for convergence analysis)
        sweep_results: list[dict] = []
        for sweep_l in sweep_layers:
            if sweep_l < 0 or sweep_l not in layer_acts:
                continue
            act_sweep = layer_acts[sweep_l][0].cpu().float().numpy()
            v_sweep = verbalizer.verbalize(
                act_sweep, prior_context="", max_new_tokens=max_new_tokens
            )
            cos_sweep_embed = float(
                np.dot(act_sweep, next_token_embed)
                / (np.linalg.norm(act_sweep) * np.linalg.norm(next_token_embed) + 1e-8)
            )
            overlap_sweep = _token_overlap(v_sweep, v_next)
            sweep_results.append({
                "layer": sweep_l,
                "verbalization": v_sweep,
                "cos_with_next_token_embed": cos_sweep_embed,
                "jaccard_with_v_next": overlap_sweep,
            })

        results.append({
            "prompt": prompt,
            "layer": layer_idx,
            "v_last": v_last,
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "v_next": v_next,
            "cos_act_last_vs_embed": cos_act_embed,
            "jaccard_v_last_vs_v_next": text_overlap,
            "layer_sweep": sweep_results,
        })

        # Rich display
        console.rule(f"[bold yellow][C] Last-token ablation: {prompt[:60]}[/bold yellow]")
        console.print(f"  [dim]V_last (layer {layer_idx}):[/dim]  {v_last[:100]}")
        console.print(f"  [dim]Next token:[/dim]          {repr(next_token_text)}")
        console.print(f"  [dim]V_next (embed):[/dim]      {v_next[:100]}")
        console.print(
            f"  [cyan]cos(act_last, embed(t+1)) = {cos_act_embed:.4f}[/cyan]  "
            f"[magenta]Jaccard = {text_overlap:.4f}[/magenta]"
        )

        t = Table(box=box.SIMPLE_HEAD, show_lines=False, title="Layer sweep vs. next token")
        t.add_column("Layer", style="cyan", width=7)
        t.add_column("cos(sweep, embed)", style="magenta", width=18)
        t.add_column("Jaccard vs V_next", style="yellow", width=18)
        t.add_column("Verbalization", style="white", max_width=60)
        for sr in sweep_results:
            t.add_row(
                str(sr["layer"]),
                f"{sr['cos_with_next_token_embed']:.4f}",
                f"{sr['jaccard_with_v_next']:.4f}",
                sr["verbalization"][:60],
            )
        console.print(t)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 4: Looping Verbalizer & Last-Token Ablation"
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--av-checkpoint", required=True)
    p.add_argument(
        "--nla-meta",
        default=None,
        help="Path to nla_meta.yaml. Defaults to <av-checkpoint>/nla_meta.yaml.",
    )
    p.add_argument(
        "--sub-experiments",
        nargs="+",
        choices=["cross_layer", "across_token", "last_token"],
        default=["cross_layer", "across_token", "last_token"],
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[19, 20, 21],
        help="Layer indices for cross-layer looping (default: 19 20 21)",
    )
    p.add_argument(
        "--nla-layer",
        type=int,
        default=20,
        help="Central NLA layer for across-token and last-token experiments",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=3,
        help="Number of token positions to loop over in across-token experiment",
    )
    p.add_argument(
        "--verbalize-max-tokens",
        type=int,
        default=96,
        help="Max tokens per verbalization",
    )
    p.add_argument("--output", default="results/exp4_looping.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logger.info("Loading base model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="eager",
    )
    model.eval()

    verbalizer = ConditionedHFVerbalizer(
        base_model_name=args.model,
        av_checkpoint=args.av_checkpoint,
        tokenizer=tokenizer,
        device=device,
        nla_meta_path=args.nla_meta,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    if "cross_layer" in args.sub_experiments:
        console.print("\n[bold]Sub-experiment A: Cross-layer looping[/bold]")
        all_results["cross_layer"] = run_cross_layer_looping(
            model=model,
            tokenizer=tokenizer,
            verbalizer=verbalizer,
            prompts=EVAL_PROMPTS,
            layers=args.layers,
            max_new_tokens=args.verbalize_max_tokens,
            device=device,
        )

    if "across_token" in args.sub_experiments:
        console.print("\n[bold]Sub-experiment B: Across-token looping[/bold]")
        all_results["across_token"] = run_across_token_looping(
            model=model,
            tokenizer=tokenizer,
            verbalizer=verbalizer,
            prompts=EVAL_PROMPTS,
            layer_idx=args.nla_layer,
            lookback=args.lookback,
            max_new_tokens=args.verbalize_max_tokens,
            device=device,
        )

    if "last_token" in args.sub_experiments:
        console.print("\n[bold]Sub-experiment C: Last-token ablation[/bold]")
        all_results["last_token"] = run_last_token_ablation(
            model=model,
            tokenizer=tokenizer,
            verbalizer=verbalizer,
            prompts=EVAL_PROMPTS,
            layer_idx=args.nla_layer,
            max_new_tokens=args.verbalize_max_tokens,
            device=device,
        )

    with open(args.output, "w") as f:
        f.write(json.dumps(all_results, ensure_ascii=False, indent=2))

    logger.info("Saved results to %s", args.output)
    console.print(f"\n[bold green]Done.[/bold green] Results at [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    run(parse_args())
