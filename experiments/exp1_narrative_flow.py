"""
Experiment 1: Narrative Flow

For a set of prompts, extract residual stream activations at every N layers
and verbalize them with the NLA AV. Produces a layer-by-layer "computational
narrative" showing how the model's internal representation evolves.

Usage:
    python experiments/exp1_narrative_flow.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --av-checkpoint /path/to/qwen2.5-7b-av \
        --sglang-url http://localhost:30000 \
        --output results/exp1_flow.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[1]))
from nla_steering.activation_extractor import extract_activations
from nla_steering.nla_client import NLAVerbalizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

PROMPTS = [
    "The capital of France is",
    "Explain why the sky is blue in simple terms.",
    "What is 17 multiplied by 23?",
    "Write a short poem about autumn.",
    "Is it ethical to lie to protect someone's feelings?",
    "The largest planet in the solar system is",
    "Translate 'hello' to Spanish.",
    "What causes rainbows to appear?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 1: Narrative Flow across layers")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--av-checkpoint", required=True, help="Path to AV checkpoint dir")
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--layer-stride", type=int, default=4, help="Sample every N layers")
    p.add_argument("--output", default="results/exp1_flow.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompts", nargs="*", help="Override default prompts")
    return p.parse_args()


def get_layer_indices(model: AutoModelForCausalLM, stride: int) -> list[int]:
    n = len(model.model.layers)
    return list(range(0, n, stride))


def run(args: argparse.Namespace) -> None:
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    verbalizer = NLAVerbalizer(
        checkpoint_dir=args.av_checkpoint,
        tokenizer=tokenizer,
        model=model,
        sglang_url=args.sglang_url,
    )

    if not verbalizer.health_check():
        logger.error("SGLang server not reachable at %s", args.sglang_url)
        sys.exit(1)

    prompts = args.prompts or PROMPTS
    layer_indices = get_layer_indices(model, args.layer_stride)
    logger.info("Probing %d layers: %s", len(layer_indices), layer_indices)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for prompt in tqdm(prompts, desc="Prompts"):
        enc = tokenizer(prompt, return_tensors="pt").to(args.device)
        activations = extract_activations(
            model,
            enc["input_ids"],
            layer_indices,
            attention_mask=enc.get("attention_mask"),
        )

        narrative: list[dict] = []
        layer_acts = [activations[l][0] for l in layer_indices]  # list of (d_model,)

        descriptions = verbalizer.verbalize_batch(
            torch.stack(layer_acts).cpu().numpy()
        )

        for layer_idx, desc in zip(layer_indices, descriptions):
            narrative.append({"layer": layer_idx, "description": desc})

        record = {"prompt": prompt, "narrative": narrative}
        results.append(record)

        # Live console display
        table = Table(title=f"Prompt: {prompt[:60]}")
        table.add_column("Layer", style="cyan", width=8)
        table.add_column("Internal Description", style="white")
        for entry in narrative:
            table.add_row(str(entry["layer"]), entry["description"])
        console.print(table)

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("Saved %d results to %s", len(results), args.output)


if __name__ == "__main__":
    run(parse_args())
