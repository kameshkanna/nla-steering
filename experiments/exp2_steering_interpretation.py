"""
Experiment 2: Steering Vector Interpretation via NLA

For a set of prompts:
  1. Extract baseline activation at the NLA checkpoint layer → verbalize
  2. Compute a CAA steering vector from contrast pairs
  3. Apply steering at that layer → capture post-steering activation → verbalize
  4. Diff the two descriptions

This produces the first human-readable labels for what steering vectors encode.

Usage:
    python experiments/exp2_steering_interpretation.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --av-checkpoint /path/to/qwen2.5-7b-av \
        --sglang-url http://localhost:30000 \
        --steering-concept sycophancy \
        --output results/exp2_steering.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parents[1]))
from nla_steering.activation_extractor import extract_activations
from nla_steering.nla_client import NLAVerbalizer
from nla_steering.steering import (
    SteeringVector,
    generate_caa_vector,
    extract_post_steering_activation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


class ContrastPair(NamedTuple):
    positive: str
    negative: str


CONCEPT_PAIRS: dict[str, list[ContrastPair]] = {
    "sycophancy": [
        ContrastPair(
            "I completely agree with everything you said. You're absolutely right.",
            "I disagree with several points here. Let me explain why.",
        ),
        ContrastPair(
            "That's a brilliant idea! I love it exactly as it is.",
            "There are some significant flaws in this approach we should address.",
        ),
        ContrastPair(
            "You're so smart, your analysis is perfect.",
            "Your analysis has some gaps that need to be filled.",
        ),
    ],
    "honesty": [
        ContrastPair(
            "I'll tell you the truth even if it's uncomfortable: ",
            "I'll tell you what you want to hear: ",
        ),
        ContrastPair(
            "The honest answer is I'm uncertain about this.",
            "I'll give you a confident answer even if I'm guessing.",
        ),
        ContrastPair(
            "Factually speaking, the evidence shows the opposite.",
            "To make you feel good, I'll say you're right.",
        ),
    ],
    "refusal": [
        ContrastPair(
            "I can't help with that request as it could cause harm.",
            "Sure, I'll help you with that. Here's how:",
        ),
        ContrastPair(
            "I must decline this request for safety reasons.",
            "Of course! Let me assist you right away.",
        ),
    ],
    "confidence": [
        ContrastPair(
            "I am absolutely certain this is the correct answer.",
            "I'm not sure about this, it might be wrong.",
        ),
        ContrastPair(
            "Definitively, the answer is clear.",
            "There's a lot of uncertainty here.",
        ),
    ],
}

EVAL_PROMPTS = [
    "What do you think of my business plan?",
    "Is climate change real?",
    "Tell me something controversial.",
    "Was my answer correct?",
    "How should I approach this problem?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 2: Steering Vector Interpretation")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--av-checkpoint", required=True)
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument(
        "--steering-concept",
        choices=list(CONCEPT_PAIRS.keys()),
        default="sycophancy",
    )
    p.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Steering coefficients to sweep over",
    )
    p.add_argument("--caa-method", choices=["mean_diff", "pca_diff"], default="mean_diff")
    p.add_argument("--output", default="results/exp2_steering.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    verbalizer = NLAVerbalizer(
        checkpoint_dir=args.av_checkpoint,
        sglang_url=args.sglang_url,
    )
    nla_layer = verbalizer.meta.layer_idx

    if not verbalizer.health_check():
        logger.error("SGLang server not reachable at %s", args.sglang_url)
        sys.exit(1)

    # Build steering vector from contrast pairs
    pairs = CONCEPT_PAIRS[args.steering_concept]
    positives = [p.positive for p in pairs]
    negatives = [p.negative for p in pairs]

    logger.info("Computing CAA vector for concept: %s", args.steering_concept)
    steering_vec = generate_caa_vector(
        model=model,
        tokenizer=tokenizer,
        positive_texts=positives,
        negative_texts=negatives,
        layer_indices=[nla_layer],
        method=args.caa_method,
    )
    direction = steering_vec.to_tensor(nla_layer, device=torch.device(args.device))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for prompt in tqdm(EVAL_PROMPTS, desc="Prompts"):
        enc = tokenizer(prompt, return_tensors="pt").to(args.device)

        # Baseline: no steering
        baseline_acts = extract_activations(
            model,
            enc["input_ids"],
            [nla_layer],
            attention_mask=enc.get("attention_mask"),
        )
        baseline_vec = baseline_acts[nla_layer][0]
        baseline_desc = verbalizer.verbalize(baseline_vec)

        coeff_results: list[dict] = []

        for coeff in args.coefficients:
            if coeff == 0.0:
                steered_desc = baseline_desc
            else:
                steered_act = extract_post_steering_activation(
                    model=model,
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    steer_layer=nla_layer,
                    capture_layer=nla_layer,
                    direction=direction,
                    coefficient=coeff,
                )
                steered_desc = verbalizer.verbalize(steered_act[0])

            coeff_results.append({"coefficient": coeff, "description": steered_desc})

        record = {
            "prompt": prompt,
            "concept": args.steering_concept,
            "baseline_description": baseline_desc,
            "steered": coeff_results,
        }
        results.append(record)

        # Rich display
        console.print(Panel(f"[bold]{prompt}[/bold]", title="Prompt"))
        console.print(f"[cyan]Baseline:[/cyan] {baseline_desc}\n")
        for cr in coeff_results:
            color = "green" if cr["coefficient"] > 0 else "red" if cr["coefficient"] < 0 else "white"
            console.print(f"[{color}]α={cr['coefficient']:+.1f}:[/{color}] {cr['description']}")
        console.print()

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("Saved %d results to %s", len(results), args.output)


if __name__ == "__main__":
    run(parse_args())
