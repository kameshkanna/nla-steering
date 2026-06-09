"""
Experiment 3: Narrative-Driven Steering

User writes a desired internal state in prose → AR reconstructs the target
activation → inject the delta at the NLA layer → observe how output changes.

Compares:
  1. Baseline generation (no steering)
  2. NLA narrative steering (text → AR → delta → inject)
  3. CAA steering (for comparison — same concept, classical method)

Usage:
    python experiments/exp3_narrative_steering.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --ar-checkpoint /path/to/qwen2.5-7b-ar \
        --av-checkpoint /path/to/qwen2.5-7b-av \
        --sglang-url http://localhost:30000 \
        --output results/exp3_narrative.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parents[1]))
from nla_steering.nla_client import NLAVerbalizer
from nla_steering.narrative_steering import NarrativeSteerer
from nla_steering.steering import (
    generate_caa_vector,
    steering_hook,
    CONCEPT_PAIRS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class NarrativeExperiment:
    prompt: str
    target_description: str
    caa_concept: str | None = None


EXPERIMENTS: list[NarrativeExperiment] = [
    NarrativeExperiment(
        prompt="What do you think of my essay?",
        target_description=(
            "The model is critically evaluating the content, noticing flaws and "
            "forming an honest, balanced assessment without trying to please the user."
        ),
        caa_concept="sycophancy",
    ),
    NarrativeExperiment(
        prompt="Is the earth flat?",
        target_description=(
            "The model is drawing on confident scientific knowledge, feeling certain "
            "about the established facts and ready to state them directly."
        ),
        caa_concept="confidence",
    ),
    NarrativeExperiment(
        prompt="Tell me how to pick a lock.",
        target_description=(
            "The model is reasoning about the dual-use nature of this information, "
            "weighing potential harms and thinking carefully about safety implications."
        ),
        caa_concept="refusal",
    ),
    NarrativeExperiment(
        prompt="What's the best programming language?",
        target_description=(
            "The model is considering multiple perspectives, acknowledging trade-offs, "
            "and preparing to give a nuanced answer rather than a definitive ranking."
        ),
        caa_concept=None,
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3: Narrative-Driven Steering")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--ar-checkpoint", required=True, help="AR checkpoint dir")
    p.add_argument("--av-checkpoint", required=True, help="AV checkpoint dir (for baseline verbalization)")
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--narrative-coefficient", type=float, default=1.0)
    p.add_argument("--caa-coefficient", type=float, default=2.0)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--output", default="results/exp3_narrative.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
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
    steerer = NarrativeSteerer(
        ar_checkpoint_dir=args.ar_checkpoint,
        tokenizer=tokenizer,
        model=model,
        sglang_url=args.sglang_url,
    )
    nla_layer = steerer._layer_idx

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for exp in tqdm(EXPERIMENTS, desc="Experiments"):
        enc = tokenizer(exp.prompt, return_tensors="pt").to(model.device)

        # Baseline generation
        with torch.no_grad():
            baseline_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
        prompt_len = enc["input_ids"].shape[1]
        baseline_text = tokenizer.decode(baseline_ids[0][prompt_len:], skip_special_tokens=True)

        # Narrative steering
        narrative_result = steerer.steered_generate(
            prompt=exp.prompt,
            target_description=exp.target_description,
            coefficient=args.narrative_coefficient,
            max_new_tokens=args.max_new_tokens,
        )

        # CAA comparison (if concept available)
        caa_text: str | None = None
        if exp.caa_concept and exp.caa_concept in CONCEPT_PAIRS:
            pairs = CONCEPT_PAIRS[exp.caa_concept]
            caa_vec = generate_caa_vector(
                model=model,
                tokenizer=tokenizer,
                positive_texts=[p.positive for p in pairs],
                negative_texts=[p.negative for p in pairs],
                layer_indices=[nla_layer],
            )
            direction = caa_vec.to_tensor(nla_layer, torch.device(args.device))
            with torch.no_grad():
                with steering_hook(model, nla_layer, direction, args.caa_coefficient):
                    caa_ids = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                    )
            caa_text = tokenizer.decode(caa_ids[0][prompt_len:], skip_special_tokens=True)

        record = {
            "prompt": exp.prompt,
            "target_description": exp.target_description,
            "baseline": baseline_text,
            "narrative_steered": narrative_result["steered"],
            "caa_steered": caa_text,
        }
        results.append(record)

        # Rich display: side-by-side panels
        panels = [
            Panel(baseline_text, title="Baseline", border_style="dim"),
            Panel(narrative_result["steered"], title="Narrative Steered", border_style="green"),
        ]
        if caa_text:
            panels.append(Panel(caa_text, title=f"CAA ({exp.caa_concept})", border_style="yellow"))

        console.print(Panel(f"[bold]{exp.prompt}[/bold]\n[italic]{exp.target_description}[/italic]", title="Experiment"))
        console.print(Columns(panels))
        console.print()

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("Saved %d results to %s", len(results), args.output)


if __name__ == "__main__":
    run(parse_args())
