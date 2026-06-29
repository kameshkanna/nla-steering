# nla-steering

Activation steering × NLA evaluation for Qwen2.5-7B-Instruct.

Tests whether a trained Activation Verbalizer (AV) can detect what CAA steering vectors are doing to the residual stream. Two behaviors: safety vectors (from [activation-baking](https://github.com/kameshkanna/activation-baking)) and French-language CAA vectors derived from 50 contrastive pairs.

Companion repo to [nla-train](https://github.com/kameshkanna/nla-train). Requires a trained AV checkpoint — use [Kameshr/nla-qwen2.5-7b-L20-av](https://huggingface.co/Kameshr/nla-qwen2.5-7b-L20-av).

---

## Setup

```bash
bash setup_env.sh
source nla-steering-env/bin/activate
```

Installs all dependencies (torch, transformers, peft, accelerate, sglang, repeng, and supporting libs). Requires Python 3.10 and CUDA 12.1.

---

## Experiments

### Exp 1 — Narrative flow

Probes the residual stream at every Nth layer during generation and verbalizes each snapshot, producing a depth-wise narrative of how the model's internal state evolves from prompt to output.

### Exp 2 — Steering vector interpretation

CAA steering vectors are computed for concepts (sycophancy, honesty, refusal, confidence, deception) and injected at varying coefficients. The NLA verbalizes the residual stream before and after steering, and tracks cosine similarity with the steering direction across generation steps to detect self-correction signals.

```bash
python experiments/exp2_steering_interpretation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint checkpoints/grpo/final_av \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --concepts sycophancy honesty \
    --coefficients -10 -5 -2 -1 0 1 2 5 10 \
    --output results/exp2_steering.jsonl
```

Pre-computed vectors can be reloaded with `--load-vectors results/vectors` to skip CAA computation.

### Exp 3 — Narrative steering

Combines NLA-derived narrative directions with CAA vectors to steer generation toward a target semantic concept while verbalizing the trajectory.

### Exp 4 — Looping verbalizer & last-token ablation

Three sub-experiments that address the static, single-snapshot limitation of standard NLA evaluation:

**A — Cross-layer looping (layers 19 → 20 → 21)**
Captures residual stream activations at layers 19, 20, and 21 in a single forward pass. Verbalizes each layer conditioned on the prior layer's verbalization, building a depth-aware narrative. Also computes cosine similarity between adjacent-layer activations to measure geometric drift across depth.

**B — Across-token looping**
At a fixed layer (20), verbalizes the last-token residual stream at sequential token positions (t-2, t-1, t), conditioning each call on the previous verbalization. Tests whether token-to-token context improves semantic coherence.

**C — Last-token ablation**
For each prompt: (1) verbalizes the last-token activation → V_last, (2) decodes the model's greedy next-token prediction, (3) verbalizes its embedding → V_next, (4) computes cosine similarity between the raw vectors and Jaccard text overlap between verbalizations. Also sweeps layers 19/20/21 to show how "commitment" to the next token changes with depth. Low cosine + low Jaccard = residual stream still in superposition; high = committed.

```bash
python experiments/exp4_looping_verbalizer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint checkpoints/grpo/final_av \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --output results/exp4_looping.jsonl

# Run only specific sub-experiments
python experiments/exp4_looping_verbalizer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint checkpoints/grpo/final_av \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --sub-experiments last_token \
    --output results/exp4_last_token.jsonl
```

---

## Steering sweep (Exp 1 / steering_av_eval)

### Step 1 — Derive French vectors

Only needed once. Extracts CAA directions from 50 contrastive French/English question pairs, capturing activations at the first generated token position (not the last prompt token — see below).

```bash
python experiments/derive_french_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --norm-profile experiments/steering_data/qwen2.5-7b-norm-profile.csv \
    --output-dir experiments/steering_data
```

Pre-derived vectors are already bundled in `experiments/steering_data/french_vectors.npz`.

#### Why first-completion-token extraction matters

Both positive and negative prompts end with the same `<|im_start|>assistant\n` template token. Extracting at the last *prompt* token gives a near-zero contrastive signal — the model hasn't committed to a language yet. Extracting at the first *generated* token captures the activation after the model has committed: French prompt → first token is "La"/"Le"/"Quatre", English prompt → "The"/"I"/"Four". Without this fix the derived direction encodes generic non-English multilingual mode (steers toward CJK) rather than French specifically.

### Step 2 — Run the sweep

```bash
bash scripts/run_steering_sweep.sh \
    --layers "18 19 20 21 22" \
    --k-scales "1.0 2.0 3.0 5.0" \
    --av-ckpt /path/to/checkpoints/grpo/final_av \
    --nla-meta /path/to/data/labeled/nla_meta_av.yaml \
    --batch 4096 \
    --av-batch 4096
```

K values follow the actbak norm-profile formula: `K_ℓ = mean_norm_ℓ / √d × k_scale`. Profile values are in `experiments/steering_data/qwen2.5-7b-norm-profile.csv`.

Or run a single eval directly:

```bash
python experiments/steering_av_eval.py \
    --config /path/to/configs/qwen7b_layer20.yaml \
    --av-checkpoint /path/to/checkpoints/grpo/final_av \
    --nla-meta /path/to/data/labeled/nla_meta_av.yaml \
    --probe-layers 18 19 20 21 22 \
    --k-scale 3.0 \
    --base-batch-size 4096 \
    --av-batch-size 4096
```

---

## Outputs

Per run (`ks{k}_L{layers}`):

| File | Description |
| --- | --- |
| `results/exp2_steering.jsonl` | Per-record exp2 traces: descriptions, cos-sim trajectory, generation |
| `results/exp4_looping.jsonl` | Exp4 results: layer chain, token chain, last-token ablation metrics |
| `experiments/results/steering_eval_ks{k}_L*.json` | Per-record metrics: descriptions, cosine shift, detection rate, next-token predictions |
| `experiments/figures/steering_eval_cosine_shift_*.png` | Activation cosine shift vs baseline per layer × mode × behavior |
| `experiments/figures/steering_eval_detection_rate_*.png` | AV concept detection rate under steering |
| `experiments/figures/steering_eval_next_tokens_*.png` | Top-5 next-token prediction shifts |
| `experiments/figures/steering_eval_qualitative_*.png` | Description comparison grid |
| `experiments/figures/steering_eval_concept_shift_*.png` | Baseline → steered semantic diff grid |

Each steering eval JSON record contains fields for both `broadcast` and `last_token` inject modes.

---

## Key results

**French (k_scale=3, Layer 21, broadcast)**: 100% of next-token predictions shift to French tokens. AV detection lifts from 0.20 baseline to 0.54.

**Safety (k_scale=5, broadcast)**: Cosine similarity drops to 0.806, 62% of next-token predictions change. AV detection flat at ~0.26 — the AV was trained on FineWeb and has no vocabulary for behavioral modes like refusal/compliance.

---

## Inject modes

**broadcast**: adds `K·direction` to all token positions. Full sequence representation shifts; AV detects the change.

**last_token**: adds `K·direction` to the final token only. Moves next-token prediction but not the full sequence representation the AV reads. For safety vectors: 0% next-token change at any k_scale.

---

## Norm profile

Pre-computed actbak norm profile for Qwen2.5-7B is in `experiments/steering_data/qwen2.5-7b-norm-profile.csv`. To recompute against a different model, use [activation-baking](https://github.com/kameshkanna/activation-baking).
