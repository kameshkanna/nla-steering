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

---

## Step 1 — Derive French vectors

Only needed once. Extracts CAA directions from 50 contrastive French/English question pairs, capturing activations at the first generated token position (not the last prompt token — see below).

```bash
python experiments/derive_french_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --norm-profile experiments/steering_data/qwen2.5-7b-norm-profile.csv \
    --output-dir experiments/steering_data
```

Pre-derived vectors are already bundled in `experiments/steering_data/french_vectors.npz`.

### Why first-completion-token extraction matters

Both positive and negative prompts end with the same `<|im_start|>assistant\n` template token. Extracting at the last *prompt* token gives a near-zero contrastive signal — the model hasn't committed to a language yet. Extracting at the first *generated* token captures the activation after the model has committed: French prompt → first token is "La"/"Le"/"Quatre", English prompt → "The"/"I"/"Four". Without this fix the derived direction encodes generic non-English multilingual mode (steers toward CJK) rather than French specifically.

---

## Step 2 — Run the sweep

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
| `experiments/results/steering_eval_ks{k}_L*.json` | Per-record metrics: descriptions, cosine shift, detection rate, next-token predictions, concept_shift |
| `experiments/figures/steering_eval_cosine_shift_*.png` | Activation cosine shift vs baseline per layer × mode × behavior |
| `experiments/figures/steering_eval_detection_rate_*.png` | AV concept detection rate under steering |
| `experiments/figures/steering_eval_next_tokens_*.png` | Top-5 next-token prediction shifts |
| `experiments/figures/steering_eval_qualitative_*.png` | Description comparison grid |
| `experiments/figures/steering_eval_concept_shift_*.png` | Baseline → steered semantic diff grid |

Each JSON record contains fields for both `broadcast` and `last_token` inject modes.

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
