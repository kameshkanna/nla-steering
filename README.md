# NLA Steering Experiments

Three experiments using released [Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/) checkpoints to interpret and control model internals.

## Experiments

### Exp 1 — Narrative Flow
Run NLA AV at every N layers of a forward pass. Produces a layer-by-layer prose description of the model's internal state — a semantic logit lens.

### Exp 2 — Steering Interpretation
Apply CAA steering vectors at the NLA checkpoint layer, capture the post-steering activation, and verbalize it. First human-readable labels for what steering vectors actually encode.

### Exp 3 — Narrative-Driven Steering
Write a desired internal state in prose → AR reconstructs the target activation → inject the delta → observe output change. Compare against classical CAA steering.

## Setup

```bash
pip install -r requirements.txt
pip install -e git+https://github.com/kitft/nla-inference.git#egg=nla_inference
```

## Verified NLA Checkpoint IDs (HuggingFace)

| Model | AV | AR |
|---|---|---|
| Qwen2.5-7B | `kitft/nla-qwen2.5-7b-L20-av` | `kitft/nla-qwen2.5-7b-L20-ar` |
| Gemma-3-12B | `kitft/nla-gemma3-12b-L32-av` | `kitft/nla-gemma3-12b-L32-ar` |
| Gemma-3-27B | `kitft/nla-gemma3-27b-L41-av` | `kitft/nla-gemma3-27b-L41-ar` |
| Llama-3.3-70B | `kitft/Llama-3.3-70B-NLA-L53-av` | `kitft/Llama-3.3-70B-NLA-L53-ar` |

## Running (Qwen2.5-7B, single H100)

```bash
# Download AV checkpoint locally (needed for nla_meta.yaml sidecar)
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir checkpoints/av

# Terminal 1: Launch AV SGLang server (streams from HF on first run)
./scripts/launch_sglang.sh kitft/nla-qwen2.5-7b-L20-av 30000

# Terminal 2 (once server shows "Server is ready"):

# Exp 1 — Narrative Flow
python experiments/exp1_narrative_flow.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint checkpoints/av \
    --sglang-url http://localhost:30000

# Exp 2 — Steering Interpretation
python experiments/exp2_steering_interpretation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint checkpoints/av \
    --steering-concept sycophancy

# Exp 3 — Narrative Steering (needs AR server on port 30001 too)
hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir checkpoints/ar
./scripts/launch_sglang.sh kitft/nla-qwen2.5-7b-L20-ar 30001  # Terminal 3

python experiments/exp3_narrative_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint checkpoints/av \
    --ar-checkpoint checkpoints/ar
```
