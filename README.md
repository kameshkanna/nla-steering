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

Download NLA checkpoints (Qwen2.5-7B AV + AR) from the Neuronpedia / paper release page.

Update `configs/qwen_7b.yaml` with your checkpoint paths.

## Running

```bash
# Terminal 1: Launch AV SGLang server
./scripts/launch_sglang.sh /checkpoints/qwen2.5-7b-nla/av 30000

# Terminal 2: Launch AR SGLang server (Exp 3 only)
./scripts/launch_sglang.sh /checkpoints/qwen2.5-7b-nla/ar 30001

# Exp 1
python experiments/exp1_narrative_flow.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint /checkpoints/qwen2.5-7b-nla/av

# Exp 2
python experiments/exp2_steering_interpretation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint /checkpoints/qwen2.5-7b-nla/av \
    --steering-concept sycophancy

# Exp 3
python experiments/exp3_narrative_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --av-checkpoint /checkpoints/qwen2.5-7b-nla/av \
    --ar-checkpoint /checkpoints/qwen2.5-7b-nla/ar
```
