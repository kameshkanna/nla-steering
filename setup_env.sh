#!/bin/bash
# Steering × AV evaluation environment.
#
# Usage:
#   bash setup_env.sh
#   source nla-steering-env/bin/activate

set -euo pipefail

ENV_NAME="nla-steering-env"
PYTHON="python3.10"

echo "==> Python version"
$PYTHON --version

echo "==> Creating venv: $ENV_NAME"
$PYTHON -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing torch 2.5.1 (CUDA 12.1)"
pip install --quiet \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing transformers + PEFT + accelerate"
pip install --quiet \
    "transformers>=4.47.0" \
    "peft>=0.14.0" \
    "accelerate>=1.2.1"

echo "==> Installing data + util deps"
pip install --quiet \
    "safetensors>=0.4.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "scikit-learn>=1.4.0" \
    "httpx>=0.27.0" \
    "pyyaml>=6.0" \
    "orjson>=3.9.0" \
    "tqdm>=4.66.0" \
    "datasets>=2.20.0" \
    "pandas>=2.2.0" \
    "matplotlib>=3.9.0" \
    "seaborn>=0.13.0" \
    "rich>=13.7.0" \
    "repeng>=0.1.0" \
    "tokenizers>=0.20.0" \
    "huggingface_hub>=0.24.0"

echo "==> Installing SGLang"
pip install --quiet "sglang[all]>=0.5.6"

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "==> Verifying key versions:"
python -c "
import torch, transformers, peft
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  peft:         {peft.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
"

echo ""
echo "Done. Activate with:"
echo "  source $ENV_NAME/bin/activate"
