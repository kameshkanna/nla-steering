#!/bin/bash
set -euo pipefail

ENV_NAME="nla-steering"
PYTHON="python3.11"

echo "==> Checking Python"
if ! command -v $PYTHON &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install it first."
    exit 1
fi
$PYTHON --version

echo "==> Creating virtual environment: $ENV_NAME"
$PYTHON -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing core dependencies"
pip install --quiet \
    "torch>=2.3.0" \
    "safetensors>=0.4.0" \
    "numpy>=1.26.0" \
    "httpx>=0.27.0" \
    "pyyaml>=6.0" \
    "orjson>=3.9.0" \
    "tqdm>=4.66.0" \
    "datasets>=2.20.0" \
    "pandas>=2.2.0" \
    "matplotlib>=3.9.0" \
    "seaborn>=0.13.0" \
    "rich>=13.7.0"

echo "==> Installing repeng (steering vectors)"
pip install --quiet repeng

echo "==> Installing SGLang 0.5.10 + transformers 5.3.0"
# sglang>=0.5.11 pins transformers==5.6.0 which hard-requires kernels<=0.15.2
# with a broken LayerRepository (ValueError on missing revision).
# sglang==0.5.10 pulls transformers==5.3.0 where kernels is optional only.
pip install --quiet "sglang[all]==0.5.10"
pip install --quiet "transformers==5.3.0" "huggingface_hub>=1.5.0"

echo "==> Installing nla-inference (NLAClient)"
pip install --quiet git+https://github.com/kitft/nla-inference.git

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "Done. To activate:"
echo "  source $ENV_NAME/bin/activate"
