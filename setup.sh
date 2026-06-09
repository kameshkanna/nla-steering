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

echo "==> Installing SGLang (pulls transformers==5.6.0)"
pip install --quiet "sglang[all]>=0.5.6"

echo "==> Patching kernels 0.15.x LayerRepository bug"
# kernels<=0.15.2 raises ValueError if revision/version not set.
# Patch: make revision default to 'main' instead of raising.
LAYER_PY=$(python -c "import kernels.layer.layer as m; print(m.__file__)")
sed -i 's/raise ValueError("Either a revision or a version must be specified.")/revision = "main"/' "$LAYER_PY"

echo "==> Installing nla-inference (NLAClient)"
pip install --quiet git+https://github.com/kitft/nla-inference.git

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "Done. To activate:"
echo "  source $ENV_NAME/bin/activate"
