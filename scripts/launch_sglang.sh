#!/bin/bash
# Launch SGLang server for an NLA AV or AR checkpoint.
#
# Usage:
#   ./scripts/launch_sglang.sh <hf-model-id-or-local-path> [port]
#
# Examples:
#   ./scripts/launch_sglang.sh kitft/nla-qwen2.5-7b-L20-av 30000
#   ./scripts/launch_sglang.sh kitft/nla-qwen2.5-7b-L20-ar 30001

set -euo pipefail

CHECKPOINT=${1:?Usage: $0 <hf-model-id-or-local-path> [port]}
PORT=${2:-30000}

mkdir -p logs

echo "Launching SGLang server"
echo "  Checkpoint : $CHECKPOINT"
echo "  Port       : $PORT"

sglang serve --model-path "$CHECKPOINT" \
    --port "$PORT" \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    2>&1 | tee "logs/sglang_${PORT}.log"
