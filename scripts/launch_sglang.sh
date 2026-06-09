#!/bin/bash
# Launch SGLang server for an NLA AV or AR checkpoint.
# Usage: ./scripts/launch_sglang.sh <checkpoint_path> [port]
#
# On Lambda Labs (H100):
#   ./scripts/launch_sglang.sh /checkpoints/qwen2.5-7b-av 30000

set -euo pipefail

CHECKPOINT=${1:?Usage: $0 <checkpoint_path> [port]}
PORT=${2:-30000}

mkdir -p logs

echo "Launching SGLang server"
echo "  Checkpoint : $CHECKPOINT"
echo "  Port       : $PORT"

python -m sglang.launch_server \
    --model-path "$CHECKPOINT" \
    --port "$PORT" \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    2>&1 | tee "logs/sglang_${PORT}.log"
