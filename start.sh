#!/usr/bin/env bash
set -euo pipefail

# Official vLLM 0.27.1, one DGX Spark, port 8000, TP=1, Marlin.
# Image pin: vllm/vllm-openai:v0.27.1
# digest sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967

NAME="${NAME:-ornith-1.5-35b-nvfp4-official}"
IMAGE="vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
MODEL="ornith-ai/Ornith-1.5-35B-A3B-NVFP4"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "set HF_TOKEN" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already exists. ./stop.sh first." >&2
  exit 1
fi

docker pull "$IMAGE"

docker run -d --name "$NAME" \
  --gpus all -p 8000:8000 -e HF_TOKEN="$HF_TOKEN" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "$IMAGE" \
  "$MODEL" \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --trust-remote-code \
  --kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend marlin \
  --gpu-memory-utilization 0.4 --max-model-len 262144 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice

echo "waiting for :8000 /health"
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "up: http://127.0.0.1:8000"
    exit 0
  fi
  sleep 2
done

echo "container started but /health did not return 200 in time" >&2
exit 1
