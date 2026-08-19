#!/usr/bin/env bash
set -euo pipefail

# Exact vllm bench serve commands from the spark1 run (19 Aug 2026).
# Default (no args) is table 5: completions 256/512, 8 prompts, max-concurrency 2
# (89.88 out tok/s on that run). Do not merge tables.

NAME="${NAME:-ornith-1.5-35b-nvfp4-official}"
MODEL="ornith-ai/Ornith-1.5-35B-A3B-NVFP4"
TOK="/root/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-NVFP4/snapshots/9660379a2f2c429c465eeed2f3a0f2433fc4381e"
WHICH="${1:-5}"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME is not running. ./run.sh first." >&2
  exit 1
fi

exec_bench() {
  docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$NAME" \
    vllm bench serve "$@"
}

run_1() {
  echo "=== 1. completions 256/128, 20 prompts, request-rate inf (no max-concurrency cap) ==="
  exec_bench \
    --backend openai --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
    --model "$MODEL" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 20 --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el
}

run_2() {
  echo "=== 2. thinking=false openai-chat 512/256, 32 prompts, max-concurrency 4 ==="
  exec_bench \
    --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$TOK" \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 32 --max-concurrency 4 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --extra-body '{"thinking":false,"chat_template_kwargs":{"enable_thinking":false}}' \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --num-warmups 1
}

run_3() {
  echo "=== 3. same as 2, 128 prompts ==="
  exec_bench \
    --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$TOK" \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 128 --max-concurrency 4 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --extra-body '{"thinking":false,"chat_template_kwargs":{"enable_thinking":false}}' \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --num-warmups 1
}

run_4() {
  echo "=== 4. completions 256/128, 8 prompts, max-concurrency 1 ==="
  exec_bench \
    --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model "$MODEL" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 8 --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99
}

run_5() {
  echo "=== 5. completions 256/512, 8 prompts, max-concurrency 2 ==="
  exec_bench \
    --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model "$MODEL" \
    --dataset-name random --random-input-len 256 --random-output-len 512 \
    --num-prompts 8 --max-concurrency 2 --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99
}

run_6() {
  echo "=== 6. thinking=false openai-chat 128/128, 8 prompts, max-concurrency 2 ==="
  exec_bench \
    --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --dataset-name random --random-input-len 128 --random-output-len 128 \
    --num-prompts 8 --max-concurrency 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --extra-body '{"thinking":false,"chat_template_kwargs":{"enable_thinking":false}}' \
    --chat-template-kwargs '{"enable_thinking":false}'
}

case "$WHICH" in
  1) run_1 ;;
  2) run_2 ;;
  3) run_3 ;;
  4) run_4 ;;
  5) run_5 ;;
  6) run_6 ;;
  all)
    run_1
    run_2
    run_3
    run_4
    run_5
    run_6
    ;;
  *)
    echo "usage: $0 [1|2|3|4|5|6|all]" >&2
    echo "default 5 = completions 256/512, 8 prompts, max-concurrency 2" >&2
    exit 1
    ;;
esac
