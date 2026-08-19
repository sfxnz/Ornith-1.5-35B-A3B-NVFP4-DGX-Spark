# Ornith-1.5-35B-A3B-NVFP4 on 1× DGX Spark

Clone, then `HF_TOKEN=… ./run.sh`. `./stop.sh` tears the container down. `./bench.sh` is the measured `vllm bench serve` commands (default: completions 256/512, 8 prompts, max-concurrency 2). `run.sh` is the live docker command with the image digest pinned.

Official vLLM on one NVIDIA DGX Spark (GB10). Measured on spark1, 19 Aug 2026. A stranger should be able to stand up the same serve and re-run the same `vllm bench serve` commands to land inside these tables.

- Official `:8000`, **TP=1**, Marlin MoE, FlashInfer, FP8 KV, MTP 3
- Image pin below, not a floating `:latest`
- Unified memory via `free -h` only. Never `nvidia-smi` VRAM
- Each bench is its own command. Do not merge tables. Do not treat any one tok/s as a general decode number
- Label **max-concurrency** (the flag you passed). The harness also prints peak concurrent; that field was 2× the cap on every dump here. Do not call a max-concurrency=1 run single-stream

Playbook section: [NVIDIA Run Agent Ready Qwen3.6 35B](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md) with the Ornith handle swapped in.

## Hardware

- 1× NVIDIA DGX Spark GB10 (`sm_121`)
- One GPU, tensor-parallel-size 1
- After load (`free -h` on spark1): **62Gi available / 59Gi used / 121Gi total**
- After the bench set: **61Gi available / 59Gi used / 121Gi total**, swap 1.7Gi used

## Image pin

Live container pulled `vllm/vllm-openai:latest`, which resolved to `v0.27.1`. Pin the digest so you do not drift.

```
vllm/vllm-openai:v0.27.1
Repo digest:  sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
Image ID:     sha256:2c211a1273b48e8929f893b267aeb1509e6b84654cdbde1bad56d79e3964224d
Fingerprint:  vllm-0.27.1-48c81327
```

```bash
docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

## Model

- HF: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`
- Snapshot: `9660379a2f2c429c465eeed2f3a0f2433fc4381e`
- Cache after `hf download`: ~22G
- Arch: `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`
- Quant: ModelOpt W4A16_NVFP4 experts + FP8 attention; MTP tensors present

```bash
hf download ornith-ai/Ornith-1.5-35B-A3B-NVFP4
```

## Serve

Container on spark1: `ornith-1.5-35b-nvfp4-official`. Live run used `:latest`; recreate with the digest. `./run.sh` runs this command. `./stop.sh` stops and removes the container. `./bench.sh` runs the measured benches (default table 5).

```bash
docker run -d --name ornith-1.5-35b-nvfp4-official \
  --gpus all -p 8000:8000 -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967 \
  ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --trust-remote-code \
  --kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend marlin \
  --gpu-memory-utilization 0.4 --max-model-len 262144 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

`./run.sh` binds `$HF_CACHE` (default `$HOME/.cache/huggingface`).

Wait until `GET /health` is 200 and `GET /v1/models` lists the model. This load: weights 21.99 GiB in 24.7 s; engine init ~148 s.

**thinking off is required for short generate and for chat benches.** Default chat with thinking on eats the token budget (`content` null, `finish_reason=length`). Ready-gate generate only passed with thinking off.

### Ready-gate

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
    "messages": [{"role":"user","content":"Reply with exactly: pong"}],
    "max_tokens": 128,
    "temperature": 0,
    "thinking": false,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Expected: HTTP 200, `content` = `pong`, `reasoning` null, `finish_reason=stop`, `system_fingerprint` starts `vllm-0.27.1-`. Both `thinking:false` and `enable_thinking:false` are required. `max_tokens:64` without temperature can return null content.

### Non-fatal load warnings

Expected on GB10. Do not abort.

- `Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel.`
- `Checkpoint does not provide a q scaling factor. Setting it to k_scale.`
- `Using KV cache scaling factor 1.0 for fp8_e4m3.`
- `Not enough SMs to use max_autotune_gemm mode`
- First-request Triton JIT warmup on eagle / causal_conv1d / gating

## How to bench

`vllm` is not on the Spark host PATH. Run the client inside the live container, or `./bench.sh` / `./bench.sh 5` / `./bench.sh all`.

```bash
docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  ornith-1.5-35b-nvfp4-official \
  vllm bench serve ...
```

One bench at a time. Serve is `--max-num-seqs 4`. Random dataset. Server-side default temperature (vLLM 0.27.1 no longer forces temperature=0). Seed 0.

Copy concurrency exactly. Inf + 20 prompts ≠ max-concurrency 1 ≠ 2 ≠ 4. TTFT moves by two orders of magnitude.

Do not claim a KV hit rate from this set. `vllm:prefix_cache_hits_total` is unchecked.

---

## 1. Completions 256/128, 20 prompts, request-rate inf (no max-concurrency cap)

Vera PASS as this tool output. Not chat. Not single-stream. Queueing TTFT (mean 12633 ms). Peak concurrent 20. Serve cap `--max-num-seqs 4`.

```bash
./bench.sh 1
```

```
============ Serving Benchmark Result ============
Successful requests:                     20
Failed requests:                         0
Benchmark duration (s):                  27.31
Total input tokens:                      5120
Total generated tokens:                  2560
Request throughput (req/s):              0.73
Output token throughput (tok/s):         93.73
Peak output token throughput (tok/s):    92.00
Peak concurrent requests:                20.00
Total token throughput (tok/s):          281.19
---------------Time to First Token----------------
Mean TTFT (ms):                          12632.97
Median TTFT (ms):                        12431.30
P99 TTFT (ms):                           23753.37
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.10
Median TPOT (ms):                        30.28
P99 TPOT (ms):                           59.82
---------------Inter-token Latency----------------
Mean ITL (ms):                           47.85
Median ITL (ms):                         44.59
P99 ITL (ms):                            100.48
----------------End-to-end Latency----------------
Mean E2EL (ms):                          16963.70
Median E2EL (ms):                        16778.25
P99 E2EL (ms):                           27212.21
---------------Speculative Decoding---------------
Acceptance rate (%):                     13.57
Acceptance length:                       1.41
Drafts:                                  1810
Draft tokens:                            5430
Accepted tokens:                         737
Per-position acceptance (%):
  Position 0:                            37.79
  Position 1:                            2.54
  Position 2:                            0.39
==================================================
```

---

## 2. thinking=false openai-chat 512/256, 32 prompts, max-concurrency 4

Vera PASS as this random thinking=false chat blast. Do not publish as general decode.

```bash
./bench.sh 2
```

```
============ Serving Benchmark Result ============
Successful requests:                     32
Failed requests:                         0
Maximum request concurrency:             4
Benchmark duration (s):                  60.95
Total input tokens:                      16784
Total generated tokens:                  8192
Request throughput (req/s):              0.52
Output token throughput (tok/s):         134.40
Peak output token throughput (tok/s):    96.00
Peak concurrent requests:                8.00
Total token throughput (tok/s):          409.75
---------------Time to First Token----------------
Mean TTFT (ms):                          343.12
Median TTFT (ms):                        234.26
P50 TTFT (ms):                           234.26
P95 TTFT (ms):                           946.19
P99 TTFT (ms):                           946.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.07
Median TPOT (ms):                        27.92
P50 TPOT (ms):                           27.92
P95 TPOT (ms):                           32.39
P99 TPOT (ms):                           32.79
---------------Inter-token Latency----------------
Mean ITL (ms):                           47.30
Median ITL (ms):                         45.37
P50 ITL (ms):                            45.37
P95 ITL (ms):                            49.05
P99 ITL (ms):                            138.96
----------------End-to-end Latency----------------
Mean E2EL (ms):                          7501.50
Median E2EL (ms):                        7474.63
P50 E2EL (ms):                           7474.63
P95 E2EL (ms):                           8583.91
P99 E2EL (ms):                           8756.02
---------------Speculative Decoding---------------
Acceptance rate (%):                     22.31
Acceptance length:                       1.67
Drafts:                                  4898
Draft tokens:                            14694
Accepted tokens:                         3278
Per-position acceptance (%):
  Position 0:                            57.21
  Position 1:                            8.80
  Position 2:                            0.92
==================================================
```

---

## 3. Same command as §2, 128 prompts

Vera PASS as the longer-run repeat. Do not merge with §2. Only `--num-prompts 128` changes.

```bash
./bench.sh 3
```

```
============ Serving Benchmark Result ============
Successful requests:                     128
Failed requests:                         0
Maximum request concurrency:             4
Benchmark duration (s):                  239.12
Total input tokens:                      67121
Total generated tokens:                  32768
Request throughput (req/s):              0.54
Output token throughput (tok/s):         137.04
Peak output token throughput (tok/s):    96.00
Peak concurrent requests:                8.00
Total token throughput (tok/s):          417.74
---------------Time to First Token----------------
Mean TTFT (ms):                          246.80
Median TTFT (ms):                        231.72
P50 TTFT (ms):                           231.72
P95 TTFT (ms):                           321.59
P99 TTFT (ms):                           501.08
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.16
Median TPOT (ms):                        27.96
P50 TPOT (ms):                           27.96
P95 TPOT (ms):                           31.77
P99 TPOT (ms):                           35.17
---------------Inter-token Latency----------------
Mean ITL (ms):                           48.05
Median ITL (ms):                         45.01
P50 ITL (ms):                            45.01
P95 ITL (ms):                            48.76
P99 ITL (ms):                            139.13
----------------End-to-end Latency----------------
Mean E2EL (ms):                          7427.69
Median E2EL (ms):                        7373.78
P50 E2EL (ms):                           7373.78
P95 E2EL (ms):                           8329.95
P99 E2EL (ms):                           9202.23
---------------Speculative Decoding---------------
Acceptance rate (%):                     22.01
Acceptance length:                       1.66
Drafts:                                  19694
Draft tokens:                            59082
Accepted tokens:                         13002
Per-position acceptance (%):
  Position 0:                            56.35
  Position 1:                            8.88
  Position 2:                            0.79
==================================================
```

---

## 4. Completions 256/128, 8 prompts, max-concurrency 1

Vera PASS on the numbers. Fail the single-stream label: peak concurrent was 2.00 against cap 1. Call this **max-concurrency=1 completions**.

```bash
./bench.sh 4
```

```
============ Serving Benchmark Result ============
Successful requests:                     8
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  19.46
Total input tokens:                      2048
Total generated tokens:                  1024
Request throughput (req/s):              0.41
Output token throughput (tok/s):         52.61
Peak output token throughput (tok/s):    37.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          157.83
---------------Time to First Token----------------
Mean TTFT (ms):                          120.92
Median TTFT (ms):                        121.62
P50 TTFT (ms):                           121.62
P95 TTFT (ms):                           126.15
P99 TTFT (ms):                           126.51
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.20
Median TPOT (ms):                        17.79
P50 TPOT (ms):                           17.79
P95 TPOT (ms):                           22.02
P99 TPOT (ms):                           22.62
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.98
Median ITL (ms):                         28.17
P50 ITL (ms):                            28.17
P95 ITL (ms):                            29.57
P99 ITL (ms):                            30.25
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2432.73
Median E2EL (ms):                        2380.89
P50 E2EL (ms):                           2380.89
P95 E2EL (ms):                           2910.11
P99 E2EL (ms):                           2984.42
---------------Speculative Decoding---------------
Acceptance rate (%):                     18.10
Acceptance length:                       1.54
Drafts:                                  661
Draft tokens:                            1983
Accepted tokens:                         359
Per-position acceptance (%):
  Position 0:                            50.68
  Position 1:                            3.48
  Position 2:                            0.15
==================================================
```

---

## 5. Completions 256/512, 8 prompts, max-concurrency 2

Vera PASS. Not chat. Not single-stream. Default `./bench.sh`.

```bash
./bench.sh
```

```
============ Serving Benchmark Result ============
Successful requests:                     8
Failed requests:                         0
Maximum request concurrency:             2
Benchmark duration (s):                  45.57
Total input tokens:                      2048
Total generated tokens:                  4096
Request throughput (req/s):              0.18
Output token throughput (tok/s):         89.88
Peak output token throughput (tok/s):    58.00
Peak concurrent requests:                4.00
Total token throughput (tok/s):          134.81
---------------Time to First Token----------------
Mean TTFT (ms):                          185.83
Median TTFT (ms):                        162.71
P50 TTFT (ms):                           162.71
P95 TTFT (ms):                           263.97
P99 TTFT (ms):                           273.89
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.90
Median TPOT (ms):                        22.73
P50 TPOT (ms):                           22.73
P95 TPOT (ms):                           23.85
P99 TPOT (ms):                           23.90
---------------Inter-token Latency----------------
Mean ITL (ms):                           36.05
Median ITL (ms):                         35.92
P50 ITL (ms):                            35.92
P95 ITL (ms):                            38.30
P99 ITL (ms):                            39.56
----------------End-to-end Latency----------------
Mean E2EL (ms):                          11374.68
Median E2EL (ms):                        11778.79
P50 E2EL (ms):                           11778.79
P95 E2EL (ms):                           12348.62
P99 E2EL (ms):                           12375.85
---------------Speculative Decoding---------------
Acceptance rate (%):                     21.61
Acceptance length:                       1.65
Drafts:                                  2483
Draft tokens:                            7449
Accepted tokens:                         1610
Per-position acceptance (%):
  Position 0:                            60.77
  Position 1:                            3.66
  Position 2:                            0.40
==================================================
```

---

## 6. thinking=false openai-chat 128/128, 8 prompts, max-concurrency 2

Vera PASS on the result and on thinking=false (`Namespace` extra_body thinking False). Not user-visible decode.

```bash
./bench.sh 6
```

```
============ Serving Benchmark Result ============
Successful requests:                     8
Failed requests:                         0
Maximum request concurrency:             2
Benchmark duration (s):                  11.36
Total input tokens:                      1123
Total generated tokens:                  1024
Request throughput (req/s):              0.70
Output token throughput (tok/s):         90.16
Peak output token throughput (tok/s):    58.00
Peak concurrent requests:                4.00
Total token throughput (tok/s):          189.03
---------------Time to First Token----------------
Mean TTFT (ms):                          166.35
Median TTFT (ms):                        155.51
P50 TTFT (ms):                           155.51
P95 TTFT (ms):                           201.42
P99 TTFT (ms):                           203.75
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.04
Median TPOT (ms):                        21.00
P50 TPOT (ms):                           21.00
P95 TPOT (ms):                           22.48
P99 TPOT (ms):                           22.67
---------------Inter-token Latency----------------
Mean ITL (ms):                           34.71
Median ITL (ms):                         34.89
P50 ITL (ms):                            34.89
P95 ITL (ms):                            36.91
P99 ITL (ms):                            38.32
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2838.99
Median E2EL (ms):                        2822.33
P50 E2EL (ms):                           2822.33
P95 E2EL (ms):                           3044.17
P99 E2EL (ms):                           3073.93
---------------Speculative Decoding---------------
Acceptance rate (%):                     22.40
Acceptance length:                       1.67
Drafts:                                  610
Draft tokens:                            1830
Accepted tokens:                         410
Per-position acceptance (%):
  Position 0:                            58.85
  Position 1:                            8.20
  Position 2:                            0.16
==================================================
```

---

## Unified memory after the set

`free -h` on spark1 (not nvidia-smi):

```
               total        used        free      shared  buff/cache   available
Mem:           121Gi        59Gi        36Gi       106Mi        26Gi        61Gi
Swap:           15Gi       1.7Gi        14Gi
```

`vllm:prefix_cache_hits_total` unchecked.

## Replicate checklist

1. Same image digest, same flags, TP=1, port 8000, `gpu-memory-utilization 0.4`, `max-num-seqs 4`, MTP 3 / triton, Marlin MoE, FlashInfer, FP8 KV, fastsafetensors.
2. Chat benches must send `thinking=false` and `enable_thinking=false`. Completions benches do not use the chat template.
3. Copy max-concurrency exactly.
4. One bench at a time.
5. Measure memory with `free -h`. ~59Gi used / 121Gi total after load is the check.
6. Expect the Marlin weight-only FP4 warning.
