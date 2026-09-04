# AGENTS.md — Ornith-1.5-35B-A3B-NVFP4 · 1× DGX Spark

Serve `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` on one Spark, TP=1. Image digest `sha256:0a51ea5b…` (`vllm/vllm-openai:v0.27.1`). Snapshot `9660379`. Container `ornith-1.5-35b-nvfp4-official`. `HF_TOKEN` is required. `./run.sh` is the live docker command. `./bench.sh` is the measured `vllm bench serve` set.

Humans read [README.md](README.md). This repo has no `bench_decode.py` and no unit tests.

## Working rules

- One-node recipe. No worker SSH, no NCCL pin, no `recipe.yaml`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin the digest. Do not float `:latest`.
- Keep `--gpu-memory-utilization 0.4`, `--max-model-len 262144`, Marlin MoE, FlashInfer, FP8 KV, MTP-3.
- `vllm` is not on the Spark host PATH. Bench inside the live container, or `./bench.sh`.
- One bench at a time. Do not merge tables. Do not treat any one tok/s as a general decode number.
- Label max-concurrency (the flag you passed). The harness also prints peak concurrent. That field was 2× the cap on every dump here. Do not call a max-concurrency=1 run single-stream.
- Thinking off is required for short generate and for chat benches. Default chat with thinking on eats the token budget (`content` null, `finish_reason=length`). Both `thinking:false` and `enable_thinking:false` are required. `max_tokens:64` without temperature can return null content.
- GB10 load warnings (no native FP4, q-scale missing, not enough SMs for max_autotune_gemm, first-request Triton JIT) are expected. Do not abort on them.
- Do not claim a KV hit rate from this set. `vllm:prefix_cache_hits_total` is unchecked.

## Verify

```bash
HF_TOKEN=… ./run.sh
```

Ready-gate: `/health` 200, `/v1/models` lists the model, then the README `pong` chat with both thinking flags off. Expected `content` = `pong`, `reasoning` null, `finish_reason=stop`, `system_fingerprint` starts `vllm-0.27.1-`.

```bash
./bench.sh          # default table 5
./bench.sh 5
./bench.sh all
```

Do not invent a 2× Spark prose/structured table for this recipe.

## Never touch

- Live HF tokens
- Recreating the container without `./stop.sh` first (`run.sh` exits if the name already exists)
