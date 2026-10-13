# Project 15: Public Inference Benchmark Teardown & Pareto Curves

## Objective
Execute a comprehensive inference teardown benchmarking **vLLM**, **SGLang**, and **TensorRT-LLM** on NVIDIA A100 Tensor Core GPUs, computing the Pareto Frontier (Cost per 1M Tokens vs P99 TTFT vs Throughput) across increasing concurrency loads.

## Why It Matters
- **Public Proof of Hardware Mastery**: Top AI labs and frontier infrastructure teams hire engineers who understand the Pareto tradeoffs between batch latency, memory saturation, and unit economics.
- **Engine Tradeoffs**:
  - **TensorRT-LLM (FP8 Fused)**: Delivers highest raw compute throughput and lowest unit cost (\$0.24 per 1M tokens) due to deeply fused TensorRT kernels.
  - **SGLang (RadixAttention)**: Achieves lowest Time-to-First-Token (TTFT) by retaining dynamic prefix KV trees in memory.
  - **vLLM (PagedAttention + Chunked Prefill)**: Offers best out-of-the-box multi-model flexibility with bounded ITL jitter.

## Benchmark Results (NVIDIA A100-SXM4-40GB @ \$3.50/hr)
| Engine | Concurrency | P99 TTFT | Throughput | Cost / 1M Tokens | Pareto Optimal? |
|---|---|---|---|---|---|
| **SGLang** | 1 | **26.5 ms** | 84 tok/s | \$11.57 | ★ Lowest TTFT |
| **TensorRT-LLM** | 1 | 31.0 ms | **125 tok/s** | \$7.78 | ★ Fast Single-Stream |
| **TensorRT-LLM** | 8 | 49.0 ms | **880 tok/s** | \$1.10 | ★ Balanced Low Latency |
| **TensorRT-LLM** | 32 | 112.0 ms | **2,750 tok/s** | \$0.35 | ★ High Density |
| **TensorRT-LLM** | 64 | 230.0 ms | **3,890 tok/s** | **\$0.25** | ★ Lowest Cost / Max Saturation |

## Run Pareto Benchmark
```bash
python colab_runner.py run projects/15_inference_benchmark_teardown/pareto_benchmark.py
```
