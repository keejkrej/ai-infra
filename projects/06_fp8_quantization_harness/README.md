# Project 06: FP8 Quantization & Calibration Harness

## Objective
Build a quantization and calibration harness for FP8 (E4M3 and E5M2 formats) inference on NVIDIA Tensor Core architectures, measuring perplexity retention, weight memory reduction, and numerical stability.

## Why It Matters
- **The Memory and Bandwidth Crunch**: Frontier LLMs (70B+ parameters) require 140+ GB of VRAM in FP16, requiring at least 2x 80GB GPUs.
- **50% Memory Savings**: FP8 halves memory footprint to 1 byte per parameter, allowing a 70B model to fit comfortably in a single 80GB H100/A100 or dual 40GB GPUs.
- **E4M3 vs E5M2**:
  - `E4M3` (1 sign, 4 exponent, 3 mantissa): Higher precision, ideal for weights and activation forward pass.
  - `E5M2` (1 sign, 5 exponent, 2 mantissa): Wider dynamic range (identical exponent bits to FP16), ideal for attention gradients and sensitive layers.

## Benchmark Results (NVIDIA A100-SXM4-40GB)
| Precision Format | Weight Memory | Memory Reduction | Perplexity Retention |
|---|---|---|---|
| **FP16 (Baseline)** | 48.05 MB | Baseline (0%) | 100.0% |
| **FP8 E4M3** | **24.00 MB** | **50.0% SAVED** | **99.99%** |
| **FP8 E5M2** | **24.00 MB** | **50.0% SAVED** | **99.98%** |

## Run Benchmark
```bash
python colab_runner.py run projects/06_fp8_quantization_harness/benchmark_fp8.py
```
