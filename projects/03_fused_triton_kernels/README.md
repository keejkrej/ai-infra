# Project 03: Custom Fused Triton Kernels (RMSNorm & RoPE)

## Objective
Write high-performance fused kernels in OpenAI Triton for Root Mean Square Normalization (RMSNorm) and Rotary Position Embeddings (RoPE), benchmarking them against PyTorch native operations on NVIDIA Tensor Core GPUs (A100-SXM4-40GB).

## Why It Matters
- **The Memory-Bound Bottleneck**: In transformer inference, point-wise operations like RMSNorm and RoPE have very low arithmetic intensity ($FLOPs / Byte < 1$).
- **The PyTorch Multi-Kernel Overhead**: Standard PyTorch RMSNorm (`x * torch.rsqrt(x.pow(2).mean(-1) + eps) * w`) executes 4 distinct CUDA kernels and makes 4 round-trips to high-bandwidth memory (HBM).
- **Triton Kernel Fusion**: Loads data into on-chip SRAM once, performs accumulation, reciprocal square root, multiplication, and writes back directly, achieving over 1.1 TB/s memory throughput.

## Benchmark Results (NVIDIA A100-SXM4-40GB)
| Kernel | Hidden Dim / Shape | PyTorch Latency | Triton Fused Latency | Speedup | Memory Bandwidth |
|---|---|---|---|---|---|
| **RMSNorm** | 2048 | 335.24 µs | **32.46 µs** | **10.33x** | **962.65 GB/s** |
| **RMSNorm** | 4096 | 579.33 µs | **57.31 µs** | **10.11x** | **1090.55 GB/s** |
| **RMSNorm** | 8192 | 1103.86 µs | **106.88 µs** | **10.33x** | **1169.55 GB/s** |
| **RoPE** | [4, 1024, 32, 128] | 306.40 µs | **104.28 µs** | **2.94x** | **599.35 GB/s** |

## Run Benchmark
```bash
python colab_runner.py run projects/03_fused_triton_kernels/benchmark_triton.py
```
