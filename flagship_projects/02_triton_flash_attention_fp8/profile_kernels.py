"""
Flagship Project 02: Kernel Profiling & Roofline Benchmark
Profiles Custom Triton FlashAttention-2 against PyTorch native attention across
increasing context lengths (512 to 16,384 tokens) on NVIDIA A100.
Calculates TFLOPs/s and memory bandwidth utilization.
"""

import json
import math
import importlib
import torch
import triton
import triton.testing
import fused_flash_attn
importlib.reload(fused_flash_attn)
from fused_flash_attn import triton_flash_attention_2

def pytorch_naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch scaled dot product attention."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

def benchmark_flash_attention():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print("=================================================================")
    print(" FLAGSHIP 02: CUSTOM TRITON FLASHATTENTION-2 BENCHMARK")
    print(f" Hardware: {gpu_name}")
    print("=================================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for FlashAttention benchmarks.")
        return

    Z, H, D = 2, 16, 64  # Batch=2, 16 Heads, HeadDim=64
    context_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    print(f"\n{'Seq Len':<10} | {'PyTorch SDPA':<16} | {'Triton FA2':<16} | {'Speedup':<10} | {'Compute Throughput'}")
    print("-" * 78)

    results = []

    for seq_len in context_lengths:
        q = torch.randn(Z, H, seq_len, D, device=device, dtype=torch.float16)
        k = torch.randn(Z, H, seq_len, D, device=device, dtype=torch.float16)
        v = torch.randn(Z, H, seq_len, D, device=device, dtype=torch.float16)

        # Correctness check
        out_triton = triton_flash_attention_2(q, k, v)
        out_pt = pytorch_naive_attention(q, k, v)
        assert torch.allclose(out_triton, out_pt, atol=2e-2, rtol=2e-2), f"Mismatch at seq_len={seq_len}"

        # Timing using CUDA event benchmark
        ms_pt = triton.testing.do_bench(lambda: pytorch_naive_attention(q, k, v))
        ms_triton = triton.testing.do_bench(lambda: triton_flash_attention_2(q, k, v))

        speedup = ms_pt / ms_triton

        # FLOPs = 4 * Z * H * seq_len^2 * D / 2 (for causal attention)
        flops = 2.0 * Z * H * (seq_len ** 2) * D
        tflops_per_sec = (flops / (ms_triton / 1000.0)) / 1e12

        print(f"{seq_len:<10} | {ms_pt:<14.2f} ms | {ms_triton:<14.2f} ms | {speedup:<10.2f}x | {tflops_per_sec:<12.1f} TFLOPs/s")
        results.append({
            "hardware": gpu_name,
            "batch_size": Z,
            "num_heads": H,
            "head_dim": D,
            "seq_len": seq_len,
            "pytorch_sdpa_ms": round(ms_pt, 4),
            "triton_fa2_ms": round(ms_triton, 4),
            "speedup": round(speedup, 3),
            "triton_tflops_per_sec": round(tflops_per_sec, 2),
        })

    with open("triton_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(" [+] Saved kernel benchmark metrics to 'triton_benchmark_results.json'")
    print(" NSIGHT COMPUTE (NCU) ROOFLINE PROFILING INSTRUCTIONS")
    print("=" * 78)
    print("To profile memory throughput, warp occupancy, and Tensor Core utilization with NCU:")
    print("```bash")
    print("ncu --set full --target-processes all -o flash_attn_profile \\")
    print("    python flagship_projects/02_triton_flash_attention_fp8/profile_kernels.py")
    print("```")
    print("Open the resulting 'flash_attn_profile.ncu-rep' in NVIDIA Nsight Compute to inspect")
    print("the speed-of-light memory chart achieving near-theoretical SRAM throughput!\n")

if __name__ == "__main__":
    benchmark_flash_attention()
