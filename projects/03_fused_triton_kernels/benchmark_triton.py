"""
Project 03: Triton Kernel Benchmarking Suite
Microbenchmarks Custom Fused Triton kernels (RMSNorm & RoPE) against Native PyTorch
on NVIDIA GPU hardware.
"""

import math
import torch
import triton
import triton.language as tl
import triton.testing

# =====================================================================
# 1. FUSED RMSNORM KERNEL
# =====================================================================
@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row = X_ptr + row_idx * stride_xm
    y_row = Y_ptr + row_idx * stride_ym

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_row + col_offsets * stride_xn, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)

    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(variance + eps)

    normed = x * rsqrt * w
    tl.store(y_row + col_offsets * stride_yn, normed.to(tl.float16), mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](
        x, y, weight,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

class PyTorchRMSNorm(torch.nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return (normed * self.weight).to(x.dtype)

# =====================================================================
# 2. FUSED RoPE KERNEL
# =====================================================================
@triton.jit
def _rope_fwd_kernel(
    Q_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_q_bs, stride_q_h, stride_q_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_out_bs, stride_out_h, stride_out_d,
    seq_len,
    D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    s_idx = token_idx % seq_len

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D_HALF

    q_first_half_ptr = Q_ptr + token_idx * stride_q_bs + head_idx * stride_q_h + cols * stride_q_d
    q_second_half_ptr = q_first_half_ptr + D_HALF * stride_q_d

    cos_ptrs = Cos_ptr + s_idx * stride_cos_s + cols * stride_cos_d
    sin_ptrs = Sin_ptr + s_idx * stride_sin_s + cols * stride_sin_d

    q1 = tl.load(q_first_half_ptr, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_second_half_ptr, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptrs, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptrs, mask=mask, other=0.0).to(tl.float32)

    out1 = q1 * cos - q2 * sin
    out2 = q1 * sin + q2 * cos

    out_first_half = Out_ptr + token_idx * stride_out_bs + head_idx * stride_out_h + cols * stride_out_d
    out_second_half = out_first_half + D_HALF * stride_out_d

    tl.store(out_first_half, out1.to(tl.float16), mask=mask)
    tl.store(out_second_half, out2.to(tl.float16), mask=mask)

def triton_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, S, H, D = q.shape
    D_HALF = D // 2
    q_flat = q.contiguous().view(B * S, H, D)
    out_flat = torch.empty_like(q_flat)

    BLOCK_SIZE = triton.next_power_of_2(D_HALF)
    grid = (B * S, H)

    _rope_fwd_kernel[grid](
        q_flat, cos, sin, out_flat,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        S,
        D_HALF=D_HALF,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out_flat.view(B, S, H, D)

def pytorch_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, S, H, D = q.shape
    d_half = D // 2
    q1 = q[..., :d_half]
    q2 = q[..., d_half:]
    cos = cos.view(1, S, 1, d_half)
    sin = sin.view(1, S, 1, d_half)
    out1 = q1 * cos - q2 * sin
    out2 = q1 * sin + q2 * cos
    return torch.cat([out1, out2], dim=-1)

# =====================================================================
# BENCHMARK HARNESS
# =====================================================================
def benchmark_rmsnorm():
    print("\n--- [ 1. BENCHMARKING FUSED RMSNORM ] ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for Triton benchmarks.")
        return

    hidden_dims = [2048, 4096, 8192]
    batch_tokens = 4096  # e.g., Batch Size 4 x 1024 seq_len

    print(f"{'Hidden Dim':<12} | {'PyTorch (µs)':<14} | {'Triton (µs)':<14} | {'Speedup':<10} | {'Bandwidth (GB/s)':<18}")
    print("-" * 75)

    for hidden_dim in hidden_dims:
        x = torch.randn(batch_tokens, hidden_dim, device=device, dtype=torch.float16)
        weight = torch.ones(hidden_dim, device=device, dtype=torch.float16)
        pt_norm = PyTorchRMSNorm(hidden_dim).to(device=device, dtype=torch.float16)

        # 1. Correctness check
        y_triton = triton_rmsnorm(x, weight)
        y_pt = pt_norm(x)
        assert torch.allclose(y_triton, y_pt, atol=1e-2, rtol=1e-2), "Numerical mismatch detected!"

        # 2. Timing
        ms_pt = triton.testing.do_bench(lambda: pt_norm(x))
        ms_triton = triton.testing.do_bench(lambda: triton_rmsnorm(x, weight))

        us_pt = ms_pt * 1000.0
        us_triton = ms_triton * 1000.0
        speedup = us_pt / us_triton

        # Bandwidth: 2 bytes per float16 element. Read X + W, Write Y = 2 * M * N * 2 bytes
        total_bytes = batch_tokens * hidden_dim * 2 * 2
        bandwidth_gb_s = (total_bytes / (ms_triton / 1000.0)) / (1024**3)

        print(f"{hidden_dim:<12} | {us_pt:<14.2f} | {us_triton:<14.2f} | {speedup:<10.2f}x | {bandwidth_gb_s:<18.2f}")

def benchmark_rope():
    print("\n--- [ 2. BENCHMARKING FUSED RoPE ] ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return

    B, S, H, D = 4, 1024, 32, 128
    D_HALF = D // 2

    q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    cos = torch.randn(S, D_HALF, device=device, dtype=torch.float16)
    sin = torch.randn(S, D_HALF, device=device, dtype=torch.float16)

    # 1. Correctness check
    out_triton = triton_rope(q, cos, sin)
    out_pt = pytorch_rope(q, cos, sin)
    assert torch.allclose(out_triton, out_pt, atol=1e-2, rtol=1e-2), "RoPE Numerical mismatch!"

    # 2. Timing
    ms_pt = triton.testing.do_bench(lambda: pytorch_rope(q, cos, sin))
    ms_triton = triton.testing.do_bench(lambda: triton_rope(q, cos, sin))

    us_pt = ms_pt * 1000.0
    us_triton = ms_triton * 1000.0
    speedup = us_pt / us_triton

    total_bytes = (B * S * H * D * 2 * 2)  # Read + Write
    bandwidth_gb_s = (total_bytes / (ms_triton / 1000.0)) / (1024**3)

    print(f"Shape: [Batch={B}, SeqLen={S}, Heads={H}, HeadDim={D}]")
    print(f"PyTorch RoPE : {us_pt:.2f} µs")
    print(f"Triton RoPE  : {us_triton:.2f} µs (Speedup: {speedup:.2f}x, Bandwidth: {bandwidth_gb_s:.2f} GB/s)")

def main():
    print("=================================================================")
    print(" PROJECT 03: CUSTOM FUSED TRITON KERNELS BENCHMARK")
    print(f" Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("=================================================================")
    benchmark_rmsnorm()
    benchmark_rope()
    print("\n[AI Infra Insight]: Fused Triton kernels bypass PyTorch operator launch")
    print("overhead and achieve extreme HBM memory bandwidth efficiency!\n")

if __name__ == "__main__":
    main()
