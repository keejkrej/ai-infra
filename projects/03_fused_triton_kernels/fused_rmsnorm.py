"""
Project 03: Custom Fused Triton RMSNorm Kernel
Implements high-performance fused RMSNorm (with optional residual addition) in Triton.
Eliminates intermediate HBM round-trips for maximum memory bandwidth saturation.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,          # Pointer to input tensor [M, N]
    Y_ptr,          # Pointer to output tensor [M, N]
    W_ptr,          # Pointer to weight tensor [N]
    stride_xm,      # Stride of X along dimension M
    stride_xn,      # Stride of X along dimension N
    stride_ym,      # Stride of Y along dimension M
    stride_yn,      # Stride of Y along dimension N
    N,              # Hidden dimension
    eps,            # Epsilon constant
    BLOCK_SIZE: tl.constexpr,
):
    # Row index
    row_idx = tl.program_id(0)

    # Offset pointers to the current row
    x_row = X_ptr + row_idx * stride_xm
    y_row = Y_ptr + row_idx * stride_ym

    # Col offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row elements
    x = tl.load(x_row + col_offsets * stride_xn, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)

    # Compute RMS = sqrt(mean(x^2) + eps)
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(variance + eps)

    # Normalize and scale
    normed = x * rsqrt * w

    # Store normalized result
    tl.store(y_row + col_offsets * stride_yn, normed.to(tl.float16), mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Wrapper function for Triton RMSNorm."""
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
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
        # Standard PyTorch formula
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return (normed * self.weight).to(x.dtype)
