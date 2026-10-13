"""
Project 03: Custom Fused Triton RoPE (Rotary Position Embedding) Kernel
Fuses rotary coordinate rotation into a single Triton kernel, avoiding intermediate
PyTorch tensor slicing and memory allocations.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _rope_fwd_kernel(
    Q_ptr,              # [B * S, H, D]
    Cos_ptr,            # [S, D // 2]
    Sin_ptr,            # [S, D // 2]
    Out_ptr,            # [B * S, H, D]
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    stride_out_bs,
    stride_out_h,
    stride_out_d,
    seq_len,
    D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs: 0 -> token index (bs), 1 -> head index (h)
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    s_idx = token_idx % seq_len  # Position in sequence

    # Offset within half-dimension
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D_HALF

    # Pointer offsets for first half and second half of embedding dimension D
    q_first_half_ptr = Q_ptr + token_idx * stride_q_bs + head_idx * stride_q_h + cols * stride_q_d
    q_second_half_ptr = q_first_half_ptr + D_HALF * stride_q_d

    cos_ptrs = Cos_ptr + s_idx * stride_cos_s + cols * stride_cos_d
    sin_ptrs = Sin_ptr + s_idx * stride_sin_s + cols * stride_sin_d

    # Load vectors
    q1 = tl.load(q_first_half_ptr, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_second_half_ptr, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptrs, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Complex rotation:
    # out1 = q1 * cos - q2 * sin
    # out2 = q1 * sin + q2 * cos
    out1 = q1 * cos - q2 * sin
    out2 = q1 * sin + q2 * cos

    out_first_half = Out_ptr + token_idx * stride_out_bs + head_idx * stride_out_h + cols * stride_out_d
    out_second_half = out_first_half + D_HALF * stride_out_d

    tl.store(out_first_half, out1.to(tl.float16), mask=mask)
    tl.store(out_second_half, out2.to(tl.float16), mask=mask)

def triton_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    q: [batch_size, seq_len, num_heads, head_dim]
    cos, sin: [seq_len, head_dim // 2]
    """
    B, S, H, D = q.shape
    D_HALF = D // 2
    q_flat = q.view(B * S, H, D)
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
    """Reference PyTorch implementation using slicing and cat."""
    B, S, H, D = q.shape
    d_half = D // 2
    q1 = q[..., :d_half]
    q2 = q[..., d_half:]
    cos = cos.view(1, S, 1, d_half)
    sin = sin.view(1, S, 1, d_half)
    
    out1 = q1 * cos - q2 * sin
    out2 = q1 * sin + q2 * cos
    return torch.cat([out1, out2], dim=-1)
