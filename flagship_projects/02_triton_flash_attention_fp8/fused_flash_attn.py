"""
Flagship Project 02: Custom Fused FlashAttention-2 Kernel in Triton
Implements FlashAttention-2 online softmax scaling in OpenAI Triton,
eliminating intermediate N x N attention matrix writes to High Bandwidth Memory (HBM).
"""

import math
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Pointers
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize online softmax statistics: max m_i and sum l_i
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Loop over K, V blocks
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # Load K block
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        
        # Q * K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Causal masking
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Rescale previous accumulation
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V block
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        # Update running max
        m_i = m_ij

        # Advance pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # Normalize by sum(exp)
    acc = acc / l_i[:, None]

    # Store output
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc.to(Q.dtype), mask=offs_m[:, None] < N_CTX)

def triton_flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float = None) -> torch.Tensor:
    """
    q, k, v: [Batch, Heads, SeqLen, HeadDim]
    """
    Z, H, N_CTX, D_HEAD = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_HEAD)

    out = torch.empty_like(q)
    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        L, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D_HEAD,
        BLOCK_N=BLOCK_N,
    )
    return out
