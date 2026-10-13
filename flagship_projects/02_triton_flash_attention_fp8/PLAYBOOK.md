# Flagship 02: Custom Triton FlashAttention & FP8 Suite Playbook

## Objective
Author, benchmark, and profile high-performance custom kernels in OpenAI Triton (**FlashAttention-2** causal attention and **Fused FP8 GEMM+SwiGLU**), proving low-level CUDA, SRAM tiling, and Tensor Core mastery.

---

## Compute Budget & Sizing
- **Recommended Hardware**: NVIDIA A100-SXM4-40GB (or H100)
- **Expected Runtime**: 1–2 hours for full context sweep (512 to 16,384 tokens) and NCU profiling
- **Colab Credit Burn**: ~5–15 compute units

---

## Technical Concepts Mastered
1. **Online Softmax Scaling**: Computing row-wise max $m_i$ and running sum $l_i$ incrementally during tiling, eliminating intermediate $O(N^2)$ memory writes to High Bandwidth Memory (HBM).
2. **SRAM Tiling**: Block sizes ($B_r = 64, B_c = 64$) fitted into the 192 KB on-chip Shared Memory / L1 cache per Streaming Multiprocessor (SM).
3. **Roofline Analysis**: Visualizing compute vs memory bandwidth ceilings using NVIDIA Nsight Compute (`ncu`).

---

## Execution Guide

### Step 1: Run the Micro-Benchmark Suite
```bash
# Run on Colab A100 GPU
python colab_runner.py run flagship_projects/02_triton_flash_attention_fp8/profile_kernels.py
```

### Step 2: Capture Nsight Compute (`ncu`) Trace
```bash
# Profile warp occupancy and memory bandwidth saturation
ncu --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python flagship_projects/02_triton_flash_attention_fp8/profile_kernels.py
```

### Step 3: What to Include in Your Portfolio Writeup
- A side-by-side table comparing execution time (ms) and TFLOPs/s across sequence lengths (512 to 8,192).
- The roofline graph screenshot from Nsight Compute demonstrating that FlashAttention-2 shifts attention from memory-bound ($FLOP/Byte < 1$) to compute-dense ($FLOP/Byte > 50$).
