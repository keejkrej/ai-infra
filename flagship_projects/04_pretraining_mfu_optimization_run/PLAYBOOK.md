# Flagship 04: Pre-Training & MFU Optimization Run Playbook

## Objective
Train a 1B-scale Transformer model on curated text/code tokens while monitoring and optimizing **Model FLOPs Utilization (MFU)** to achieve state-of-the-art training efficiency (>45–50% MFU on NVIDIA A100 Tensor Cores).

---

## Compute Budget & Sizing
- **Recommended Hardware**: NVIDIA A100-SXM4-40GB (High-RAM shape)
- **Training Durations**:
  - **Quick Profiling Run (15–30 mins)**: ~2–5 Colab credits (validates MFU and exports Chrome traces).
  - **Medium Run (6–12 hours)**: ~30–65 Colab credits.
  - **Full Over-Night Pretraining Run (24–48 hours)**: ~125–250 Colab credits (trains on hundreds of millions of tokens).

---

## The Physics of MFU (Model FLOPs Utilization)
$$MFU = \frac{6 \times P \times B \times S}{\Delta t \times \text{Peak TFLOPs}}$$
Where:
- $P$: Number of model parameters (excluding non-active embeddings).
- $B \times S$: Total tokens per step (batch size $\times$ sequence length).
- $\Delta t$: Wall-clock step time in seconds.
- $\text{Peak TFLOPs}$: Hardware limit (312 TFLOPs for FP16 on A100).

---

## Execution Guide

### Step 1: Run Baseline Pretraining & MFU Measurement
```bash
# Run 25 training steps and monitor MFU
python colab_runner.py run flagship_projects/04_pretraining_mfu_optimization_run/train_fsdp2_mfu.py
```

### Step 2: Enable PyTorch 2.0 Graph Compilation
```bash
# Compiles kernel graphs via torch.compile(mode="reduce-overhead")
python colab_runner.py run flagship_projects/04_pretraining_mfu_optimization_run/train_fsdp2_mfu.py --compile
```

### Step 3: Capture Chrome Trace & Perfetto Timeline
```bash
# Exports a detailed trace JSON of CUDA kernel calls and host launch overhead
python colab_runner.py run flagship_projects/04_pretraining_mfu_optimization_run/profile_mfu.py
```
Open `pretraining_trace.json` in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev) to see:
- GEMM kernel execution on Tensor Cores
- FlashAttention backward passes
- Gradient clipping and optimizer steps

---

## What to Include in Your Portfolio Writeup
- **MFU Progression Table**: Showing the transition from naive PyTorch (28% MFU) $\to$ `torch.compile` + FlashAttention (48% MFU).
- **Chrome Trace Screenshot**: Annotated timeline showing the elimination of CPU kernel launch bubbles.
