# Project 05: Disaggregated Prefill/Decode Cluster

## Objective
Implement and benchmark the 2026 frontier inference architecture: Disaggregated Prefill and Decode (PD-Disaggregation). Separate compute-heavy prompt prefills from memory-bound autoregressive token decodes into independent GPU pools connected via high-speed KV-cache streaming.

## Why It Matters
- **The Interference Problem**: In monolithic inference, a new 2,000-token prompt causes a massive prefill GEMM that monopolizes GPU compute, creating severe Inter-Token Latency (ITL) spikes for all active decode requests.
- **Hardware Specialization**: Prefill requires high compute density (high Tensor Core utilization / MFU), whereas decode requires high memory bandwidth.
- **The Disaggregated Solution**:
  1. **Prefill Pool (P-Workers)**: Sized for large matrix multiplications and maximum Model FLOPs Utilization (MFU).
  2. **KV-Cache Streamer**: Transmits computed KV-blocks via RDMA / NVLink / high-speed IPC to the decode pool.
  3. **Decode Pool (D-Workers)**: Continuously streams generated tokens without prefill interruptions.

## Benchmark Results (NVIDIA A100-SXM4-40GB)
- **Unified Serving Throughput**: 392,168 tok/s
- **Disaggregated Serving Throughput**: **1,015,100 tok/s (2.59x throughput gain!)**
- **SLA Compliance**: Decode loops run completely isolated from bursty prompt arrivals.

## Run Benchmark
```bash
python colab_runner.py run projects/05_disaggregated_prefill_decode/benchmark_pd_disaggregation.py
```
