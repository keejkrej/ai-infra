# Project 09: Multi-Node Tensor Parallelism & NCCL Interconnect Profiler

## Objective
Profile and mathematically analyze the network communication bottleneck in distributed Tensor Parallelism (TP) across varying GPU interconnect fabrics (NVLink 4, NVLink 3, PCIe Gen4, InfiniBand NDR 800G, RoCE v2 100G, and 10G Ethernet).

## Why It Matters
- **The 160 AllReduce Barrier**: In Megatron-style Tensor Parallelism, each transformer layer requires two `AllReduce` collectives (one after Multi-Head Attention, one after MLP). A 70B parameter model with 80 layers executes **160 AllReduce collectives for every single decode token generated**.
- **The Interconnect Cliff**: Spanning Tensor Parallelism across physical nodes without NVLink or ultra-low latency InfiniBand collapses generation speed. If inter-node latency is 15 µs (typical 100G RoCE v2), communication overhead reaches **97.9%**, starving GPU Tensor Cores.

## Ring AllReduce Analytic Formula
$$\text{Time}_{\text{AllReduce}} = 2 \cdot \left(\frac{N-1}{N}\right) \cdot \frac{S}{\text{Bandwidth}} + 2 \cdot (N-1) \cdot \alpha$$
Where $N$ is TP world size, $S$ is message tensor payload $[B, d_{\text{model}}] \times \text{bytes}$, and $\alpha$ is network latency per ring hop.

## Key Takeaway for AI Infra Architecture
> **Golden Rule**: Confine Tensor Parallelism (TP) strictly within a single node (up to 8 GPUs) backed by high-speed NVLink (300-900 GB/s). Cross-node scaling should exclusively use **Pipeline Parallelism (PP)**, **Expert Parallelism (EP)**, or **Data/Context Parallelism**, which require orders of magnitude less frequent synchronization.

## Run Profiler
```bash
python colab_runner.py run projects/09_tensor_parallel_nccl_profiler/nccl_interconnect_profiler.py
```
