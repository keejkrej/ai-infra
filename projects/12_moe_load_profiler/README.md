# Project 12: MoE (Mixture of Experts) Load Profiler

## Objective
Implement an expert-parallel routing profiler for sparse Mixture of Experts (MoE) architectures (e.g. 8 experts with Top-2 gating), tracking token-to-expert dispatch skew, Coefficient of Variation (CV), routing entropy, and evaluating auxiliary load balancing loss to eliminate straggler latency.

## Why It Matters
- **The 2026 MoE Standard**: Models like Mixtral 8x7B, DeepSeek-V2/V3, and DBRX rely on sparse activation where only a subset of experts activate per token.
- **The Expert Straggler Bottleneck**: In distributed Expert Parallelism (EP), tokens are dispatched across GPUs via synchronous `All-to-All` communication. If router weights collapse and send 40% of tokens to Expert #0 while Expert #7 receives 2%, the GPU hosting Expert #0 becomes an extreme straggler. All other GPUs idle, destroying throughput.
- **Auxiliary Balancing Loss**: Formulates a continuous penalty $\mathcal{L}_{\text{balance}} = N \sum_{i=1}^N f_i \cdot P_i$, encouraging the router to distribute workload uniformly across all available hardware.

## Profiler Metrics
- **Coefficient of Variation (CV)**: $CV = \frac{\sigma}{\mu}$ (Measures deviation from uniform 1/N load)
- **Routing Entropy**: $H(X) = -\sum_{i=1}^N p_i \log_2(p_i)$
- **Peak-to-Trough Ratio**: $\frac{\max(\text{Tokens}_i)}{\min(\text{Tokens}_i)}$

## Benchmark Results (NVIDIA A100-SXM4-40GB)
| Router State | Coefficient of Variation (CV) | Entropy | Peak-to-Trough Ratio |
|---|---|---|---|
| **Imbalanced / Collapsed** | 0.599 | 2.80 bits | 3.1x Imbalance (Straggler Delay) |
| **Balanced (Auxiliary Loss)** | **0.020** | **3.00 / 3.00 bits** | **1.1x (Uniform Parallelism)** |

## Run Profiler
```bash
python colab_runner.py run projects/12_moe_load_profiler/analyze_expert_routing.py
```
