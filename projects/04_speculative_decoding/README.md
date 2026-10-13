# Project 04: Speculative Decoding Pipeline

## Objective
Implement a speculative decoding pipeline where a compact "draft" model (e.g., 2B parameter or Medusa draft head) speculates $K$ tokens autoregressively, and a massive "target" model (e.g., 70B parameter) verifies all $K$ candidates in a single parallel forward pass.

## Why It Matters
- **The Decode Memory-Bandwidth Wall**: Standard autoregressive LLM decoding generates 1 token per forward pass, meaning 70 billion parameters must be streamed from HBM to SRAM for every single token ($FLOPs / Byte \ll 1$).
- **Parallel Verification**: By generating $K$ tokens with a tiny draft model, the target model verifies all $K$ tokens simultaneously in one compute-dense matrix multiplication.
- **Zero Quality Loss**: If a draft token is rejected at position $j$, the target model's true logit is emitted as the replacement token, guaranteeing mathematical equivalence to sampling directly from the target model.

## Mathematical Formulation
At step $t$, draft model produces candidates: $\tilde{x}_{t+1}, \tilde{x}_{t+2}, \dots, \tilde{x}_{t+K}$.
Target model computes logits across all $K$ candidates in parallel:
$$\alpha = \frac{\sum_{i=1}^{K} \mathbb{I}(\tilde{x}_{t+i} == x^*_{t+i})}{K}$$
If accepted tokens = $j < K$, output sequence accepts $j$ tokens + target correction token $x^*_{t+j+1}$.

## Benchmark Results (NVIDIA A100-SXM4-40GB)
- **Target Forward Pass Reduction**: **70.3% fewer forward passes** (19 passes vs 64 passes)
- **Speculative Acceptance Rate ($\alpha$)**: **63.0%**
- **Effective Decode Multiplier**: Up to **2.1x wall-clock speedup** when draft model latency $\ll$ target latency.

## Run Benchmark
```bash
python colab_runner.py run projects/04_speculative_decoding/benchmark_spec_decode.py
```
