# Project 08: Complexity-Routing AI Gateway

## Objective
Build an AI reverse proxy gateway that inspects prompt complexity (Shannon token entropy, reasoning syntax, code blocks, token length) to dynamically route simple tasks to local quantized 8B models (\$0.08/1M tokens) and complex tasks to frontier reasoning models (\$5.00/1M tokens).

## Why It Matters
- **Unit Economics**: Blended API gross margins collapse if 100% of user traffic is sent to expensive frontier models. Over 70-80% of everyday queries (translations, simple extractions, chat banter) can be served with equal satisfaction by a quantized local model.
- **Shannon Token Entropy**: Measures the informational surprise and compressibility of prompt text. Technical code and mathematical proofs exhibit high entropy and reasoning keyword density.

## Scoring Equation
$$\text{Complexity} = 0.30 \cdot H(X) + 0.45 \cdot \text{ReasoningDensity} + 0.25 \cdot \text{LengthFactor}$$
If $\text{Complexity} \ge 0.45 \implies \text{Route to Frontier Reasoning API}$, else $\implies \text{Local Quantized 8B Replica}$.

## Benchmark Results
- **Queries Evaluated**: Mixed real-world distribution
- **Routed to Local Model**: **90%**
- **Routed to Frontier API**: **10%**
- **Net Margin Protection**: **83.9% Cloud API Cost Reduction**

## Run Benchmark
```bash
python colab_runner.py run projects/08_complexity_routing_gateway/complexity_router.py
```
