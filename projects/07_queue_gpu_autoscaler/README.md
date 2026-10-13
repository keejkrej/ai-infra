# Project 07: Queue-Based GPU Autoscaler (KEDA)

## Objective
Implement a token-depth-aware autoscaler (Kubernetes Event-driven Autoscaling - KEDA pattern) that scales GPU instances based on in-flight token volume rather than raw HTTP requests, incorporating graceful draining for spot instances.

## Why It Matters
- **The LLM Scaling Flaw**: Standard Kubernetes HPA scales on CPU% or HTTP request count. However, a single 16,000-token prompt requires 160x more VRAM and compute than ten 10-token prompts. Scaling on request count results in catastrophic VRAM Out-Of-Memory (OOM) errors.
- **FinOps Optimization**: Modern GPU nodes (e.g. H100 / A100) cost \$2.50 to \$4.00+ per hour. Static peak provisioning wastes tens of thousands of dollars on idle silicon.
- **Graceful Draining**: Spot GPUs require draining active autoregressive decodes before releasing the instance, preventing dropped user connections.

## Autoscaler Mathematical Policy
$$\text{Desired Replicas} = \min\left(N_{\text{max}}, \max\left(N_{\text{min}}, \left\lceil \frac{\sum_{r \in \text{Queue}} (\text{prompt}_r + \text{est\_decode}_r)}{\text{Target Tokens Per GPU}} \right\rceil\right)\right)$$

## Benchmark Results
- **Peak Static Cost**: \$7.35 (2.10 GPU-hours)
- **Token-Queue Autoscaled Cost**: \$3.85 (1.10 GPU-hours)
- **Cloud Spend Reduction**: **47.6% SAVED**

## Run Benchmark
```bash
python colab_runner.py run projects/07_queue_gpu_autoscaler/simulate_traffic.py
```
