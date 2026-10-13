# Flagship AI Infrastructure Portfolio Projects

A collection of **four high-impact, flagship portfolio projects** designed specifically to burn Google Colab A100/H100 compute credits into tangible, career-defining proof of silicon and distributed systems mastery.

---

## Flagship Project Matrix

| Flagship Project | Directory | Core Technical Focus | Estimated Credits | Artifacts Generated |
|---|---|---|---|---|
| **01. Frontier Inference Benchmark Teardown** | [`01_frontier_inference_teardown/`](file:///Users/jack/workspace/ai-infra/flagship_projects/01_frontier_inference_teardown) | Comparative benchmarking of vLLM vs SGLang vs TensorRT-LLM up to 256 concurrency. | **15–30 credits** | Interactive Pareto HTML curves (`pareto_curve.html`), Cost vs TTFT analysis. |
| **02. Custom Triton FlashAttention & FP8** | [`02_triton_flash_attention_fp8/`](file:///Users/jack/workspace/ai-infra/flagship_projects/02_triton_flash_attention_fp8) | Low-level OpenAI Triton kernel engineering (FlashAttention-2 causal attention & fused FP8). | **10–20 credits** | TFLOPs/s roofline curves, Nsight Compute (`ncu`) reports. |
| **03. Disaggregated Prefill/Decode Fabric** | [`03_disaggregated_prefill_decode_fabric/`](file:///Users/jack/workspace/ai-infra/flagship_projects/03_disaggregated_prefill_decode_fabric) | Decoupled P-worker and D-worker cluster with asynchronous KV-cache streaming fabric. | **20–40 credits** | Cluster stress test logs, TTFT vs ITL isolation proof. |
| **04. Large-Scale Pre-Training & MFU Run** | [`04_pretraining_mfu_optimization_run/`](file:///Users/jack/workspace/ai-infra/flagship_projects/04_pretraining_mfu_optimization_run) | End-to-end 1B Transformer pretraining with PyTorch 2.0 `torch.compile` & MFU optimization. | **50–250 credits** | Chrome trace JSON (`pretraining_trace.json`) for Perfetto, MFU efficiency progression. |

---

## How to Execute in a Separate Session

You can run each project directly in a separate dedicated Colab session whenever you are ready:

### 1. Provision Your Colab Session
```bash
# Provision a fresh high-RAM A100 GPU session
colab new -s flagship-session --gpu A100 --high-mem
```

### 2. Run the Flagship Harness
Use the included `colab_runner.py` utility:

```bash
# Run Flagship 01: Inference Teardown
python colab_runner.py run flagship_projects/01_frontier_inference_teardown/run_benchmarks.py --session flagship-session

# Run Flagship 02: Custom Triton FlashAttention
python colab_runner.py run flagship_projects/02_triton_flash_attention_fp8/profile_kernels.py --session flagship-session

# Run Flagship 03: Disaggregated Serving Fabric
python colab_runner.py run flagship_projects/03_disaggregated_prefill_decode_fabric/cluster_coordinator.py --session flagship-session

# Run Flagship 04: Pre-Training & MFU Optimization
python colab_runner.py run flagship_projects/04_pretraining_mfu_optimization_run/train_fsdp2_mfu.py --session flagship-session
```

---

## Playbooks Included
Each flagship directory contains a dedicated **`PLAYBOOK.md`** file outlining exact step-by-step commands, hardware math, and publication blueprints (Twitter threads, technical blog post templates, and diagram generators).
