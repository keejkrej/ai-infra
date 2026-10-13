# Flagship 03: Disaggregated Prefill/Decode Fabric Playbook

## Objective
Build and stress-test a production-grade Disaggregated Prefill/Decode serving cluster, decoupling compute-bound prompt GEMMs (Prefill nodes) from memory-bandwidth-bound autoregressive token generation (Decode nodes) via asynchronous KV-cache streaming.

---

## Compute Budget & Sizing
- **Recommended Hardware**: 1x or 2x NVIDIA A100 (can be run multi-process on a single SXM4-40GB or across two concurrent Colab sessions)
- **Expected Runtime**: 2–3 hours
- **Colab Credit Burn**: ~15–30 compute units

---

## Why Top AI Labs Use This Architecture
1. **Model FLOPs Utilization (MFU) Mismatch**:
   - Prefill achieves **60–80% MFU** when batching large prompt GEMMs.
   - Decode achieves only **10–25% MFU** because it is memory-bandwidth bound ($FLOP/Byte \approx 1$).
   - Running them together forces the GPU to compromise between high batching and low latency.
2. **Elimination of Decode Jitter**:
   - An incoming 4,000-token prompt in a monolithic engine monopolizes the GPU for 50+ ms, creating severe Inter-Token Latency (ITL) stutter.
   - Disaggregation completely insulates active decodes from incoming prefill storms.

---

## Execution Guide

```bash
# Run the stress test harness on Colab GPU
python colab_runner.py run flagship_projects/03_disaggregated_prefill_decode_fabric/cluster_coordinator.py
```

### Deliverable in Your Portfolio
- A clear architectural schematic showing how the KV Transfer Fabric streams intermediate activations directly from P-node VRAM into D-node VRAM.
- Throughput and latency plots contrasting monolithic vs disaggregated serving under bursty prompt arrivals.
