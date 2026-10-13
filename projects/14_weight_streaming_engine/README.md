# Project 14: Checkpointing & Weight Streaming Engine

## Objective
Implement a high-throughput weight streaming pipeline that streams LLM model weights directly from network memory buffers into GPU VRAM using page-locked (pinned) host memory and asynchronous CUDA DMA streams (`cudaMemcpyAsync`), bypassing local disk I/O bottlenecks to minimize cold-start autoscaling latency.

## Why It Matters
- **The Cold-Start Disaster**: In cloud environments, spinning up a new GPU node (e.g. for a 70B parameter model) requires downloading ~140 GB from S3. Traditional approaches write weights to local disk (EBS/NVMe), read them back into CPU RAM with `torch.load()`, and then copy to GPU. This takes **10 to 20 minutes**, rendering autoscaling useless against sudden traffic spikes.
- **Direct Pinned Streaming**: By streaming chunks directly into page-locked host memory buffers and overlapping PCIe transfers via asynchronous CUDA streams, weights enter GPU High-Bandwidth Memory (HBM) at maximum bus bandwidth (25+ GB/s over PCIe Gen4), cutting cold-start delays from minutes to seconds.

## Cold-Start Comparison (1.0 GB Tensor Chunk)
- **Naive Disk Load**: Writes to disk $\to$ reads back $\to$ synchronous VRAM copy.
- **Direct Async Stream**: Zero-copy pinned memory buffer $\to$ asynchronous DMA transfer stream.

## Run Benchmark
```bash
python colab_runner.py run projects/14_weight_streaming_engine/benchmark_cold_start.py
```
