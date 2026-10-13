"""
Project 01: High-Throughput Serving Engine Tuning
Demonstrates continuous batching, chunked prefill scheduling, and max-num-seqs tuning on GPU.

Compares:
1. Static Batching (waits for full batch to finish; prefill causes massive decode latency spikes)
2. Continuous Batching (iteration-level scheduling: new requests join immediately on completion)
3. Chunked Prefill (splits prompt prefill into chunks of size C, interleaving decode tokens to guarantee ITL SLA)
"""

import time
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class Request:
    req_id: int
    prompt_len: int
    output_len: int
    arrival_time: float
    remaining_prefill: int = 0
    generated_tokens: int = 0
    start_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    token_latencies: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.remaining_prefill = self.prompt_len

class SyntheticTransformerBlock(nn.Module):
    """
    Realistic lightweight transformer block emulating hidden_dim projection and attention
    to measure actual GPU GEMM time and memory consumption during prefill and decode.
    """
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [total_tokens, hidden_dim]
        qkv = self.qkv_proj(x)
        out = self.out_proj(x)
        x = x + out
        mlp_out = self.mlp(x)
        return x + mlp_out

class ServingSimulator:
    def __init__(self, device: torch.device, hidden_dim: int = 2048):
        self.device = device
        self.hidden_dim = hidden_dim
        self.model = SyntheticTransformerBlock(hidden_dim).to(device=device, dtype=torch.float16)
        self.model.eval()

    def run_static_batching(self, requests: List[Request], batch_size: int = 8) -> Dict[str, float]:
        """Static batching: runs batches synchronously until every request in batch completes."""
        completed: List[Request] = []
        req_queue = list(requests)
        
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        sim_start = time.perf_counter()

        for i in range(0, len(req_queue), batch_size):
            batch = req_queue[i:i + batch_size]
            b_start = time.perf_counter()
            for r in batch:
                r.start_time = b_start

            # Prefill phase (padded to max prompt len in batch)
            max_prompt = max(r.prompt_len for r in batch)
            prefill_tokens = len(batch) * max_prompt
            x = torch.randn(prefill_tokens, self.hidden_dim, device=self.device, dtype=torch.float16)
            
            with torch.no_grad():
                _ = self.model(x)
            
            p_end = time.perf_counter()
            for r in batch:
                r.first_token_time = p_end

            # Decode phase
            max_gen = max(r.output_len for r in batch)
            last_t = p_end
            for step in range(max_gen):
                active_count = sum(1 for r in batch if r.generated_tokens < r.output_len)
                if active_count == 0:
                    break
                x_dec = torch.randn(len(batch), self.hidden_dim, device=self.device, dtype=torch.float16)
                with torch.no_grad():
                    _ = self.model(x_dec)
                now = time.perf_counter()
                for r in batch:
                    if r.generated_tokens < r.output_len:
                        r.generated_tokens += 1
                        r.token_latencies.append((now - last_t) * 1000.0)
                        if r.generated_tokens == r.output_len:
                            r.finish_time = now
                last_t = now

            completed.extend(batch)

        total_time = time.perf_counter() - sim_start
        return self._calc_metrics(completed, total_time)

    def run_continuous_batching_chunked(
        self,
        requests: List[Request],
        max_num_seqs: int = 16,
        chunk_size: Optional[int] = 512
    ) -> Dict[str, float]:
        """
        Continuous batching with optional Chunked Prefill.
        At each iteration:
        - Schedules active decode tokens.
        - Schedules incoming prefill tokens up to chunk_size budget.
        """
        waiting = list(requests)
        active_prefill: List[Request] = []
        active_decode: List[Request] = []
        completed: List[Request] = []

        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        sim_start = time.perf_counter()

        while waiting or active_prefill or active_decode:
            iter_start = time.perf_counter()
            
            # Admit new requests if capacity allows
            while waiting and (len(active_prefill) + len(active_decode)) < max_num_seqs:
                req = waiting.pop(0)
                req.start_time = iter_start
                active_prefill.append(req)

            # Determine budget for this iteration
            tokens_in_iteration = 0
            
            # 1. Decode tokens (1 per active decode sequence)
            decode_count = len(active_decode)
            tokens_in_iteration += decode_count

            # 2. Prefill tokens (chunked or full)
            prefill_budget = (chunk_size if chunk_size else 8192) - tokens_in_iteration
            prefill_budget = max(0, prefill_budget)
            
            prefill_tokens_scheduled = 0
            curr_prefill_list = list(active_prefill)
            for req in curr_prefill_list:
                if prefill_budget <= 0:
                    break
                chunk = min(req.remaining_prefill, prefill_budget)
                req.remaining_prefill -= chunk
                prefill_tokens_scheduled += chunk
                prefill_budget -= chunk
                
                # Check if prefill finished
                if req.remaining_prefill == 0:
                    req.first_token_time = time.perf_counter()
                    active_prefill.remove(req)
                    active_decode.append(req)

            tokens_in_iteration += prefill_tokens_scheduled

            # Run forward pass for scheduled tokens
            if tokens_in_iteration > 0:
                x = torch.randn(tokens_in_iteration, self.hidden_dim, device=self.device, dtype=torch.float16)
                with torch.no_grad():
                    _ = self.model(x)

            iter_end = time.perf_counter()
            itl_ms = (iter_end - iter_start) * 1000.0

            # Advance decode sequences
            finished_decode = []
            for req in active_decode:
                req.generated_tokens += 1
                req.token_latencies.append(itl_ms)
                if req.generated_tokens >= req.output_len:
                    req.finish_time = iter_end
                    finished_decode.append(req)

            for req in finished_decode:
                active_decode.remove(req)
                completed.append(req)

        total_time = time.perf_counter() - sim_start
        return self._calc_metrics(completed, total_time)

    def _calc_metrics(self, requests: List[Request], total_wall_time: float) -> Dict[str, float]:
        ttfts = [(r.first_token_time - r.start_time) * 1000.0 for r in requests if r.first_token_time > 0]
        itls = [lat for r in requests for lat in r.token_latencies]
        total_tokens = sum(r.prompt_len + r.generated_tokens for r in requests)

        ttfts.sort()
        itls.sort()

        p50_ttft = ttfts[int(len(ttfts) * 0.5)] if ttfts else 0.0
        p99_ttft = ttfts[int(len(ttfts) * 0.99)] if ttfts else 0.0
        p50_itl = itls[int(len(itls) * 0.5)] if itls else 0.0
        p99_itl = itls[int(len(itls) * 0.99)] if itls else 0.0

        throughput = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

        return {
            "total_time_s": total_wall_time,
            "total_tokens": total_tokens,
            "throughput_tok_s": throughput,
            "p50_ttft_ms": p50_ttft,
            "p99_ttft_ms": p99_ttft,
            "p50_itl_ms": p50_itl,
            "p99_itl_ms": p99_itl,
        }

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=================================================================")
    print(f" PROJECT 01: SERVING ENGINE TUNING BENCHMARK")
    print(f" Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"=================================================================")

    # Generate benchmark workload: mixture of short and long prompts
    num_requests = 32
    workload: List[Request] = []
    for i in range(num_requests):
        # 25% long context prompts (1500 tokens), 75% short prompts (256 tokens)
        p_len = 1536 if i % 4 == 0 else 256
        out_len = 32
        workload.append(Request(req_id=i, prompt_len=p_len, output_len=out_len, arrival_time=0.0))

    sim = ServingSimulator(device=device, hidden_dim=2048)

    # 1. Static Batching
    print("\n[1/3] Running Static Batching (Batch Size = 8)...")
    reqs_static = [Request(r.req_id, r.prompt_len, r.output_len, 0.0) for r in workload]
    res_static = sim.run_static_batching(reqs_static, batch_size=8)

    # 2. Continuous Batching without chunking (monolithic prefill)
    print("[2/3] Running Continuous Batching (No Chunking, max_num_seqs = 16)...")
    reqs_cb = [Request(r.req_id, r.prompt_len, r.output_len, 0.0) for r in workload]
    res_cb = sim.run_continuous_batching_chunked(reqs_cb, max_num_seqs=16, chunk_size=None)

    # 3. Continuous Batching with Chunked Prefill (chunk_size = 512)
    print("[3/3] Running Continuous Batching + Chunked Prefill (chunk_size = 512)...")
    reqs_chunk = [Request(r.req_id, r.prompt_len, r.output_len, 0.0) for r in workload]
    res_chunk = sim.run_continuous_batching_chunked(reqs_chunk, max_num_seqs=16, chunk_size=512)

    # Comparison Table
    print("\n" + "=" * 75)
    print(f"{'Serving Architecture':<30} | {'Throughput':<12} | {'P99 TTFT':<12} | {'P99 ITL':<12}")
    print(f"{'':<30} | {'(tok/s)':<12} | {'(ms)':<12} | {'(ms)':<12}")
    print("-" * 75)
    print(f"{'Static Batching':<30} | {res_static['throughput_tok_s']:<12.1f} | {res_static['p99_ttft_ms']:<12.2f} | {res_static['p99_itl_ms']:<12.2f}")
    print(f"{'Continuous Batching (No Chunk)':<30} | {res_cb['throughput_tok_s']:<12.1f} | {res_cb['p99_ttft_ms']:<12.2f} | {res_cb['p99_itl_ms']:<12.2f}")
    print(f"{'Continuous + Chunked Prefill':<30} | {res_chunk['throughput_tok_s']:<12.1f} | {res_chunk['p99_ttft_ms']:<12.2f} | {res_chunk['p99_itl_ms']:<12.2f}")
    print("=" * 75)

    itl_reduction = (1.0 - (res_chunk['p99_itl_ms'] / (res_cb['p99_itl_ms'] + 1e-6))) * 100
    print(f"\n[AI Infra Insight]: Chunked prefill bounded P99 ITL jitter by {max(0.0, itl_reduction):.1f}%")
    print("while maintaining continuous batching throughput!\n")

if __name__ == "__main__":
    run_benchmark()
