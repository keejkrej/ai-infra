"""
Flagship Project 04: PyTorch Profiler & Chrome Trace Timeline Generator
Executes training steps with torch.profiler, exporting a JSON trace file
viewable in chrome://tracing or Perfetto to inspect CUDA kernel launch overhead,
Tensor Core occupancy, and HBM memory transfers.
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_fsdp2_mfu
importlib.reload(train_fsdp2_mfu)
from train_fsdp2_mfu import PretrainingTransformerLM

def generate_profiler_trace(trace_filename: str = "pretraining_trace.json"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Generating PyTorch Profiler trace on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})...")

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    model = PretrainingTransformerLM(vocab_size=8192, hidden_dim=2048, num_layers=8).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device.type == "cuda"))
    batch_size, seq_len = 4, 1024

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for step in range(4):
            x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
            y = torch.randint(0, 8192, (batch_size, seq_len), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 8192), y.view(-1))
            loss.backward()
            optimizer.step()
            prof.step()

    # Export chrome trace
    prof.export_chrome_trace(trace_filename)
    print(f"[+] Profiler trace successfully exported to: '{trace_filename}'")
    print("\nTop 10 CUDA Kernels by Total GPU Time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with open("profiler_kernel_summary.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    generate_profiler_trace()
