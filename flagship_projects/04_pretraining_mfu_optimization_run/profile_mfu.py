"""
Flagship Project 04: PyTorch Profiler & Chrome Trace Timeline Generator
Executes training steps with torch.profiler, exporting a JSON trace file
viewable in chrome://tracing or Perfetto to inspect CUDA kernel launch overhead,
Tensor Core occupancy, and HBM memory transfers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_fsdp2_mfu import PretrainingTransformerLM

def generate_profiler_trace(trace_filename: str = "pretraining_trace.json"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Generating PyTorch Profiler trace on {device}...")

    model = PretrainingTransformerLM(vocab_size=8192, hidden_dim=2048, num_layers=8).to(device=device, dtype=torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch_size, seq_len = 4, 1024

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_traces"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(5):
            x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
            y = torch.randint(0, 8192, (batch_size, seq_len), device=device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 8192), y.view(-1))
            loss.backward()
            optimizer.step()
            prof.step()

    # Export chrome trace
    prof.export_chrome_trace(trace_filename)
    print(f"[+] Profiler trace successfully exported to: '{trace_filename}'")
    print(f"[+] Open 'chrome://tracing' or https://ui.perfetto.dev to inspect the trace timeline!\n")

if __name__ == "__main__":
    generate_profiler_trace()
