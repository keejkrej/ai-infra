"""
Common utilities for AI Infrastructure projects.
Provides GPU inspection, timing decorators, metric printing, and device management.
"""

import time
import os
import sys
import json
import torch
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

def get_gpu_info() -> Dict[str, Any]:
    """Retrieve detailed information about available GPUs."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            })
    return info

def print_header(title: str, subtitle: Optional[str] = None):
    """Print a visually distinct CLI header."""
    bar = "=" * 70
    print("\n" + bar)
    print(f"  AI INFRA // {title.upper()}")
    if subtitle:
        print(f"  > {subtitle}")
    print(bar)

def print_section(name: str):
    """Print section divider."""
    print(f"\n--- [ {name} ] ---")

def print_metric(label: str, value: Any, unit: str = ""):
    """Print a formatted metric."""
    if isinstance(value, float):
        formatted = f"{value:.4f}"
    else:
        formatted = str(value)
    print(f"  * {label:<35}: {formatted} {unit}".rstrip())

class CUDATimer:
    """Accurate CUDA event timer with CPU fallback."""
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.cpu_start = 0.0
        self.cpu_end = 0.0

    def start(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            self.start_event.record()
        else:
            self.cpu_start = time.perf_counter()

    def stop(self) -> float:
        """Stops timer and returns elapsed time in milliseconds."""
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            return self.start_event.elapsed_time(self.end_event)
        else:
            self.cpu_end = time.perf_counter()
            return (self.cpu_end - self.cpu_start) * 1000.0

def measure_peak_memory_mb(device: int = 0) -> float:
    """Return peak allocated CUDA memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return 0.0

def reset_memory_stats(device: int = 0):
    """Reset CUDA peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
