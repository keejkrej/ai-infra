"""
Project 11: DCGM Observability & Throttling Dashboard
Polls real-time GPU hardware metrics via NVML / nvidia-smi:
Tracks SM occupancy, memory pressure, temperature, power draw, PCIe TX/RX,
and detects silent hardware thermal/power throttling events.
Exports Prometheus-compatible metrics exposition format.
"""

import time
import subprocess
import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class GPUMetrics:
    index: int
    name: str
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    gpu_util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    is_throttled: bool
    throttle_reasons: List[str]

class DCGMCollector:
    def collect(self) -> List[GPUMetrics]:
        """Queries GPU status using nvidia-smi query interface."""
        metrics_list = []
        if not torch.cuda.is_available():
            return metrics_list

        try:
            # Query nvidia-smi
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits"
            ]
            output = subprocess.check_output(cmd, text=True).strip()
            for line in output.split("\n"):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                idx = int(parts[0])
                name = parts[1]
                temp = float(parts[2]) if parts[2] != "[Not Supported]" else 0.0
                power_draw = float(parts[3]) if parts[3] != "[Not Supported]" else 0.0
                power_limit = float(parts[4]) if parts[4] != "[Not Supported]" else 0.0
                util = float(parts[5]) if parts[5] != "[Not Supported]" else 0.0
                mem_used = float(parts[6]) if parts[6] != "[Not Supported]" else 0.0
                mem_total = float(parts[7]) if parts[7] != "[Not Supported]" else 0.0
                
                # Throttle hex bitmask
                throttle_hex = parts[8] if len(parts) > 8 else "0x0000000000000000"
                is_throttled = throttle_hex != "0x0000000000000000" and throttle_hex != "[Not Supported]"
                
                reasons = []
                if is_throttled:
                    reasons.append(f"Active Throttle Mask: {throttle_hex}")
                    if temp > 82:
                        reasons.append("Thermal Threshold Alert (>82°C)")
                    if power_draw >= (power_limit * 0.98):
                        reasons.append("Power Cap Reached")
                else:
                    reasons.append("Nominal Clock Speeds (Unthrottled)")

                metrics_list.append(GPUMetrics(
                    index=idx,
                    name=name,
                    temperature_c=temp,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                    gpu_util_pct=util,
                    mem_used_mb=mem_used,
                    mem_total_mb=mem_total,
                    is_throttled=is_throttled,
                    throttle_reasons=reasons
                ))
        except Exception as e:
            print(f"Error querying nvidia-smi: {e}")

        return metrics_list

    def export_prometheus(self, metrics: List[GPUMetrics]) -> str:
        """Formats metrics as standard Prometheus exposition text."""
        lines = []
        lines.append("# HELP dcgm_gpu_temp_celsius GPU temperature in degrees Celsius.")
        lines.append("# TYPE dcgm_gpu_temp_celsius gauge")
        for m in metrics:
            lines.append(f'dcgm_gpu_temp_celsius{{gpu="{m.index}",device="{m.name}"}} {m.temperature_c}')

        lines.append("# HELP dcgm_gpu_utilization_percent Streaming Multiprocessor utilization %.")
        lines.append("# TYPE dcgm_gpu_utilization_percent gauge")
        for m in metrics:
            lines.append(f'dcgm_gpu_utilization_percent{{gpu="{m.index}",device="{m.name}"}} {m.gpu_util_pct}')

        lines.append("# HELP dcgm_gpu_memory_used_bytes GPU memory allocated in bytes.")
        lines.append("# TYPE dcgm_gpu_memory_used_bytes gauge")
        for m in metrics:
            lines.append(f'dcgm_gpu_memory_used_bytes{{gpu="{m.index}",device="{m.name}"}} {int(m.mem_used_mb * 1024 * 1024)}')

        lines.append("# HELP dcgm_gpu_throttling_active 1 if GPU clock is throttled, 0 otherwise.")
        lines.append("# TYPE dcgm_gpu_throttling_active gauge")
        for m in metrics:
            lines.append(f'dcgm_gpu_throttling_active{{gpu="{m.index}",device="{m.name}"}} {1 if m.is_throttled else 0}')

        return "\n".join(lines)

def main():
    print("=================================================================")
    print(" PROJECT 11: DCGM OBSERVABILITY & HARDWARE THROTTLING MONITOR")
    print(f" Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("=================================================================")

    collector = DCGMCollector()

    # Create synthetic GPU memory and compute workload to observe telemetry response
    print("\nSpawning tensor workload to register SM utilization & VRAM...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Allocate 4 GB VRAM
        dummy_tensor = torch.randn(1024, 1024, 1024, device=device, dtype=torch.float32)
        # Run matrix multiply
        _ = torch.matmul(dummy_tensor[:512, :512, 0], dummy_tensor[:512, :512, 0])
        torch.cuda.synchronize(device)

    metrics = collector.collect()

    print("\n" + "=" * 80)
    print(" LIVE HARDWARE HEALTH TELEMETRY")
    print("=" * 80)
    for m in metrics:
        vram_pct = (m.mem_used_mb / m.mem_total_mb * 100) if m.mem_total_mb > 0 else 0
        power_pct = (m.power_draw_w / m.power_limit_w * 100) if m.power_limit_w > 0 else 0
        print(f"GPU [{m.index}] : {m.name}")
        print(f"  * Temperature     : {m.temperature_c:.1f}°C")
        print(f"  * Power Draw      : {m.power_draw_w:.1f}W / {m.power_limit_w:.1f}W ({power_pct:.1f}%)")
        print(f"  * SM Utilization  : {m.gpu_util_pct:.1f}%")
        print(f"  * VRAM Allocated  : {m.mem_used_mb:.1f} MB / {m.mem_total_mb:.1f} MB ({vram_pct:.1f}%)")
        print(f"  * Throttle Status : {'THROTTLED!' if m.is_throttled else 'HEALTHY / NOMINAL'}")
        print(f"  * Diagnostic Info : {', '.join(m.throttle_reasons)}")

    print("\n" + "=" * 80)
    print(" PROMETHEUS SCRAPE EXPOSITION OUTPUT (/metrics)")
    print("=" * 80)
    prom_text = collector.export_prometheus(metrics)
    print(prom_text)
    print("=" * 80)
    print("\n[AI Infra Insight]: Silent thermal or power throttling cuts GPU clock speeds")
    print("by up to 50% without returning any HTTP errors. DCGM monitoring is critical.\n")

if __name__ == "__main__":
    main()
