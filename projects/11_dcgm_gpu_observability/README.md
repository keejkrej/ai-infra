# Project 11: DCGM Observability & Throttling Dashboard

## Objective
Implement an NVIDIA Data Center GPU Manager (DCGM) & NVML telemetry collector tracking GPU temperature, SM occupancy, memory usage, power caps, and hardware throttling flags, exporting Prometheus `/metrics` exposition data for Grafana visualization.

## Why It Matters
- **The "HTTP 200 While Melting" Trap**: When a GPU reaches thermal junction limits (>82°C) or exceeds power caps, it does not fail health checks. Instead, it enters **Hardware Thermal/Power Throttling**, silently reducing clock frequencies by 30-50%.
- **SLO Violations**: API response latency suddenly doubles or triples for users while backend dashboards still show green HTTP 200 statuses.
- **Root Cause Telemetry**: By polling DCGM throttle reason bitmasks (`HW_SLOWDOWN`, `SW_POWER_CAP`, `HW_THERMAL`), engineers detect hardware degradation immediately.

## Exported Metrics Format
- `dcgm_gpu_temp_celsius`: GPU temperature in °C
- `dcgm_gpu_utilization_percent`: SM Tensor Core utilization %
- `dcgm_gpu_memory_used_bytes`: VRAM allocated in bytes
- `dcgm_gpu_throttling_active`: Binary indicator (1 if throttled, 0 if nominal)

## Run Telemetry Collector
```bash
python colab_runner.py run projects/11_dcgm_gpu_observability/dcgm_collector.py
```
