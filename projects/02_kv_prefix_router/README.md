# Project 02: KV-Cache Prefix Routing Proxy

## Objective
Build a reverse proxy load balancer that hashes incoming system prompts / context prefixes and routes requests to the specific worker replica holding those KV-cache blocks in VRAM.

## Why It Matters
- **The Redundant Prefill Tax**: Frontier LLM applications repeat identical system instructions (2,000+ tokens) across thousands of requests. Naive round-robin routing shuffles requests arbitrarily across workers, causing constant cache misses and redundant prefill computations.
- **Prefix Routing**: By combining consistent hashing of token blocks (e.g. 64-token chunks) with real-time replica cache tracking, requests sharing context prefixes hit the same replica, achieving 80%+ TTFT reductions and saving millions of FLOPs.

## Architecture
```
User Request (System Prompt + Query)
         │
         ▼
 [ Prefix Routing Proxy ]
    ├── Tokenize & Chunk into 64-token Blocks
    ├── Compute Hash Chain: [h_0, h_1, h_2, ...]
    └── Query Replica Prefix Directory
         │
         ├──► [ Replica 0 ] (Cache Hit: 100% prefix -> Zero Prefill TTFT!)
         ├──► [ Replica 1 ] (Cache Miss)
         └──► [ Replica 2 ] (Cache Miss)
```

## Benchmark Execution
```bash
python colab_runner.py run projects/02_kv_prefix_router/prefix_router_proxy.py
```
