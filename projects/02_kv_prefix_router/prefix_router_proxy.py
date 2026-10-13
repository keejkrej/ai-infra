"""
Project 02: KV-Cache Prefix Routing Proxy
A high-performance reverse proxy that hashes incoming prompt prefixes
and routes requests to replicas caching those KV blocks in memory.
"""

import hashlib
import time
import random
from typing import List, Dict, Optional, Tuple, Set

BLOCK_SIZE = 64  # Tokens per PagedAttention KV-block

def hash_token_block(tokens: Tuple[int, ...]) -> str:
    """Computes a fast 16-character SHA-256 hash digest for a block of tokens."""
    hasher = hashlib.sha256()
    for t in tokens:
        hasher.update(t.to_bytes(4, byteorder='little', signed=False))
    return hasher.hexdigest()[:16]

class ReplicaKVCache:
    """Represents an inference worker replica with a local PagedAttention KV-cache pool."""
    def __init__(self, replica_id: str, max_blocks: int = 500):
        self.replica_id = replica_id
        self.max_blocks = max_blocks
        # Set of block hashes currently retained in VRAM
        self.cached_blocks: Set[str] = set()
        self.lru_order: List[str] = []
        self.active_requests: int = 0

    def query_prefix_hits(self, block_hashes: List[str]) -> int:
        """Counts how many contiguous prefix blocks match this replica's cache."""
        hits = 0
        for h in block_hashes:
            if h in self.cached_blocks:
                hits += 1
            else:
                break
        return hits

    def allocate_and_cache(self, block_hashes: List[str]):
        """Caches new blocks using LRU eviction if full."""
        for h in block_hashes:
            if h in self.cached_blocks:
                self.lru_order.remove(h)
                self.lru_order.append(h)
            else:
                if len(self.cached_blocks) >= self.max_blocks and self.lru_order:
                    evict = self.lru_order.pop(0)
                    self.cached_blocks.remove(evict)
                self.cached_blocks.add(h)
                self.lru_order.append(h)

class PrefixRoutingProxy:
    """
    Load balancer routing incoming requests based on KV-cache prefix hits
    and replica load.
    """
    def __init__(self, replicas: List[ReplicaKVCache], load_penalty_weight: float = 0.5):
        self.replicas = replicas
        self.load_penalty_weight = load_penalty_weight

    def extract_block_hashes(self, token_ids: List[int]) -> List[str]:
        hashes = []
        num_blocks = len(token_ids) // BLOCK_SIZE
        for i in range(num_blocks):
            block = tuple(token_ids[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE])
            hashes.append(hash_token_block(block))
        return hashes

    def route_request(self, token_ids: List[int], strategy: str = "prefix_aware") -> Tuple[ReplicaKVCache, int]:
        block_hashes = self.extract_block_hashes(token_ids)
        total_blocks = len(block_hashes)

        if strategy == "round_robin":
            # Round-robin: route cyclically across replicas
            best_replica = self.replicas[getattr(self, "_rr_idx", 0) % len(self.replicas)]
            self._rr_idx = getattr(self, "_rr_idx", 0) + 1
            hits = best_replica.query_prefix_hits(block_hashes)
            best_replica.allocate_and_cache(block_hashes)
            return best_replica, hits

        # Prefix-aware routing:
        # 1. Search for replica with highest prefix match
        best_replica = None
        max_hits = 0
        for r in self.replicas:
            hits = r.query_prefix_hits(block_hashes)
            if hits > max_hits:
                max_hits = hits
                best_replica = r

        # 2. If no replica has cached this prefix yet, use consistent hashing on the first block
        if best_replica is None or max_hits == 0:
            if block_hashes:
                # Consistent hash of prefix block
                primary_idx = int(hashlib.md5(block_hashes[0].encode()).hexdigest(), 16) % len(self.replicas)
                best_replica = self.replicas[primary_idx]
            else:
                best_replica = self.replicas[0]
            max_hits = 0

        # Register cache on replica
        best_replica.allocate_and_cache(block_hashes)
        return best_replica, max_hits

def run_simulation():
    print("=================================================================")
    print(" PROJECT 02: KV-CACHE PREFIX ROUTING PROXY BENCHMARK")
    print(" Simulating 4 Replicas, Shared System Prompts & Multi-Turn Chats")
    print("=================================================================")

    # Define 50 distinct context prefixes (documents / system instructions)
    # Each document is 256 to 1024 tokens (4 to 16 blocks)
    random.seed(42)
    documents = {}
    for doc_id in range(50):
        length = random.choice([256, 512, 768, 1024])
        documents[f"doc_{doc_id}"] = [doc_id * 1000 + i for i in range(length)]

    # Generate 200 requests following a Zipf-like skew (common in production: top docs requested repeatedly)
    requests = []
    doc_keys = list(documents.keys())
    weights = [1.0 / (i + 1) ** 0.8 for i in range(len(doc_keys))]
    
    for req_id in range(200):
        chosen_doc = random.choices(doc_keys, weights=weights, k=1)[0]
        sys_tokens = documents[chosen_doc]
        user_tokens = [99999 + random.randint(1, 1000) for _ in range(random.randint(16, 64))]
        full_tokens = sys_tokens + user_tokens
        requests.append((req_id, chosen_doc, full_tokens, len(sys_tokens) // BLOCK_SIZE))

    # Constrain cache capacity per replica so naive round-robin thrashes
    cache_capacity_blocks = 32  # Only holds ~2-3 long docs per replica!

    # Test 1: Naive Round-Robin Routing
    replicas_rr = [ReplicaKVCache(f"replica-{i}", max_blocks=cache_capacity_blocks) for i in range(4)]
    proxy_rr = PrefixRoutingProxy(replicas_rr)
    rr_hits = []
    
    for _, _, tokens, _ in requests:
        _, hits = proxy_rr.route_request(tokens, strategy="round_robin")
        rr_hits.append(hits)

    # Test 2: Prefix-Aware Consistent Routing
    replicas_prefix = [ReplicaKVCache(f"replica-{i}", max_blocks=cache_capacity_blocks) for i in range(4)]
    proxy_prefix = PrefixRoutingProxy(replicas_prefix)
    prefix_hits = []

    for _, _, tokens, _ in requests:
        _, hits = proxy_prefix.route_request(tokens, strategy="prefix_aware")
        prefix_hits.append(hits)

    # Cost / TTFT model:
    # Cold prefill cost = 1.5 ms per block
    # Warm cached block cost = 0.05 ms per block (memory lookup only, no FLOPs!)
    total_blocks_requested = sum(len(r[2]) // BLOCK_SIZE for r in requests)
    
    def calculate_ttft(hits_list):
        ttfts = []
        for (_, _, tokens, _), hits in zip(requests, hits_list):
            total_b = len(tokens) // BLOCK_SIZE
            missed_b = total_b - hits
            ttft = (hits * 0.05) + (missed_b * 1.5)
            ttfts.append(ttft)
        return ttfts

    ttft_rr = calculate_ttft(rr_hits)
    ttft_prefix = calculate_ttft(prefix_hits)

    avg_rr_hit_rate = (sum(rr_hits) / total_blocks_requested) * 100
    avg_prefix_hit_rate = (sum(prefix_hits) / total_blocks_requested) * 100

    avg_rr_ttft = sum(ttft_rr) / len(ttft_rr)
    avg_prefix_ttft = sum(ttft_prefix) / len(ttft_prefix)
    ttft_reduction = ((avg_rr_ttft - avg_prefix_ttft) / avg_rr_ttft) * 100

    print("\n" + "=" * 70)
    print(f"{'Routing Strategy':<25} | {'KV Hit Rate %':<15} | {'Avg TTFT (ms)':<15}")
    print("-" * 70)
    print(f"{'Naive Round-Robin':<25} | {avg_rr_hit_rate:<15.1f} | {avg_rr_ttft:<15.2f}")
    print(f"{'Prefix-Aware Hash Router':<25} | {avg_prefix_hit_rate:<15.1f} | {avg_prefix_ttft:<15.2f}")
    print("=" * 70)
    print(f"\n[AI Infra Insight]: Prefix-aware routing slashed TTFT by {ttft_reduction:.1f}%")
    print(f"Total KV blocks reused without redundant GPU prefill: {sum(prefix_hits)} blocks ({sum(prefix_hits)*BLOCK_SIZE} tokens)\n")

if __name__ == "__main__":
    run_simulation()
