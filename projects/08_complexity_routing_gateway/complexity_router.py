"""
Project 08: Complexity-Routing AI Gateway
Analyzes prompt Shannon entropy, token length, code syntax, and reasoning markers
to dynamically route requests between a local quantized 8B model and a frontier reasoning API.
"""

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class RoutingDecision:
    prompt_snippet: str
    token_count: int
    entropy: float
    reasoning_score: float
    composite_complexity: float
    target_backend: str  # "LOCAL_8B_QUANT" or "FRONTIER_API"
    estimated_cost_usd: float

class ComplexityRouter:
    REASONING_KEYWORDS = {
        "prove", "theorem", "derive", "analyze", "critique", "architect",
        "step-by-step", "mathematical", "optimize", "debug", "refactor",
        "concurrency", "deadlock", "algorithm", "tradeoff", "bottleneck"
    }

    def __init__(
        self,
        complexity_threshold: float = 0.45,
        local_cost_per_1m: float = 0.08,     # Local 8B FP8 on spot GPU
        frontier_cost_per_1m: float = 5.00   # Frontier API (e.g. Claude / GPT-4o)
    ):
        self.complexity_threshold = complexity_threshold
        self.local_cost_per_1m = local_cost_per_1m
        self.frontier_cost_per_1m = frontier_cost_per_1m

    def compute_shannon_entropy(self, text: str) -> float:
        """Computes normalized Shannon entropy of character n-grams."""
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        # Normalize by maximum theoretical entropy for ASCII (approx 6.5)
        return min(1.0, entropy / 5.5)

    def compute_reasoning_density(self, text: str) -> float:
        """Detects presence of technical keywords and code syntax."""
        lower = text.lower()
        words = re.findall(r'\b\w+\b', lower)
        if not words:
            return 0.0

        keyword_hits = sum(1 for w in words if w in self.REASONING_KEYWORDS)
        keyword_density = min(1.0, (keyword_hits * 3.0) / len(words))

        # Check for code blocks or math symbols
        has_code = 0.3 if "```" in text or "def " in text or "class " in text else 0.0
        has_math = 0.2 if any(c in text for c in ["∑", "∫", "λ", "≤", "≥", "≠"]) else 0.0

        return min(1.0, keyword_density + has_code + has_math)

    def route(self, prompt: str) -> RoutingDecision:
        tokens = len(prompt.split()) * 1.3  # Estimated token count
        entropy = self.compute_shannon_entropy(prompt)
        reasoning = self.compute_reasoning_density(prompt)

        # Length factor (prompts > 1000 tokens are more likely complex docs)
        length_factor = min(1.0, tokens / 800.0)

        # Composite complexity score in [0.0, 1.0]
        complexity = (entropy * 0.30) + (reasoning * 0.45) + (length_factor * 0.25)

        if complexity >= self.complexity_threshold:
            backend = "FRONTIER_API"
            cost = (tokens / 1_000_000.0) * self.frontier_cost_per_1m
        else:
            backend = "LOCAL_8B_QUANT"
            cost = (tokens / 1_000_000.0) * self.local_cost_per_1m

        snippet = prompt if len(prompt) < 45 else prompt[:42] + "..."
        return RoutingDecision(
            prompt_snippet=snippet,
            token_count=int(tokens),
            entropy=entropy,
            reasoning_score=reasoning,
            composite_complexity=complexity,
            target_backend=backend,
            estimated_cost_usd=cost
        )

def run_gateway_benchmark():
    print("=================================================================")
    print(" PROJECT 08: COMPLEXITY-ROUTING AI GATEWAY")
    print(" Shannon Entropy + Reasoning Heuristics + Unit Economics")
    print("=================================================================")

    # Sample realistic queries across simple chat, QA, and complex engineering
    queries = [
        "Hello, how are you doing today?",
        "What is the capital of France?",
        "Translate this email to Spanish: 'Meeting tomorrow at 9am.'",
        "Summarize this short note: 'John bought 3 apples and 2 oranges.'",
        "Write a quick regex to match standard email addresses.",
        "Derive the gradient update for a multi-head self-attention layer and prove convergence.",
        "Analyze the deadlock vulnerability in this concurrent Rust channel implementation: ```rust fn send() {}```",
        "Architect a disaggregated KV-cache cluster over RDMA RoCE v2 with sub-millisecond failover.",
        "Give me 5 synonyms for the word 'innovative'.",
        "Write a complete fused Triton kernel for RoPE rotary embeddings with benchmark comparisons."
    ]

    router = ComplexityRouter(complexity_threshold=0.45)
    decisions: List[RoutingDecision] = []

    print(f"\n{'Query':<35} | {'Entropy':<8} | {'Score':<8} | {'Target Backend':<16} | {'Est Cost'}")
    print("-" * 80)

    for q in queries:
        dec = router.route(q)
        decisions.append(dec)
        print(f"{dec.prompt_snippet:<35} | {dec.entropy:<8.2f} | {dec.composite_complexity:<8.2f} | {dec.target_backend:<16} | ${dec.estimated_cost_usd:.6f}")

    # Economics analysis
    local_count = sum(1 for d in decisions if d.target_backend == "LOCAL_8B_QUANT")
    frontier_count = sum(1 for d in decisions if d.target_backend == "FRONTIER_API")
    total_routed_cost = sum(d.estimated_cost_usd for d in decisions)
    
    # Baseline: If 100% of queries went to Frontier API
    baseline_frontier_cost = sum((d.token_count / 1_000_000.0) * router.frontier_cost_per_1m for d in decisions)
    cost_saved = baseline_frontier_cost - total_routed_cost
    pct_saved = (cost_saved / baseline_frontier_cost) * 100

    print("\n" + "=" * 80)
    print(" ROUTING ECONOMICS BREAKDOWN")
    print("=" * 80)
    print(f"Total Queries Evaluated     : {len(decisions)}")
    print(f"Routed to Local 8B Model    : {local_count} ({local_count / len(decisions) * 100:.0f}%)")
    print(f"Routed to Frontier API      : {frontier_count} ({frontier_count / len(decisions) * 100:.0f}%)")
    print(f"All-Frontier Baseline Spend : ${baseline_frontier_cost:.6f}")
    print(f"Gateway Routed Total Spend  : ${total_routed_cost:.6f}")
    print(f"Net Cloud Margin Protection : ${cost_saved:.6f} ({pct_saved:.1f}% SAVED!)")
    print("=" * 80)
    print("\n[AI Infra Insight]: Dynamic complexity routing protects gross margins")
    print("by preventing low-complexity queries from burning expensive frontier API tokens.\n")

if __name__ == "__main__":
    run_gateway_benchmark()
