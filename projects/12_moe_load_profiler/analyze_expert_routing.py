"""
Project 12: MoE (Mixture of Experts) Load Profiler
Analyzes token-to-expert routing distributions in sparse MoE architectures (e.g. Top-2 gating over 8 experts).
Detects expert load imbalance, calculates routing entropy and Coefficient of Variation (CV),
and evaluates auxiliary load balancing loss to eliminate straggler latency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class MoETop2Router(nn.Module):
    def __init__(self, hidden_dim: int = 1024, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch_tokens, hidden_dim]
        Returns:
            top2_weights: [batch_tokens, 2]
            top2_indices: [batch_tokens, 2]
            router_probs: [batch_tokens, num_experts]
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top2_weights, top2_indices = torch.topk(probs, k=2, dim=-1)
        # Re-normalize top-2 weights to sum to 1
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)
        return top2_weights, top2_indices, probs

def compute_auxiliary_loss(probs: torch.Tensor, indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Standard Switch Transformer / Mixtral auxiliary load balancing loss:
    L = N * sum_{i=1}^N (f_i * P_i)
    where f_i is fraction of tokens dispatched to expert i, and P_i is average gating probability.
    """
    # f_i: fraction of tokens routed to expert i
    flat_indices = indices.view(-1)
    counts = torch.bincount(flat_indices, minlength=num_experts).float()
    f = counts / flat_indices.numel()

    # P_i: average probability allocated to expert i
    P = probs.mean(dim=0)

    # Balance loss
    loss = num_experts * torch.sum(f * P)
    return loss

def analyze_expert_distribution(indices: torch.Tensor, num_experts: int) -> Dict[str, float]:
    flat = indices.view(-1)
    counts = torch.bincount(flat, minlength=num_experts).float()
    total_tokens = flat.numel()

    fractions = counts / total_tokens
    mean_val = fractions.mean().item()
    std_val = fractions.std().item()

    # Coefficient of Variation (CV) = std / mean
    cv = std_val / (mean_val + 1e-12)

    # Routing Shannon Entropy
    probs = fractions.clamp(min=1e-12)
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    max_entropy = math.log2(num_experts)

    return {
        "counts": counts.tolist(),
        "cv": cv,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "max_load_expert": int(torch.argmax(counts).item()),
        "min_load_expert": int(torch.argmin(counts).item()),
        "max_load_pct": (torch.max(counts).item() / total_tokens) * 100,
        "min_load_pct": (torch.min(counts).item() / total_tokens) * 100
    }

def main():
    print("=================================================================")
    print(" PROJECT 12: MIXTURE OF EXPERTS (MoE) LOAD PROFILER")
    print(" Architecture: 8 Experts, Top-2 Gating | Imbalance & CV Profiling")
    print("=================================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_experts = 8
    hidden_dim = 1024
    num_tokens = 4096

    torch.manual_seed(42)
    tokens = torch.randn(num_tokens, hidden_dim, device=device)

    # Case 1: Skewed / Collapsed Router (Naive unregularized routing)
    print("\n[1/2] Profiling Imbalanced / Collapsed MoE Router...")
    imbalanced_router = MoETop2Router(hidden_dim, num_experts).to(device)
    # Inject synthetic bias to create expert popularity collapse
    with torch.no_grad():
        imbalanced_router.gate.weight[0] += 3.0  # Expert 0 heavily favored
        imbalanced_router.gate.weight[1] += 2.0  # Expert 1 heavily favored

    weights, indices, probs = imbalanced_router(tokens)
    stats_bad = analyze_expert_distribution(indices, num_experts)
    aux_loss_bad = compute_auxiliary_loss(probs, indices, num_experts)

    # Case 2: Balanced Router (Trained with auxiliary load-balancing loss)
    print("[2/2] Profiling Uniformly Balanced MoE Router (With Auxiliary Loss)...")
    balanced_router = MoETop2Router(hidden_dim, num_experts).to(device)
    # Run a quick 15-step optimization on router weights using auxiliary balance loss
    optimizer = torch.optim.Adam(balanced_router.parameters(), lr=0.05)
    for _ in range(25):
        optimizer.zero_grad()
        _, b_idx, b_probs = balanced_router(tokens)
        loss = compute_auxiliary_loss(b_probs, b_idx, num_experts)
        loss.backward()
        optimizer.step()

    b_weights, b_indices, b_probs = balanced_router(tokens)
    stats_good = analyze_expert_distribution(b_indices, num_experts)
    aux_loss_good = compute_auxiliary_loss(b_probs, b_indices, num_experts)

    # Print Distribution Comparison
    print("\n" + "=" * 80)
    print(f"{'Expert ID':<12} | {'Imbalanced Load (Tokens)':<28} | {'Balanced Load (Tokens)':<25}")
    print("-" * 80)
    for i in range(num_experts):
        bad_c = int(stats_bad["counts"][i])
        good_c = int(stats_good["counts"][i])
        print(f"Expert #{i:<6} | {bad_c:<8} ({bad_c/num_tokens*50:.1f}%)                 | {good_c:<8} ({good_c/num_tokens*50:.1f}%)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print(" MoE ROUTING STABILITY & STRAGGLER METRICS")
    print("=" * 80)
    print(f"Coefficient of Variation (CV) : {stats_bad['cv']:.3f} (Collapsed)  vs  {stats_good['cv']:.3f} (Balanced!)")
    print(f"Routing Entropy               : {stats_bad['entropy']:.2f} / {stats_bad['max_entropy']:.2f} bits  vs  {stats_good['entropy']:.2f} / {stats_good['max_entropy']:.2f} bits")
    print(f"Peak-to-Trough Expert Ratio   : {stats_bad['max_load_pct'] / max(stats_bad['min_load_pct'], 0.1):.1f}x imbalance  vs  {stats_good['max_load_pct'] / max(stats_good['min_load_pct'], 0.1):.1f}x (Uniform)")
    print("=" * 80)

    print("\n[AI Infra Insight]: In Expert Parallelism, token dispatch uses synchronous All-to-All.")
    print("A 5x overloaded expert turns every other GPU into an idle straggler waiting for the slowest expert!")
    print("Monitoring CV and tuning auxiliary loss restores uniform execution latency.\n")

if __name__ == "__main__":
    main()
