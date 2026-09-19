import warnings
from typing import Dict, List, Optional, Tuple
from core.profiler import LayerStats


class CheckpointOptimizer:
    """Finds an optimal gradient-checkpointing policy via dynamic programming.

    Wrapping layer i in checkpointing frees m_i bytes (its activation) but costs
    t_i seconds (one extra forward). We must free at least R = (peak - budget)
    bytes while minimizing total recompute time -> 0/1 knapsack, solved exactly
    with a pseudo-polynomial DP over quantized memory buckets. (Greedy is only
    optimal for the fractional relaxation.)
    """

    def __init__(self, layer_stats: Dict[str, LayerStats], mem_budget: int,
                 recompute_costs: Optional[Dict[str, float]] = None,
                 bucket_bytes: int = 1024 * 1024):
        self.layer_stats = layer_stats
        self.mem_budget = mem_budget
        self.recompute_costs = recompute_costs or {
            n: float(s.activation_size) for n, s in layer_stats.items()}
        self.bucket_bytes = bucket_bytes

    def find_policy(self) -> Dict[str, bool]:
        cand: List[Tuple[str, int, float]] = [
            (n, s.activation_size, self.recompute_costs.get(n, 0.0))
            for n, s in self.layer_stats.items() if s.activation_size > 0]
        total = sum(m for _, m, _ in cand)
        required = total - self.mem_budget
        if required <= 0:
            return {n: False for n, _, _ in cand}

        B = self.bucket_bytes
        cap = (required + B - 1) // B
        n = len(cand)
        INF = float("inf")
        dp = [[INF] * (cap + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        for i in range(1, n + 1):
            _, m, t = cand[i - 1]
            s = min(cap, m // B)
            for k in range(cap + 1):
                keep = dp[i - 1][k]
                take = dp[i - 1][max(0, k - s)] + t
                dp[i][k] = min(keep, take)

        policy = {nm: False for nm, _, _ in cand}
        if dp[n][cap] == INF:
            warnings.warn("Budget infeasible; checkpointing all candidates.")
            return {nm: True for nm, _, _ in cand}
        k = cap
        for i in range(n, 0, -1):
            nm, m, _ = cand[i - 1]
            if dp[i][k] != dp[i - 1][k]:
                policy[nm] = True
                k = max(0, k - min(cap, m // B))
        return policy