import torch

from typing import Dict, Optional
from core.profiler import LayerStats

class CheckpointOptimizer:
    """Finds optimal checkpointing policy using dynamic programming."""

    def __init__(self, layer_stats: Dict[str, LayerStats], mem_budget: int, time_budget: float):
        self.layer_stats = layer_stats
        self.mem_budget = mem_budget
        self.time_budget = time_budget

    def find_policy(self) -> Dict[str, bool]:
        """
        Find optimal checkpointing policy.
        Returns:
            Dictionary mapping layer names to checkpoint decision (True/False)
        """
        
