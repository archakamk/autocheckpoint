"""Memory profiler for PyTorch models."""
import torch
import torch.nn as nn

from autocheckpoint.utils.memory import format_bytes
from typing import Dict, Optional

class LayerStats:
    """Statistics for a single layer."""

    def __init__(self, name: str):
        self.name = name
        self.activation_size = 0
        self.num_parameters = 0
        self.output_shapes = []

    def __repr__(self):
        return f"LayerStats(name={self.name}, activation_size={format_bytes(self.activation_size)})"
    

class MemoryProfiler:
    """Profile memory usage for each layer in a PyTorch model."""

    def __init__(self, model: nn.Module):
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')
        self.layer_stats = {}
        self.hooks = []

    def profile(self, sample_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, LayerStats]:
        """
        Profile the model's memory usage with a sample input.
        Args:
            sample_input: a Tensor that is input data to pass to model
            target: an optional Tensor to compute loss
        Returns:
            Dictionary mapping layer names to LayerStats objects
        """
        sample_input = sample_input.to(self.device)
        if target is not None:
            target = target.to(self.device)

        self.layer_stats = {}

        # TODO: Register hooks on all layers

        # TODO: Run forward pass

        # TODO: Run backward pass if target exists

        # TODO: Remove hooks

        # TODO: Return statistics
    
    def _register_hooks(self):
        """Register forward and backward hooks on all modules"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue

            self.layer_stats[name] = LayerStats(name)

            # TODO: Register forward hook on this module
