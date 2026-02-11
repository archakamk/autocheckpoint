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

        # Register hooks on all layers
        self._register_hooks()

        # Run forward pass
        self.model.train()
        output = self.model(sample_input)

        # Run backward pass if target exists
        if target is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
            loss.backward()

        # Remove hooks
        self._remove_hooks()

        # Return statistics
        return self.layer_stats
    
    def _register_hooks(self):
        """Register forward and backward hooks on all modules"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue

            self.layer_stats[name] = LayerStats(name)
            self.layer_stats[name].num_parameters = sum(p.numel() for p in module.parameters())

            hook = self._make_forward_hook(name)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)


    def _make_forward_hook(self, name: str):
        """Create forward hook for a specific layer"""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats = self.layer_stats[name]
                stats.activation_size = output.element_size() * output.nelement()    
            elif isinstance(output, tuple):
                total_size = 0
                for item in output:
                    if isinstance(item, torch.Tensor):
                        total_size += item.element_size() * item.nelement()
                stats.activation_size = total_size

        return hook
    
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()

        self.hooks = []

    
    def print_summary(self):
        """Print a formatted summary of profiling results."""
        print("\n" + "="*80)
        print("Memory Profiling Summary")
        print("="*80)

        # Print column headers
        print(f"{'Layer':<40} {'Activation Size':>20} {'Params':>15}")
        print("-"*80)
        
        # Initialize totals
        total_activation = 0
        total_params = 0
        
        # Loop through stats and print each layer
        for name, stats in self.layer_stats.items():
            if stats.activation_size > 0:
                print(f"{name:<40} {format_bytes(stats.activation_size):>20} {stats.num_parameters:>15,}")
                total_activation += stats.activation_size
                total_params += stats.num_parameters
        
        # Print separator
        print("-"*80)
        
        # Print totals
        print(f"{'TOTAL':<40} {format_bytes(total_activation):>20} {total_params:>15,}")
        
        print("="*80)