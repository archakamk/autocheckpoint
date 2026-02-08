"""Memory profiler for PyTorch models."""
import torch
import torch.nn as nn

class LayerStats:
    """Statistics for a single layer."""

    def __init__(self, name: str):
        self.name = name
        self.activation_size = 0
        self.num_parameters = 0
        self.output_shapes = []