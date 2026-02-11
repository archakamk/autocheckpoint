"""Test the memory profiler with a simple model."""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.profiler import MemoryProfiler


class TinyModel(nn.Module):
    """Tiny model for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print("\n🧠 Testing Memory Profiler\n")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = TinyModel().to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Create profiler
    profiler = MemoryProfiler(model)
    
    # Create sample data
    batch_size = 4
    sample_input = torch.randn(batch_size, 3, 32, 32)
    sample_target = torch.randint(0, 10, (batch_size,))
    
    # Run profiler
    print("Profiling...")
    stats = profiler.profile(sample_input, sample_target)
    
    # Print summary
    profiler.print_summary()
    
    print("\n✅ Test complete!\n")