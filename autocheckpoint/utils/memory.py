"""Memory utility functions for measuring GPU memory usage."""
import torch


def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_val / (1024 * 1024)

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string like '1.5 MB' or '2.3 GB'."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_to_mb(bytes_val):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"
    
def get_memory_allocated(device=None) -> int:
    if not torch.cuda.is_available():
        return 0
    
    if device is None:
        device = torch.cuda.current_device()

    return torch.cuda.memory_allocated(device)

def get_peak_memory(device=None) -> int:
    if not torch.cuda.is_available():
        return 0
    
    if device is None:
        device = torch.cuda.current_device()

    return torch.cuda.max_memory_allocated(device)