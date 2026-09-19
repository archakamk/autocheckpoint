"""Apply a checkpointing policy to a model in eager mode."""
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict


class CheckpointWrapper(nn.Module):
    """Wraps a submodule so its activations are recomputed in backward.

    Uses the non-reentrant checkpoint implementation (the modern default), which
    handles nested autograd and keyword arguments more robustly than reentrant.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, use_reentrant=False, **kwargs)


def _resolve(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_checkpointing(model: nn.Module, policy: Dict[str, bool]) -> nn.Module:
    """Wrap every submodule flagged True in `policy` with CheckpointWrapper.

    Mutates `model` in place and returns it. Names must match those produced by
    model.named_modules() (the same keys the profiler/optimizer use).
    """
    for name, do_ckpt in policy.items():
        if not do_ckpt or name == "":
            continue
        parent, child_name = _resolve(model, name)
        child = getattr(parent, child_name)
        if isinstance(child, CheckpointWrapper):
            continue
        setattr(parent, child_name, CheckpointWrapper(child))
    return model