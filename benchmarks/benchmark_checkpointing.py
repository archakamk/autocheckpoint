"""Speed-vs-memory benchmark: AutoCheckpoint (module-level, eager) versus
PyTorch's built-in activation-memory-budget partitioner (op-level, compiled).

Both solve the same 0/1 knapsack "minimize recompute under a memory budget";
this measures the gap between doing it at module granularity in eager mode and
at operator granularity inside torch.compile. Run on a CUDA or ROCm GPU:

    python benchmarks/benchmark_checkpointing.py --depth 12 --dim 1024 --batch 8 --seq 1024

Outputs a table of (peak memory, step time) points and, if matplotlib is
present, a tradeoff scatter to benchmarks/tradeoff.png.
"""
import argparse
import json
import time
import contextlib
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.profiler import LayerStats
from core.optimizer import CheckpointOptimizer
from core.apply import apply_checkpointing


# --------------------------- model ---------------------------
class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP."""
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class GPTLike(nn.Module):
    def __init__(self, depth, dim, heads, vocab=50257):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


# --------------------------- measurement ---------------------------
def measure(model, idx, target, device, warmup=3, iters=10):
    """Return (peak_bytes, mean_step_ms) for a full train step."""
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(idx)
        loss = loss_fn(out.view(-1, out.size(-1)), target.view(-1))
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(), start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        return 0, (time.perf_counter() - t0) * 1000 / iters


def profile_blocks(model, idx, device):
    """Measure per-block activation bytes and forward time in one pass."""
    stats, costs, order = {}, {}, []
    handles = []
    ev = {}

    def pre(name):
        def hook(mod, inp):
            if device.type == "cuda":
                ev[name] = [torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True)]
                ev[name][0].record()
            else:
                ev[name] = time.perf_counter()
        return hook

    def post(name):
        def hook(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            stats[name] = t.element_size() * t.nelement()
            if device.type == "cuda":
                ev[name][1].record()
            else:
                costs[name] = (time.perf_counter() - ev[name]) * 1000
            order.append(name)
        return hook

    for i, b in enumerate(model.blocks):
        name = f"blocks.{i}"
        handles.append(b.register_forward_pre_hook(pre(name)))
        handles.append(b.register_forward_hook(post(name)))

    model(idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
        for name in stats:
            costs[name] = ev[name][0].elapsed_time(ev[name][1])
    for h in handles:
        h.remove()
    return stats, costs


# --------------------------- strategies ---------------------------
def autocheckpoint_policy(block_bytes, block_costs, keep_fraction):
    """Free (1 - keep_fraction) of total block activation, minimizing recompute."""
    ls = {n: LayerStats(n) for n in block_bytes}
    for n, b in block_bytes.items():
        ls[n].activation_size = b
    total = sum(block_bytes.values())
    budget = int(keep_fraction * total)
    return CheckpointOptimizer(ls, budget, recompute_costs=block_costs).find_policy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU detected; memory numbers will be 0. "
              "Run on a CUDA/ROCm GPU for meaningful results.\n")

    def make():
        torch.manual_seed(0)
        return GPTLike(args.depth, args.dim, args.heads).to(device)

    idx = torch.randint(0, 50257, (args.batch, args.seq), device=device)
    target = torch.randint(0, 50257, (args.batch, args.seq), device=device)

    results = []

    def record(label, model):
        mem, ms = measure(model, idx, target, device, iters=args.iters)
        results.append({"strategy": label, "peak_gb": mem / 1e9, "step_ms": ms})
        print(f"  {label:<28} {mem/1e9:6.2f} GB   {ms:8.1f} ms")

    print("=" * 64)
    print("Eager strategies")
    print("=" * 64)
    # baseline
    record("eager baseline", make())
    # checkpoint every block
    m = make()
    apply_checkpointing(m, {f"blocks.{i}": True for i in range(args.depth)})
    record("eager checkpoint-all", m)
    # AutoCheckpoint curve
    block_bytes, block_costs = profile_blocks(make(), idx, device)
    for f in (0.75, 0.50, 0.25):
        m = make()
        pol = autocheckpoint_policy(block_bytes, block_costs, f)
        apply_checkpointing(m, pol)
        n_ck = sum(pol.values())
        record(f"autockpt keep={f:.2f} ({n_ck} blk)", m)

    print("=" * 64)
    print("torch.compile strategies")
    print("=" * 64)
    try:
        import torch._functorch.config as fconfig
        import torch._dynamo as dynamo
        # default partitioner (runtime-optimized)
        dynamo.reset()
        record("compile default", torch.compile(make()))
        # memory-budget sweep: 1.0 = save all, 0.0 = checkpoint all
        for budget in (0.7, 0.5, 0.3):
            dynamo.reset()
            fconfig.activation_memory_budget = budget
            record(f"compile budget={budget:.1f}", torch.compile(make()))
        fconfig.activation_memory_budget = 1.0
    except Exception as e:  # noqa
        print(f"  torch.compile path skipped: {e}")

    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with contextlib.suppress(Exception):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eager = [r for r in results if "compile" not in r["strategy"]]
        comp = [r for r in results if "compile" in r["strategy"]]
        for grp, mk, lbl in ((eager, "o", "AutoCheckpoint (eager)"),
                             (comp, "s", "torch.compile budget")):
            if grp:
                plt.scatter([r["peak_gb"] for r in grp],
                            [r["step_ms"] for r in grp], marker=mk, label=lbl)
        plt.xlabel("Peak memory (GB)"); plt.ylabel("Step time (ms)")
        plt.title("Memory vs. compute tradeoff"); plt.legend(); plt.grid(alpha=.3)
        plt.savefig(os.path.join(os.path.dirname(__file__), "tradeoff.png"), dpi=120)
        print("\nSaved tradeoff.png")


if __name__ == "__main__":
    main()