#!/usr/bin/env python
"""Where the time goes inside `_compute_messages`, at flyvis scale.

The cluster benchmark (`tools/bench_throughput.py`) measures the whole training
step and takes ~5-20 min per point. This one runs the message-passing kernel
alone, locally, in seconds -- so an optimisation can be rejected before it costs a
queue slot.

Reproduces the real shape: N=13741 neurons, E=434112 edges, batch_size 4, so
E*B=1,736,448 edges per forward, with g_phi 3->80->80->1 (flyvis_A) or 6->80->80->1
(conductance). At those sizes ONE [E*B, 80] fp32 activation is 556 MB, and g_phi
keeps two of them for the backward -- which is the thing worth removing.

    python tools/bench_message_passing.py
    python tools/bench_message_passing.py --model flyvis_conductance --iters 50
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3      # ms


class GPhi(nn.Module):
    """The same 3-layer MLP shape models/MLP.py builds, ReLU between layers."""

    def __init__(self, din, hidden, dout, device):
        super().__init__()
        self.l0 = nn.Linear(din, hidden, device=device)
        self.l1 = nn.Linear(hidden, hidden, device=device)
        self.l2 = nn.Linear(hidden, dout, device=device)

    def forward(self, x):
        return self.l2(F.relu(self.l1(F.relu(self.l0(x)))))


def build(args, device):
    N, E, B = args.n_neurons, args.n_edges, args.batch_size
    g = torch.Generator(device="cpu").manual_seed(0)
    edge_index = torch.stack([
        torch.randint(0, N, (E,), generator=g),
        torch.randint(0, N, (E,), generator=g),
    ]).to(device)
    # batching replicates the graph B times with an offset, as _batch_frames does
    edges_b = torch.cat([edge_index + i * N for i in range(B)], dim=1)
    v = torch.randn(N * B, 1, device=device)
    emb = torch.randn(N * B, args.embedding_dim, device=device)
    W = torch.randn(E, 1, device=device, requires_grad=True)
    din = (1 + args.embedding_dim) * (2 if args.model == "flyvis_conductance" else 1)
    gphi = GPhi(din, args.hidden, 1, device)
    return edge_index, edges_b, v, emb, W, gphi, din


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-neurons", type=int, default=13741)
    p.add_argument("--n-edges", type=int, default=434112)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--embedding-dim", type=int, default=2)
    p.add_argument("--model", default="flyvis_A",
                   choices=["flyvis_A", "flyvis_conductance"])
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index, edges_b, v, emb, W, gphi, din = build(args, device)
    N, E, B = args.n_neurons, args.n_edges, args.batch_size
    EB = E * B
    src, dst = edges_b

    props = torch.cuda.get_device_properties(0) if device.type == "cuda" else None
    if props:
        print(f"device: {props.name}  {props.multi_processor_count} SMs  "
              f"{props.total_memory/2**30:.0f} GiB")
    print(f"N={N} E={E} B={B} -> E*B={EB:,}   g_phi {din}->{args.hidden}->"
          f"{args.hidden}->1")
    act_mb = EB * args.hidden * 4 / 1e6
    print(f"one [E*B, {args.hidden}] fp32 activation = {act_mb:.0f} MB, "
          f"g_phi keeps 2\n")

    n_null = 0

    # ---- the pieces, in the order _compute_messages runs them ----
    def f_edge_W_idx():
        return torch.arange(EB, device=device) % (E + n_null)

    edge_W_idx = f_edge_W_idx()

    def f_gather_W():
        return W[edge_W_idx]

    def f_build_features():
        if args.model == "flyvis_conductance":
            return torch.cat([v[src], emb[src], v[dst], emb[dst]], dim=1)
        return torch.cat([v[src], emb[src]], dim=1)

    in_features = f_build_features()

    def f_gphi_fwd():
        with torch.no_grad():
            return gphi(in_features)

    edge_msg = (W[edge_W_idx] * gphi(in_features)).detach()

    def f_scatter():
        msg = torch.zeros(N * B, 1, device=device)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    def f_index_add():
        msg = torch.zeros(N * B, 1, device=device)
        msg.index_add_(0, dst, edge_msg)
        return msg

    def f_batch_edges():
        return torch.cat([edge_index + i * N for i in range(B)], dim=1)

    rows = [
        ("_batch_frames: rebuild batched_edges", f_batch_edges,
         edges_b.numel() * 8 / 1e6),
        ("edge_W_idx: arange + modulo", f_edge_W_idx, EB * 8 / 1e6),
        ("W[edge_W_idx] gather", f_gather_W, EB * 4 / 1e6),
        ("build in_features (gathers + cat)", f_build_features,
         EB * din * 4 / 1e6),
        ("g_phi forward (no grad)", f_gphi_fwd, act_mb * 2),
        ("scatter_add_ (+expand index)", f_scatter, EB * 4 / 1e6),
        ("index_add_", f_index_add, EB * 4 / 1e6),
    ]

    print(f"{'piece':<40}{'ms':>9}{'output MB':>11}")
    print("-" * 60)
    for name, fn, mb in rows:
        print(f"{name:<40}{timeit(fn, args.iters):>9.2f}{mb:>11.0f}")

    # ---- the whole thing, forward+backward, which is what training pays ----
    def full(cache_idx):
        idx = edge_W_idx if cache_idx else torch.arange(EB, device=device) % (E + n_null)
        if args.model == "flyvis_conductance":
            x = torch.cat([v[src], emb[src], v[dst], emb[dst]], dim=1)
        else:
            x = torch.cat([v[src], emb[src]], dim=1)
        out = gphi(x) ** 2
        em = W[idx] * out
        msg = torch.zeros(N * B, 1, device=device)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(em), em)
        msg.sum().backward()
        gphi.zero_grad(set_to_none=True)
        W.grad = None

    print("-" * 60)
    for label, cached in (("fwd+bwd, index rebuilt each call", False),
                          ("fwd+bwd, index cached", True)):
        print(f"{label:<40}{timeit(lambda c=cached: full(c), args.iters):>9.2f}")

    if device.type == "cuda":
        bw = {"NVIDIA RTX A6000": 768, "NVIDIA L4": 300, "NVIDIA A100": 1555}
        gbs = next((v for k, v in bw.items() if k in props.name), None)
        if gbs:
            traffic = 6 * act_mb / 1e3          # 2 activations, fwd write + bwd read + grad
            print(f"\nroofline: ~{traffic:.1f} GB of activation traffic / "
                  f"{gbs} GB/s = {traffic/gbs*1e3:.1f} ms of the above is "
                  f"[E*B, {args.hidden}] intermediates alone")


if __name__ == "__main__":
    main()
