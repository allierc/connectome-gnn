#!/usr/bin/env python
"""Prototype: g_phi as ONE Triton kernel, to find the ceiling before wiring it in.

The message MLP is 3->80->80->1 over E*B = 1.7M edges. PyTorch runs it as five
kernels and pushes two [E*B, 80] fp32 intermediates -- 556 MB each -- through HBM,
then reads them back in the backward. That is the same defect the MPM G2P kernel
had: intermediates materialised through global memory that never need to leave
registers.

This measures the forward-only ceiling. Nothing here is wired into NeuralGNN; if
the ceiling is not worth the backward kernel, that is a cheap answer.

H=80 is not a power of two, so the weights are zero-padded to 128. That is exact,
not approximate: relu(0 + 0) = 0, and the padded rows of the next weight matrix
are zero, so padded lanes contribute nothing at every layer.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gphi_fwd(X, SRC, EMB, WOUT, OUT,
              W0, B0, W1, B1, W2, B2,
              n_edges, D: tl.constexpr, H: tl.constexpr, E_DIM: tl.constexpr,
              SQUARE: tl.constexpr, BLOCK_M: tl.constexpr,
              PREC: tl.constexpr):
    """One program per BLOCK_M edges: gather, 3 layers, weight, all in registers."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_edges
    offs_h = tl.arange(0, H)
    # D is the PADDED input width (16): tl.arange needs a power of two and tl.dot
    # needs K >= 16, while the real g_phi input is 3 or 6 wide. The extra columns
    # are zero here and the matching rows of W0 are zero, so they contribute
    # nothing -- exact, not approximate.
    offs_d = tl.arange(0, D)

    src = tl.load(SRC + offs_m, mask=mask_m, other=0)

    # in_features = [v[src], emb[src]] -- gathered straight into registers, so the
    # [E*B, D] concat never exists.
    x = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    v = tl.load(X + src, mask=mask_m, other=0.0)
    x = tl.where(offs_d[None, :] == 0, v[:, None], x)
    for e in tl.static_range(E_DIM):
        col = tl.load(EMB + src * E_DIM + e, mask=mask_m, other=0.0)
        x = tl.where(offs_d[None, :] == e + 1, col[:, None], x)

    w0 = tl.load(W0 + offs_d[:, None] * H + offs_h[None, :])
    b0 = tl.load(B0 + offs_h)
    h = tl.maximum(tl.dot(x, w0, input_precision=PREC) + b0[None, :], 0.0)

    w1 = tl.load(W1 + offs_h[:, None] * H + offs_h[None, :])
    b1 = tl.load(B1 + offs_h)
    h = tl.maximum(tl.dot(h, w1, input_precision=PREC) + b1[None, :], 0.0)

    w2 = tl.load(W2 + offs_h)
    b2 = tl.load(B2)
    o = tl.sum(h * w2[None, :], axis=1) + b2
    if SQUARE:
        o = o * o
    o = o * tl.load(WOUT + offs_m, mask=mask_m, other=0.0)
    tl.store(OUT + offs_m, o, mask=mask_m)


def fused_gphi(v, emb, src, w_edge, params, square, block_m=64, prec="ieee"):
    (w0, b0), (w1, b1), (w2, b2) = params
    n = src.numel()
    H = w0.shape[1]
    out = torch.empty(n, device=v.device, dtype=v.dtype)
    _gphi_fwd[(triton.cdiv(n, block_m),)](
        v, src, emb, w_edge, out, w0, b0, w1, b1, w2, b2,
        n, D=w0.shape[0], H=H, E_DIM=emb.shape[1],
        SQUARE=square, BLOCK_M=block_m, PREC=prec, num_warps=4)
    return out


def pad_to(t, h, dim):
    """Zero-pad a weight/bias so H becomes a power of two. Exact, see module doc."""
    shape = list(t.shape)
    if shape[dim] >= h:
        return t
    shape[dim] = h - shape[dim]
    return torch.cat([t, torch.zeros(shape, device=t.device, dtype=t.dtype)], dim=dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-neurons", type=int, default=13741)
    p.add_argument("--n-edges", type=int, default=434112)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--embedding-dim", type=int, default=2)
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()

    dev = "cuda"
    N, E, B, H = a.n_neurons, a.n_edges, a.batch_size, a.hidden
    EB = E * B
    D = 1 + a.embedding_dim
    Hp = triton.next_power_of_2(H)

    torch.manual_seed(0)
    src = torch.randint(0, N * B, (EB,), device=dev, dtype=torch.int32)
    v = torch.randn(N * B, device=dev)
    emb = torch.randn(N * B, a.embedding_dim, device=dev)
    w_edge = torch.randn(EB, device=dev)

    l0 = nn.Linear(D, H, device=dev)
    l1 = nn.Linear(H, H, device=dev)
    l2 = nn.Linear(H, 1, device=dev)

    def torch_ref():
        x = torch.cat([v[src.long()].unsqueeze(1), emb[src.long()]], dim=1)
        o = l2(F.relu(l1(F.relu(l0(x)))))
        return (w_edge.unsqueeze(1) * (o ** 2)).squeeze(1)

    D_PAD = 16                      # tl.dot needs K >= 16; rows D..15 are zero
    params = (
        (pad_to(pad_to(l0.weight.t().contiguous(), D_PAD, 0), Hp, 1),
         pad_to(l0.bias, Hp, 0)),
        (pad_to(pad_to(l1.weight.t().contiguous(), Hp, 0), Hp, 1),
         pad_to(l1.bias, Hp, 0)),
        (pad_to(l2.weight.squeeze(0), Hp, 0), l2.bias),
    )

    def timeit(fn, n, w=8):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}  {props.multi_processor_count} SMs")
    print(f"E*B={EB:,}  g_phi {D}->{H}->{H}->1  (padded to {Hp})\n")

    with torch.no_grad():
        ref = torch_ref()
        t_ref = timeit(torch_ref, a.iters)
        print(f"{'variant':<34}{'fwd ms':>9}{'x':>7}{'max|d|':>12}{'rel':>11}")
        print(f"{'torch (nn.Linear, eager)':<34}{t_ref:>9.2f}{1.0:>7.2f}"
              f"{0.0:>12.2e}{0.0:>11.2e}")
        for prec in ("ieee", "tf32"):
            for bm in (64, 128, 256):
                try:
                    got = fused_gphi(v, emb, src, w_edge, params, True, bm, prec)
                    d = (got - ref).abs().max().item()
                    rel = d / ref.abs().max().item()
                    ms = timeit(lambda bm=bm, p=prec: fused_gphi(
                        v, emb, src, w_edge, params, True, bm, p), a.iters)
                    label = f"triton {prec}, BLOCK_M={bm}"
                    print(f"{label:<34}{ms:>9.2f}{t_ref/ms:>7.2f}"
                          f"{d:>12.2e}{rel:>11.2e}")
                except Exception as e:
                    print(f"{'triton ' + prec + ', BLOCK_M=' + str(bm):<34}  "
                          f"FAILED: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
