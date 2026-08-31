#!/usr/bin/env python
"""Prototype: the FUSED BACKWARD for g_phi, which is where the work actually is.

The forward (prototype_fused_gphi.py) was easy -- one kernel, everything in
registers, 13.5x. This is the other half, and it is a different problem, for the
reason the zap-inr Warp note states as "the launch is entirely about where the
atomics land".

WHERE OURS LAND, and why it is worse than the splat case. Per edge the backward
produces three families of gradient, with wildly different address counts:

    dW_edge[e]           E*B = 1.7e6 addresses, one writer each   -- free
    dv[src], demb[src]   N*B = 55e3 addresses, ~32 writers each   -- mild
    dW0 dW1 dW2 db*      ~16e3 addresses, 1.7e6 writers each      -- fatal

The MLP weight gradients are indexed by NOTHING. Every one of the 1.7M edges
reduces into the same ~16k words. In the splat case those words were at least
per-splat (600e3 addresses); here a naive per-edge atomic serialises the entire
kernel onto 16k addresses. So the launch cannot be "one program per chunk of
edges" with atomics at the end of each -- there are 27k such chunks at BLOCK_M=64.

THE FIX is the same one in shape, applied harder: a GRID-STRIDE loop over a
PERSISTENT grid. Launch exactly as many programs as the GPU can hold resident,
have each walk a strided slice of the whole edge list accumulating dW in
registers, and issue its atomics ONCE at the end. Contention per weight word
drops from 27k to the number of programs -- 116 on an L4, 216 on an A6000.

WHAT THIS COSTS, and it is the thing to look at: dW1 is [H, H] and has to live in
registers for the whole loop. At H padded to 128 that is a 64 KiB tile per
program. An SM has 256 KiB of registers, so the accumulator alone caps occupancy
at ~1 block/SM before anything else is allocated. That trade -- contention
against occupancy -- is the same one the Warp note's Bt sweep found, and it is
the reason this kernel is not obviously a win.

Nothing here is wired into NeuralGNN. It answers two questions: can the backward
be written, and what does the register pressure do to it.

RESULT: NEGATIVE, AND THIS KERNEL IS ALSO STILL WRONG. Do not use it. Measured on
an A6000 against torch autograd's 37.53 ms for the same backward:

    HK=32  BLOCK_M=16  168 programs    783 ms   0.05x   rel err 1.7e-1 on demb
    HK=16  BLOCK_M=16  168 programs    354 ms   0.11x   rel err 1.7e-1 on demb
    HK=32  BLOCK_M=32                  refused: 109 KiB shared, limit 100 KiB
    HK=--  BLOCK_M=32  (untiled dW1)   refused: 184 KiB shared, limit 100 KiB

Two things are wrong with it, and they are the same thing. The row-slice of h0
that feeds dW1 is built with a where/sum over a [HK, BLOCK_M, H] intermediate
instead of a real transpose: that is the 1.7e-1 error AND most of the runtime.
Replacing it is the next step if this is ever picked up again.

That HK=16 is twice as fast as HK=32 says the kernel is still far on the
occupancy-starved side of the trade -- the interior optimum the zap-inr note
finds for Bt has not even been approached here.

WHY IT WAS PARKED ANYWAY. mlp_precision: bf16 already gets 1.79x on an l4 for
about fifteen lines, and it needs no kernel to maintain. A perfect version of
this kernel is worth maybe another 1.9x on top of that, cannot be bit-identical
either (fp32 tl.dot runs at 186 ms, 9x slower than torch, so tf32 is forced), and
would hardcode g_phi's shape -- 3->80->80->1 with the square -- so hidden_dim,
n_layers, the 6-wide conductance input and the noise-probe columns would each
need kernel work.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gphi_bwd(SRC, V, EMB, WEDGE, GOUT,
              W0, B0, W1, B1, W2, B2,
              DW0, DB0, DW1, DB1, DW2, DB2,
              DV, DEMB, DWEDGE,
              n_edges, n_prog,
              D: tl.constexpr, H: tl.constexpr, E_DIM: tl.constexpr,
              SQUARE: tl.constexpr, BLOCK_M: tl.constexpr, HK: tl.constexpr,
              PREC: tl.constexpr):
    # 2D grid: (edge-slice, dW1 row-tile). A [H, H] accumulator is 64 KiB at
    # H=128 and does not fit in 100 KiB of shared memory alongside the working
    # [BLOCK_M, H] tiles -- measured, 184 KiB required. Splitting dW1's ROWS
    # across H/HK programs cuts it to [HK, H]. The price is that each row-tile
    # recomputes the same forward, so this is the contention-vs-occupancy trade
    # the zap-inr note's Bt sweep runs into, in its other form.
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    first_k = pid_k == 0
    offs_h = tl.arange(0, H)
    offs_k = pid_k * HK + tl.arange(0, HK)
    offs_d = tl.arange(0, D)

    w0 = tl.load(W0 + offs_d[:, None] * H + offs_h[None, :])
    b0 = tl.load(B0 + offs_h)
    w1 = tl.load(W1 + offs_h[:, None] * H + offs_h[None, :])
    b1 = tl.load(B1 + offs_h)
    w2 = tl.load(W2 + offs_h)
    b2 = tl.load(B2)

    # The whole point: these live here, across every chunk this program walks,
    # and are flushed to global memory exactly once.
    acc_w1 = tl.zeros((HK, H), dtype=tl.float32)
    acc_w0 = tl.zeros((D, H), dtype=tl.float32)
    acc_w2 = tl.zeros((H,), dtype=tl.float32)
    acc_b0 = tl.zeros((H,), dtype=tl.float32)
    acc_b1 = tl.zeros((H,), dtype=tl.float32)
    acc_b2 = tl.zeros((1,), dtype=tl.float32)

    n_chunk = tl.cdiv(n_edges, BLOCK_M)
    for c in range(pid, n_chunk, n_prog):
        offs_m = c * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_edges
        src = tl.load(SRC + offs_m, mask=mask_m, other=0)

        # ---- recompute the forward. Cheaper than having stored it: the two
        # [E*B, H] activations are 556 MB each, and we are memory-bound.
        x = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        v = tl.load(V + src, mask=mask_m, other=0.0)
        x = tl.where(offs_d[None, :] == 0, v[:, None], x)
        for e in tl.static_range(E_DIM):
            col = tl.load(EMB + src * E_DIM + e, mask=mask_m, other=0.0)
            x = tl.where(offs_d[None, :] == e + 1, col[:, None], x)

        z0 = tl.dot(x, w0, input_precision=PREC) + b0[None, :]
        h0 = tl.maximum(z0, 0.0)
        z1 = tl.dot(h0, w1, input_precision=PREC) + b1[None, :]
        h1 = tl.maximum(z1, 0.0)
        o = tl.sum(h1 * w2[None, :], axis=1) + b2

        we = tl.load(WEDGE + offs_m, mask=mask_m, other=0.0)
        g = tl.load(GOUT + offs_m, mask=mask_m, other=0.0)

        # ---- reverse
        p = o * o if SQUARE else o
        if first_k:
            tl.atomic_add(DWEDGE + offs_m, g * p, mask=mask_m)  # one writer per addr
        do = g * we
        if SQUARE:
            do = do * 2.0 * o
        do = tl.where(mask_m, do, 0.0)

        dh1 = tl.where(z1 > 0.0, do[:, None] * w2[None, :], 0.0)

        # dW1[j, k] = sum_e h0[e, j] dh1[e, k] -- this program owns rows j in
        # [pid_k*HK, (pid_k+1)*HK), so it slices h0 rather than transposing all
        # of it.
        h0k = tl.sum(tl.where(offs_h[None, :] == offs_k[:, None, None],
                              h0[None, :, :], 0.0), axis=2)
        acc_w1 += tl.dot(h0k, dh1, input_precision=PREC)

        # Everything below is row-tile-independent, so only tile 0 accumulates it
        # -- otherwise every gradient would be summed H/HK times.
        if first_k:
            acc_b2 += tl.sum(do)
            acc_w2 += tl.sum(do[:, None] * h1, axis=0)
            acc_b1 += tl.sum(dh1, axis=0)

            dh0 = tl.dot(dh1, tl.trans(w1), input_precision=PREC)
            dh0 = tl.where(z0 > 0.0, dh0, 0.0)
            acc_b0 += tl.sum(dh0, axis=0)
            acc_w0 += tl.dot(tl.trans(x), dh0, input_precision=PREC)

            dx = tl.dot(dh0, tl.trans(w0), input_precision=PREC)
            # node-indexed: ~32 writers per address, which is fine
            tl.atomic_add(DV + src,
                          tl.sum(tl.where(offs_d[None, :] == 0, dx, 0.0), axis=1),
                          mask=mask_m)
            for e in tl.static_range(E_DIM):
                col = tl.sum(tl.where(offs_d[None, :] == e + 1, dx, 0.0), axis=1)
                tl.atomic_add(DEMB + src * E_DIM + e, col, mask=mask_m)

    # ---- ONE flush per program, not per chunk. This is the design.
    tl.atomic_add(DW1 + offs_k[:, None] * H + offs_h[None, :], acc_w1)
    if first_k:
        tl.atomic_add(DW0 + offs_d[:, None] * H + offs_h[None, :], acc_w0)
        tl.atomic_add(DW2 + offs_h, acc_w2)
        tl.atomic_add(DB0 + offs_h, acc_b0)
        tl.atomic_add(DB1 + offs_h, acc_b1)
        tl.atomic_add(DB2 + tl.arange(0, 1), acc_b2)


def fused_bwd(v, emb, src, w_edge, params, g_out, square,
              block_m=32, n_prog=None, prec="tf32", num_warps=8, hk=32):
    (w0, b0), (w1, b1), (w2, b2) = params
    n, H, D = src.numel(), w0.shape[1], w0.shape[0]
    dev = v.device
    if n_prog is None:
        n_prog = torch.cuda.get_device_properties(dev).multi_processor_count * 2
    z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)
    dw0, db0, dw1, db1, dw2, db2 = z(D, H), z(H), z(H, H), z(H), z(H), z(1)
    dv, demb, dwe = z(v.shape), z(emb.shape), z(n)
    _gphi_bwd[(n_prog, H // hk)](
        src, v, emb, w_edge, g_out, w0, b0, w1, b1, w2, b2,
        dw0, db0, dw1, db1, dw2, db2, dv, demb, dwe,
        n, n_prog, D=D, H=H, E_DIM=emb.shape[1], SQUARE=square,
        BLOCK_M=block_m, HK=hk, PREC=prec, num_warps=num_warps)
    return dw0, db0, dw1, db1, dw2, db2, dv, demb, dwe


def pad_to(t, h, dim):
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
    p.add_argument("--iters", type=int, default=10)
    a = p.parse_args()

    dev = "cuda"
    N, E, B, H = a.n_neurons, a.n_edges, a.batch_size, a.hidden
    EB, D = E * B, 1 + a.embedding_dim
    Hp, Dp = triton.next_power_of_2(H), 16

    torch.manual_seed(0)
    src = torch.randint(0, N * B, (EB,), device=dev, dtype=torch.int32)
    v = torch.randn(N * B, device=dev, requires_grad=True)
    emb = torch.randn(N * B, a.embedding_dim, device=dev, requires_grad=True)
    w_edge = torch.randn(EB, device=dev, requires_grad=True)
    g_out = torch.randn(EB, device=dev)          # a real cotangent, not ones
    l0 = nn.Linear(D, H, device=dev)
    l1 = nn.Linear(H, H, device=dev)
    l2 = nn.Linear(H, 1, device=dev)

    def torch_bwd():
        for t in (v, emb, w_edge):
            t.grad = None
        for m in (l0, l1, l2):
            m.zero_grad(set_to_none=True)
        s = src.long()
        x = torch.cat([v[s].unsqueeze(1), emb[s]], dim=1)
        o = l2(F.relu(l1(F.relu(l0(x))))).squeeze(1)
        (w_edge * o ** 2 * g_out).sum().backward()

    params = (
        (pad_to(pad_to(l0.weight.t().contiguous(), Dp, 0), Hp, 1), pad_to(l0.bias, Hp, 0)),
        (pad_to(pad_to(l1.weight.t().contiguous(), Hp, 0), Hp, 1), pad_to(l1.bias, Hp, 0)),
        (pad_to(l2.weight.squeeze(0), Hp, 0), l2.bias),
    )

    def timeit(fn, n, w=3):
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
    print(f"E*B={EB:,}  g_phi {D}->{H}->{H}->1 (padded {Dp}->{Hp})")
    print(f"dW1 accumulator, untiled: [{Hp},{Hp}] fp32 = {Hp*Hp*4/1024:.0f} KiB "
          f"-- does not fit; tiled by HK below\n")

    torch_bwd()
    ref = {"W0": l0.weight.grad.t().contiguous(), "b0": l0.bias.grad,
           "W1": l1.weight.grad.t().contiguous(), "b1": l1.bias.grad,
           "W2": l2.weight.grad.squeeze(0), "b2": l2.bias.grad,
           "v": v.grad, "emb": emb.grad, "W_edge": w_edge.grad}
    t_ref = timeit(torch_bwd, a.iters)
    print(f"torch autograd backward: {t_ref:.2f} ms\n")

    print(f"{'config':<34}{'bwd ms':>9}{'x':>7}{'worst rel err':>15}{'on':>8}")
    for bm, npg, nw, hk in ((16, None, 4, 32), (16, None, 4, 16),
                            (32, None, 8, 32), (32, None, 8, 16),
                            (16, props.multi_processor_count, 4, 32)):
        try:
            out = fused_bwd(v.detach(), emb.detach(), src, w_edge.detach(),
                            params, g_out, True, bm, npg, "tf32", nw, hk)
            got = dict(zip(("W0", "b0", "W1", "b1", "W2", "b2",
                            "v", "emb", "W_edge"), out))
            worst, where = 0.0, ""
            for k, r in ref.items():
                gk = got[k]
                gk = gk[:r.shape[0]] if gk.dim() == 1 else gk[:r.shape[0], :r.shape[1]]
                den = r.abs().max().clamp_min(1e-20)
                rel = ((gk - r).abs().max() / den).item()
                if rel > worst:
                    worst, where = rel, k
            ms = timeit(lambda: fused_bwd(v.detach(), emb.detach(), src,
                                          w_edge.detach(), params, g_out, True,
                                          bm, npg, "tf32", nw, hk), a.iters)
            label = (f"BM={bm} HK={hk} "
                     f"prog={npg or 2*props.multi_processor_count} w={nw}")
            print(f"{label:<34}{ms:>9.2f}{t_ref/ms:>7.2f}{worst:>15.2e}{where:>8}")
        except Exception as e:
            print(f"{'BLOCK_M=' + str(bm):<34}  FAILED: {type(e).__name__}: "
                  f"{str(e)[:70]}")


if __name__ == "__main__":
    main()
