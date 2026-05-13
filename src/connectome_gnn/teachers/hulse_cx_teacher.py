"""Hulse Model A teacher — connectome-constrained CX RNN trained on path integration.

Ref: Hulse, Aneesh, Romani, Jayaraman, Hermundstad (Janelia 2026 draft),
     docs/Hidden_Symmetries.pdf Methods pp. 12-15, Eqs. 1-11.

Architecture (verbatim from the Methods):

    tau * dh_j/dt = -h_j
                    + sum_k W^rec_{jk} * sigmoid(h_k)
                    + sum_l W^in_{j,l} * u_l
                    + b_j
    y_hat_i = sum_j W^out_{ij} * sigmoid(h_j) + b^out_i,   i=1,2

    u[t] = [omega[t], cos(theta_hd[0]) * 1_{t=0}, sin(theta_hd[0]) * 1_{t=0}]

Loss (Eqs. 4, 10, 11):

    L_mse  = (1 / 2T) * sum_t sum_i (y_hat_i[t] - y_i[t])^2
    L_cosd = (lambda / |B|) * sum_{(p,q) in B}
             (1 - cos(W^rec_{pq}, W^con_{pq}))           # per-type-pair
    L_norm = (lambda / |B|) * sum_{(p,q) in B}
             max(0, kappa - <|W^rec_{pq}|>)^2

Default training (Hulse Methods):
    N = 156 (v1: 152 from Beiran hemibrain; ER6 not included)
    tau = 0.1 s, dt = 0.01 s, T = 100 steps (1 s/trial)
    200k trials in 2000 batches of 100
    10 epochs, lr 5e-3 -> 5e-4 at epoch 5
    Adam + BPTT
    tau_corr = 0.12 s, sigma_omega = 40 deg/s
    20% standing pauses, exp(mean=2s, cap=8s)
    Trainable: W^in, W^rec, b, W^out, b^out

Deviation from Hulse (v1, documented in the plan):
    - Beiran's hemibrain CX: 152 neurons, 6 types
      (EPG / EPGt / PEN_a / PEN_b / Delta7 / PEG; Beiran lumps PEN_a+PEN_b
       into 'PEN' so effectively 4 types in his loader). No ER6.
    - Hulse: 156 neurons, 7 types (adds ER6).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class HulseCxRNN(nn.Module):
    """Connectome-constrained CX RNN (Hulse Model A).

    Recurrent weights are parameterised as the unconstrained tensor
    `W_rec`. The connectome-derived template `W_con` (with signs baked in
    via the Beiran sign mask `mwrec`) is held as a non-trainable buffer
    and used by the cosine-distance regulariser.

    Parameters that are trainable: W_rec, W_in, b, W_out, b_out
    (matching Hulse Methods 'Network training' paragraph).
    """

    def __init__(
        self,
        n_units: int,
        n_input: int = 3,
        n_output: int = 2,
        tau: float = 0.1,
        dt: float = 0.01,
        W_con: torch.Tensor | None = None,
        type_pair_blocks: dict | None = None,
        init_scale: float = 1.0 / 100.0,
        rng_seed: int = 0,
    ):
        """
        Args:
            n_units: number of recurrent units N (152 with Beiran hemibrain).
            n_input: input-channel count (3 in Hulse: omega + cos/sin of
                initial heading).
            n_output: output-channel count (2 in Hulse: cos/sin of decoded HD).
            tau: membrane time constant in seconds.
            dt: Euler integration step in seconds.
            W_con: (N, N) connectome template with signs. If provided,
                used as the W_rec initial value AND as the cosine-distance
                regulariser target. If None, W_rec is initialised from
                N(0, init_scale**2).
            type_pair_blocks: dict mapping (type_pre, type_post) -> (mask, name).
                `mask` is a (N, N) bool tensor selecting the type-pair block.
                Used by L_cosd and L_norm. If None, the regularisers fall
                back to a single global block.
            init_scale: stddev of the N(0, init_scale**2) initialisation
                used when W_con is None or for W_in.  Hulse uses d=100
                (init_scale = 1/100).
            rng_seed: torch RNG seed for the init.
        """
        super().__init__()
        self.n_units = int(n_units)
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.tau = float(tau)
        self.dt = float(dt)

        gen = torch.Generator()
        gen.manual_seed(int(rng_seed))

        # --- Recurrent weights -------------------------------------------
        if W_con is not None:
            if W_con.shape != (n_units, n_units):
                raise ValueError(
                    f"W_con shape {tuple(W_con.shape)} != ({n_units}, {n_units})"
                )
            # Hulse Methods p. 13:
            #   "the recurrent weight matrix contains 156 units and was
            #    initialized and regularized using a processed version of
            #    the raw synaptic connectivity matrix"
            # We initialise W_rec from W_con directly so the network starts
            # in the connectome regime.
            W_rec_init = W_con.clone().to(torch.float32)
            self.register_buffer("W_con", W_con.clone().to(torch.float32))
        else:
            W_rec_init = torch.randn(
                n_units, n_units, generator=gen, dtype=torch.float32
            ) * init_scale
            self.register_buffer(
                "W_con", torch.zeros(n_units, n_units, dtype=torch.float32)
            )

        self.W_rec = nn.Parameter(W_rec_init)

        # --- Input / output weights and biases ---------------------------
        # Hulse Methods p. 13: W_in element from N(0, 1/d^2), d=100.
        self.W_in = nn.Parameter(
            torch.randn(n_units, n_input, generator=gen, dtype=torch.float32)
            * init_scale
        )
        # Recurrent biases initialised to 1 (Hulse Methods p. 13).
        self.b = nn.Parameter(torch.ones(n_units, dtype=torch.float32))

        # PyTorch Kaiming default for output weights (matches Hulse).
        self.W_out = nn.Parameter(torch.empty(n_output, n_units, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.W_out, a=math.sqrt(5))
        self.b_out = nn.Parameter(torch.zeros(n_output, dtype=torch.float32))

        # --- Cosine-distance regulariser blocks --------------------------
        if type_pair_blocks is None:
            self._block_names: list[str] = []
            self._block_masks: list[torch.Tensor] = []
        else:
            self._block_names = list(type_pair_blocks.keys())
            self._block_masks = [
                type_pair_blocks[k].to(torch.bool) for k in self._block_names
            ]
            # Persist them as buffers so .to(device) moves them
            for i, m in enumerate(self._block_masks):
                self.register_buffer(f"_block_mask_{i}", m, persistent=False)

    # --- Forward path ---------------------------------------------------

    def forward(
        self,
        u: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network for T timesteps over a batch.

        Args:
            u: (B, T, n_input) input stream.
            h0: (B, N) initial subthreshold activity. If None, zeros.

        Returns:
            y_hat: (B, T, n_output) readout.
            h:     (B, T, n_units) subthreshold activity (for diagnostics).
        """
        B, T, _ = u.shape
        N = self.n_units

        if h0 is None:
            h = torch.zeros(B, N, dtype=u.dtype, device=u.device)
        else:
            h = h0

        dt_over_tau = self.dt / self.tau
        h_buf = torch.empty(B, T, N, dtype=u.dtype, device=u.device)

        for t in range(T):
            r = torch.sigmoid(h)                          # (B, N)
            rec = r @ self.W_rec.t()                      # (B, N)
            inp = u[:, t, :] @ self.W_in.t()              # (B, N)
            dh = -h + rec + inp + self.b
            h = h + dt_over_tau * dh
            h_buf[:, t, :] = h

        r_full = torch.sigmoid(h_buf)                     # (B, T, N)
        y_hat = r_full @ self.W_out.t() + self.b_out      # (B, T, n_output)
        return y_hat, h_buf

    # --- Regularisers ---------------------------------------------------

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        """Hulse Eq. 10: cosine-distance regulariser against the connectome
        template W_con, computed per (presyn-type, postsyn-type) block."""
        if not self._block_names or torch.all(self.W_con == 0):
            return self.W_rec.new_zeros(())
        total = self.W_rec.new_zeros(())
        eps = 1e-12
        for i in range(len(self._block_names)):
            mask = getattr(self, f"_block_mask_{i}")
            w_rec_b = self.W_rec[mask]
            w_con_b = self.W_con[mask]
            if w_con_b.abs().sum() < eps:
                continue
            num = (w_rec_b * w_con_b).sum()
            den = w_rec_b.norm() * w_con_b.norm() + eps
            total = total + (1.0 - num / den)
        return lam * total / max(len(self._block_names), 1)

    def loss_norm_floor(self, lam: float = 1.0, kappa: float = 0.05) -> torch.Tensor:
        """Hulse Eq. 11: soft lower bound on mean |W| per type-pair block."""
        if not self._block_names:
            return self.W_rec.new_zeros(())
        total = self.W_rec.new_zeros(())
        for i in range(len(self._block_names)):
            mask = getattr(self, f"_block_mask_{i}")
            mean_abs = self.W_rec[mask].abs().mean()
            slack = F.relu(kappa - mean_abs)
            total = total + slack.pow(2)
        return lam * total / max(len(self._block_names), 1)


# ---------------------------------------------------------------------------
# Path-integration data generation
# ---------------------------------------------------------------------------


@dataclass
class PathIntegrationBatch:
    """One batch of path-integration training data.

    Shapes:
        u:        (B, T, 3) — [omega(t), cos(theta0)*1_{t=0}, sin(theta0)*1_{t=0}]
        y:        (B, T, 2) — [cos(theta_hd(t)), sin(theta_hd(t))]
        theta_hd: (B, T)    — ground-truth heading in radians (for diagnostics)
        is_stop:  (B, T)    — 1 during standing pauses, 0 otherwise
    """
    u: torch.Tensor
    y: torch.Tensor
    theta_hd: torch.Tensor
    is_stop: torch.Tensor


def generate_path_integration_batch(
    batch_size: int,
    n_steps: int,
    *,
    dt: float = 0.01,
    tau_corr: float = 0.12,
    sigma_omega_deg: float = 40.0,
    stop_fraction: float = 0.20,
    stop_mean_s: float = 2.0,
    stop_max_s: float = 8.0,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
) -> PathIntegrationBatch:
    """Generate a path-integration training batch (Hulse Methods Eqs. 5-7).

    Args:
        batch_size: number of trials B.
        n_steps:    number of timesteps T (Hulse default 100).
        dt:         step size in seconds (Hulse: 0.01).
        tau_corr:   OU autocorrelation time (Hulse: 0.12 s).
        sigma_omega_deg: stationary stddev of omega (Hulse: 40 deg/s).
        stop_fraction:   approximate fraction of trial time spent stationary.
        stop_mean_s:     mean stop duration (Hulse: 2 s, exponential).
        stop_max_s:      cap on stop duration (Hulse: 8 s).
        device, rng:     where to allocate / what RNG to use.

    Returns:
        PathIntegrationBatch on `device`.
    """
    if rng is None:
        rng = np.random.default_rng()

    B = int(batch_size)
    T = int(n_steps)
    alpha = 1.0 / tau_corr
    sigma = sigma_omega_deg * math.sqrt(2.0 * alpha)
    sqrt_dt = math.sqrt(dt)
    sigma_step = sigma * sqrt_dt

    omega = np.zeros((B, T), dtype=np.float32)
    eta = rng.standard_normal(size=(B, T)).astype(np.float32)

    # OU integration (Eq. 5). Use multiplicative form for stability.
    decay = 1.0 - alpha * dt
    for t in range(1, T):
        omega[:, t] = decay * omega[:, t - 1] + sigma_step * eta[:, t]

    # Standing pauses: insert exponential-duration stops in each trial.
    # Average fraction is roughly `stop_fraction`, capped per-stop at stop_max_s.
    is_stop = np.zeros((B, T), dtype=np.float32)
    if stop_fraction > 0.0:
        mean_steps = stop_mean_s / dt
        max_steps = int(stop_max_s / dt)
        for b in range(B):
            covered = 0
            target = int(stop_fraction * T)
            attempts = 0
            while covered < target and attempts < 100:
                attempts += 1
                start = rng.integers(0, T)
                length = min(
                    max_steps,
                    int(rng.exponential(mean_steps)),
                    T - start,
                )
                if length <= 0:
                    continue
                end = start + length
                already = int(is_stop[b, start:end].sum())
                is_stop[b, start:end] = 1.0
                covered += length - already
        omega = omega * (1.0 - is_stop)  # zero velocity during stops

    # Integrate to heading (Eq. 6).
    theta0 = rng.uniform(0.0, 2.0 * math.pi, size=B).astype(np.float32)
    omega_rad = np.deg2rad(omega)
    theta_hd = theta0[:, None] + np.cumsum(omega_rad, axis=1) * dt
    theta_hd[:, 0] = theta0  # ensure t=0 has the initial heading

    # Hold theta_hd constant during stops (replace cumsum increment with 0 above).

    cos_t = np.cos(theta_hd).astype(np.float32)
    sin_t = np.sin(theta_hd).astype(np.float32)

    # Input vector (Eq. 7): [omega, cos(theta0)*1_{t=0}, sin(theta0)*1_{t=0}].
    u = np.zeros((B, T, 3), dtype=np.float32)
    u[:, :, 0] = omega  # in deg/s, matching Hulse
    u[:, 0, 1] = np.cos(theta0)
    u[:, 0, 2] = np.sin(theta0)

    y = np.stack([cos_t, sin_t], axis=-1).astype(np.float32)

    return PathIntegrationBatch(
        u=torch.from_numpy(u).to(device),
        y=torch.from_numpy(y).to(device),
        theta_hd=torch.from_numpy(theta_hd).to(device),
        is_stop=torch.from_numpy(is_stop).to(device),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _build_type_pair_blocks(
    neuron_types: np.ndarray,
    type_names: list[str],
    W_con: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Build (post-type, pre-type) -> bool-mask blocks for the cos-distance reg.

    Only include blocks whose `W_con` block has at least one non-zero entry,
    matching the definition of set B in Hulse Eq. 10.
    """
    blocks: dict[str, torch.Tensor] = {}
    nt = np.asarray(neuron_types).astype(np.int64)
    n = nt.size
    unique = sorted(set(nt.tolist()))
    for q in unique:
        post_mask = nt == q  # (N,)
        for p in unique:
            pre_mask = nt == p  # (N,)
            block = np.outer(post_mask, pre_mask)
            if block.sum() == 0:
                continue
            sub = W_con[block]
            if np.abs(sub).sum() < 1e-12:
                continue
            tp_name = f"{type_names[int(p)]}->{type_names[int(q)]}"
            blocks[tp_name] = torch.from_numpy(block.astype(np.bool_))
    return blocks


def train_hulse_cx_teacher(
    *,
    connconstr_datapath: str,
    output_path: str,
    n_trials: int = 200_000,
    n_steps: int | list[int] = 100,
    batch_size: int = 100,
    n_epochs: int = 5,
    lr_init: float | list[float] = 1e-3,
    lr_drop_epoch: int = 3,
    lr_drop_factor: float = 0.1,
    lambda_cos: float = 1.0,
    lambda_norm: float = 1.0,
    kappa_norm: float = 0.05,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_interval: int = 50,
    eval_interval: int = 500,
    save_every_epoch: bool = True,
) -> dict:
    """Train a Hulse Model A CX teacher RNN and save the checkpoint.

    Args follow the Hulse Methods. `n_trials` is divided into batches of
    `batch_size`; `n_trials // batch_size` is the number of optimisation
    steps per epoch.

    Writes the best-loss checkpoint to `output_path` and (if
    save_every_epoch) one checkpoint per epoch to `output_path` with a
    `_epoch{E}` suffix.

    Returns a dict with training stats.
    """
    import os
    from connectome_gnn.generators.connconstr_data import load_drosophila_cx_connectome

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    # --- Load hemibrain connectome ----------------------------------------
    cx = load_drosophila_cx_connectome(connconstr_datapath)
    N = int(cx["N"])
    W_con_np = cx["J_effective"].astype(np.float32)        # signs baked in
    neuron_types = np.asarray(cx["neuron_types"]).astype(np.int64)
    type_names = list(cx["type_names"])
    # EPG indices + glomerular mapping for the live bump-FWHM metric.
    n_epg = int(cx["n_epg"])
    epg_indices = np.arange(n_epg, dtype=np.int64)
    epg_glom_ix = np.asarray(cx["epg_ix"], dtype=np.int64)

    # Hulse's normalisation: scale W_con so the typical element is O(1).
    # The Beiran-derived J_effective is already scaled by spectral radius
    # (0.9 / max(Re(eig))), so we don't rescale further here. The
    # cos-distance regulariser is scale-invariant anyway.
    type_pair_blocks = _build_type_pair_blocks(neuron_types, type_names, W_con_np)

    W_con_t = torch.from_numpy(W_con_np)

    net = HulseCxRNN(
        n_units=N,
        n_input=3,
        n_output=2,
        tau=0.1,
        dt=0.01,
        W_con=W_con_t,
        type_pair_blocks=type_pair_blocks,
        rng_seed=seed,
    ).to(device)

    # Per-epoch lr: float -> use MultiStepLR with the milestone drop;
    #                list  -> manual update at the start of each epoch.
    if isinstance(lr_init, (int, float)):
        lr_schedule = None
        _lr_init = float(lr_init)
    else:
        lr_schedule = [float(x) for x in lr_init]
        if len(lr_schedule) < n_epochs:
            lr_schedule = lr_schedule + [lr_schedule[-1]] * (n_epochs - len(lr_schedule))
        lr_schedule = lr_schedule[:n_epochs]
        _lr_init = lr_schedule[0]
        print(f"[hulse_cx] lr schedule: {lr_schedule}")

    opt = torch.optim.Adam(net.parameters(), lr=_lr_init)
    if lr_schedule is None:
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_drop_epoch], gamma=lr_drop_factor
        )
    else:
        sched = None

    steps_per_epoch = max(1, n_trials // batch_size)
    history = {"loss": [], "mse": [], "cosd": [], "norm": [], "epoch": [], "pi_acc": []}
    best_loss = float("inf")

    # Per-epoch trial length: int -> constant; list -> curriculum (pads with last).
    if isinstance(n_steps, int):
        n_steps_schedule = [int(n_steps)] * n_epochs
    else:
        _list = [int(s) for s in n_steps]
        if len(_list) < n_epochs:
            _list = _list + [_list[-1]] * (n_epochs - len(_list))
        n_steps_schedule = _list[:n_epochs]
    print(f"[hulse_cx] n_steps schedule: {n_steps_schedule}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    t0 = time.time()
    # EMA buffers for a smooth running display
    ema_loss = ema_mse = ema_cosd = ema_norm = None
    ema_alpha = 0.05
    last_pi_acc = float("nan")
    last_fwhm = float("nan")

    def _ema(prev, new):
        return float(new) if prev is None else (1 - ema_alpha) * prev + ema_alpha * float(new)

    for epoch in range(1, n_epochs + 1):
        n_steps_epoch = n_steps_schedule[epoch - 1]
        # Apply per-epoch lr if a schedule was provided.
        if lr_schedule is not None:
            for g in opt.param_groups:
                g["lr"] = lr_schedule[epoch - 1]
        pbar = trange(
            steps_per_epoch,
            ncols=200,
            desc=f"hulse_cx epoch {epoch}/{n_epochs} (T={n_steps_epoch})",
            leave=True,
        )
        for step1 in pbar:
            step = step1 + 1
            batch = generate_path_integration_batch(
                batch_size, n_steps_epoch, device=device, rng=rng
            )
            y_hat, _ = net(batch.u)
            mse = F.mse_loss(y_hat, batch.y)
            cosd = net.loss_cos_distance(lambda_cos)
            norm = net.loss_norm_floor(lambda_norm, kappa_norm)
            loss = mse + cosd + norm

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ema_loss = _ema(ema_loss, loss.item())
            ema_mse = _ema(ema_mse, mse.item())
            ema_cosd = _ema(ema_cosd, cosd.item())
            ema_norm = _ema(ema_norm, norm.item())

            if step % log_interval == 0:
                history["loss"].append(float(loss.item()))
                history["mse"].append(float(mse.item()))
                history["cosd"].append(float(cosd.item()))
                history["norm"].append(float(norm.item()))
                history["epoch"].append(epoch + step / steps_per_epoch)

            if step % eval_interval == 0 or step == steps_per_epoch:
                with torch.no_grad():
                    # Eval at the *current* trial length so pi_acc tracks
                    # the actual training distribution this epoch.
                    last_pi_acc = path_integration_accuracy(
                        net, n_trials=64, n_steps=n_steps_epoch, device=device,
                    )
                    last_fwhm = bump_fwhm(
                        net, epg_indices=epg_indices, epg_ix=epg_glom_ix,
                        n_trials=64, n_steps=n_steps_epoch, device=device,
                    )
                history["pi_acc"].append((epoch + step / steps_per_epoch, last_pi_acc))
                history.setdefault("fwhm", []).append(
                    (epoch + step / steps_per_epoch, last_fwhm)
                )

            # Live bar postfix — colour PI accuracy by quality.
            if last_pi_acc >= 0.9:
                acc_col = "\033[32m"  # green
            elif last_pi_acc >= 0.5:
                acc_col = "\033[33m"  # yellow
            else:
                acc_col = "\033[31m"  # red
            reset = "\033[0m"
            fwhm_deg = (
                f"{np.degrees(last_fwhm):.0f}°"
                if not math.isnan(last_fwhm) else "n/a"
            )
            pbar.set_postfix_str(
                f"loss={ema_loss:.5f}  mse={ema_mse:.5f}  "
                f"cosd={ema_cosd:.5f}  norm={ema_norm:.5f}  "
                f"{acc_col}pi_acc={last_pi_acc:.4f}{reset}  "
                f"fwhm={fwhm_deg}  best={best_loss:.5f}"
            )

            if float(loss.item()) < best_loss:
                best_loss = float(loss.item())
                _save_checkpoint(net, output_path, meta={
                    "epoch": epoch, "step": step, "loss": best_loss,
                    "pi_acc": last_pi_acc,
                    "n_units": N, "neuron_types": neuron_types.tolist(),
                    "type_names": type_names,
                })

        if save_every_epoch:
            epoch_path = output_path.replace(".pt", f"_epoch{epoch}.pt")
            _save_checkpoint(net, epoch_path, meta={
                "epoch": epoch, "step": steps_per_epoch,
                "loss": float(loss.item()),
                "pi_acc": last_pi_acc,
                "n_units": N, "neuron_types": neuron_types.tolist(),
                "type_names": type_names,
            })

        elapsed = time.time() - t0
        fwhm_deg = (
            f"{np.degrees(last_fwhm):.0f}°"
            if not math.isnan(last_fwhm) else "n/a"
        )
        print(
            f"[hulse_cx] epoch {epoch}/{n_epochs} done — "
            f"loss(ema)={ema_loss:.5f}  pi_acc={last_pi_acc:.4f}  fwhm={fwhm_deg}  "
            f"best_loss={best_loss:.5f}  lr={opt.param_groups[0]['lr']:.2e}  "
            f"elapsed={elapsed/60:.1f} min"
        )
        if sched is not None:
            sched.step()

    return {"best_loss": best_loss, "history": history,
            "output_path": output_path, "final_pi_acc": last_pi_acc}


def bump_fwhm(
    net: HulseCxRNN,
    epg_indices: np.ndarray,
    epg_ix: np.ndarray,
    *,
    n_trials: int = 64,
    n_steps: int = 100,
    device: str = "cpu",
    n_glom: int = 16,
) -> float:
    """Mean FWHM of the EPG bump (in radians) at the last frame of a batch.

    Computed by:
      1. Running a fresh path-integration batch through the network.
      2. Taking sigmoid(h) on the 46 EPG neurons at the final timestep.
      3. Binning into `n_glom` glomerular wedges (uniform around the ring).
      4. Rolling so the peak glomerulus is at the centre.
      5. Counting wedges with activity > half-max and converting to radians.

    Returns nan if no bump is detectable (all-uniform activity).
    """
    net.eval()
    with torch.no_grad():
        batch = generate_path_integration_batch(n_trials, n_steps, device=device)
        _, h = net(batch.u)
    net.train()

    r_epg = torch.sigmoid(h[:, -1, epg_indices]).cpu().numpy()   # (B, n_epg)
    epg_ix_arr = np.asarray(epg_ix, dtype=int)
    glom_act = np.zeros((r_epg.shape[0], n_glom), dtype=np.float32)
    for g in range(n_glom):
        mask = epg_ix_arr == g
        if mask.any():
            glom_act[:, g] = r_epg[:, mask].mean(axis=1)

    wedge_rad = 2.0 * np.pi / n_glom
    fwhms = []
    for b in range(glom_act.shape[0]):
        v = glom_act[b]
        peak = int(np.argmax(v))
        v_rolled = np.roll(v, n_glom // 2 - peak)
        half = float(v_rolled.max()) / 2.0
        if half <= 0:
            continue
        above = v_rolled > half
        if not above.any():
            continue
        idx = np.where(above)[0]
        # Width = number of contiguous wedges above half-max around the centre.
        c = n_glom // 2
        left = c
        while left - 1 >= 0 and v_rolled[left - 1] > half:
            left -= 1
        right = c
        while right + 1 < n_glom and v_rolled[right + 1] > half:
            right += 1
        width = right - left + 1
        # If the bump is wider than the rolled window (e.g. activity barely above half everywhere)
        # fall back to total above-half count.
        if width <= 1:
            width = max(width, int(above.sum()))
        fwhms.append(width * wedge_rad)

    if not fwhms:
        return float("nan")
    return float(np.mean(fwhms))


def path_integration_accuracy(
    net: HulseCxRNN,
    n_trials: int = 64,
    n_steps: int = 100,
    device: str = "cpu",
) -> float:
    """Mean cosine similarity between predicted and true head direction.

    1.0 means perfect path integration. Hulse aims for ~0.95+ on the
    test set after 10 epochs.
    """
    net.eval()
    with torch.no_grad():
        batch = generate_path_integration_batch(n_trials, n_steps, device=device)
        y_hat, _ = net(batch.u)
        # Skip the first 10 steps (initial-condition lead-in).
        warmup = 10
        y_hat_n = y_hat[:, warmup:, :] / (
            y_hat[:, warmup:, :].norm(dim=-1, keepdim=True) + 1e-8
        )
        y_n = batch.y[:, warmup:, :]
        cosine = (y_hat_n * y_n).sum(dim=-1)
        acc = cosine.mean().item()
    net.train()
    return acc


def _save_checkpoint(net: HulseCxRNN, path: str, meta: dict) -> None:
    """Save a state dict with auxiliary metadata."""
    state = {
        "W_rec": net.W_rec.detach().cpu(),
        "W_in": net.W_in.detach().cpu(),
        "b": net.b.detach().cpu(),
        "W_out": net.W_out.detach().cpu(),
        "b_out": net.b_out.detach().cpu(),
        "W_con": net.W_con.detach().cpu(),
        "tau": net.tau,
        "dt": net.dt,
        "n_units": net.n_units,
        "n_input": net.n_input,
        "n_output": net.n_output,
        "meta": meta,
    }
    torch.save(state, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main():
    import argparse

    p = argparse.ArgumentParser(description="Train Hulse Model A CX teacher")
    p.add_argument("--datapath", default="papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2",
                   help="hemibrain CSV directory")
    p.add_argument("--output", default="papers/hulse_cx/trained/hulse_cx_seed0.pt")
    p.add_argument("--n_trials", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=100)
    def _parse_n_steps(s: str):
        """Accept either '100' or '100,1000,1000' (per-epoch schedule)."""
        if "," in s:
            return [int(x) for x in s.split(",") if x.strip()]
        return int(s)
    p.add_argument("--n_steps", type=_parse_n_steps, default=100,
                   help="trial length in timesteps. Either a single int (constant)"
                        " or a comma-separated per-epoch schedule, e.g."
                        " '100,1000,1000,1000,1000' for a curriculum.")
    p.add_argument("--n_epochs", type=int, default=5)
    def _parse_lr(s: str):
        """Accept '1e-3' or '5e-3,1e-3,5e-4,2e-4,1e-4' (per-epoch schedule)."""
        if "," in s:
            return [float(x) for x in s.split(",") if x.strip()]
        return float(s)
    p.add_argument("--lr", type=_parse_lr, default=1e-3,
                   help="learning rate. Either a single float (default 1e-3, "
                        "with MultiStepLR drop at --lr-drop-epoch) or a "
                        "comma-separated per-epoch schedule, e.g. "
                        "'5e-3,1e-3,5e-4,2e-4,1e-4' matching --n_steps curriculum.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-interval", type=int, default=50,
                   help="record metrics every N steps")
    p.add_argument("--eval-interval", type=int, default=100,
                   help="refresh pi_acc every N steps (default 100; lower = laggier "
                        "but slower; was 500 prior to this commit)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run for debugging (200 trials, 1 epoch)")
    args = p.parse_args()

    if args.smoke:
        args.n_trials = 200
        args.n_epochs = 1

    stats = train_hulse_cx_teacher(
        connconstr_datapath=args.datapath,
        output_path=args.output,
        n_trials=args.n_trials,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr_init=args.lr,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )
    print(f"[hulse_cx] best_loss={stats['best_loss']:.4f}")
    print(f"[hulse_cx] checkpoint saved to {args.output}")


if __name__ == "__main__":
    _main()
