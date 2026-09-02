"""Parameter panels for the conductance twin, for `-o plot`.

Four things the twin has that a GNN does not, and that nothing else plots:

  a  learned reversal potentials E_inh / E_exc, against the teacher's voltage
     range. THE POINT OF THE PANEL is the range, not the histogram: the driving
     force (E - V_i) only carries the connectome's sign while E brackets every
     voltage V_i visits. A learned reversal that drifts inside the range flips
     excitation to inhibition for those neurons, silently, and the run still
     trains. `cond_reversal_mode: margin` makes that impossible by construction;
     'learned' does not, so this is the panel that says whether it happened.
  b  the conductance W^2 against the teacher's |W|. Not a recovery plot -- the
     teacher is current-based and has no conductance -- but the closed-form init
     sets W^2 = |alpha_curr| / |E - Vbar|, so the two should stay related, and
     how far training moves off that line is the interesting quantity.
  c  tau, learned vs teacher.
  d  V_rest, learned vs teacher.

c and d ARE recovery plots: tau and V_rest mean the same thing in both models --
Eqs. (1) and (2) share them -- so unlike W they are directly comparable.
"""
import os

import numpy as np
import torch

COLOR_TRUE, COLOR_PRED = "tab:green", "black"      # repo GT-vs-predicted convention
COLOR_INH, COLOR_EXC = "tab:blue", "tab:red"       # two distinct sources


def _state_dict(log_dir):
    import glob
    f = sorted(glob.glob(os.path.join(log_dir, "models", "*.pt")))
    if not f:
        return None
    sd = torch.load(f[-1], map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    # torch.compile wraps the module, so every key is prefixed
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def plot_twin_params(log_dir, ode_params, x_ts=None, out_name="twin_params"):
    """Four-panel figure. Returns the path, or None if this is not a twin run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    sd = _state_dict(log_dir)
    if sd is None or "E_exc" not in sd:
        return None                                  # not a conductance twin

    g = (lambda k: ode_params.get(k) if isinstance(ode_params, dict)
         else getattr(ode_params, k, None))
    tau_t, vr_t, W_t = g("tau_i"), g("V_i_rest"), g("W")

    E_inh, E_exc = sd["E_inh"].float().numpy(), sd["E_exc"].float().numpy()
    W_s = (sd["W"].float().squeeze(-1) ** 2).numpy()          # conductance = W^2
    ti = sd.get("type_index")
    tau_s = F.softplus(sd["raw_tau"].float()).numpy()
    vr_s = sd["V_rest"].float().numpy()
    if ti is not None and tau_s.shape[0] != ti.numel():        # per-type -> per-neuron
        idx = ti.long().numpy()
        tau_s, vr_s = tau_s[idx], vr_s[idx]

    vmin = vmax = None
    if x_ts is not None and getattr(x_ts, "voltage", None) is not None:
        v = x_ts.voltage
        vmin, vmax = float(v.min()), float(v.max())

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.0), dpi=150)

    # (a) reversals against the teacher's range
    ax = axs[0, 0]
    for arr, c, lab in ((E_inh, COLOR_INH, r"$E_{inh}$"), (E_exc, COLOR_EXC, r"$E_{exc}$")):
        if arr.size == 1:
            ax.axvline(float(arr), color=c, lw=1.6, label=f"{lab} = {float(arr):.2f}")
        else:
            ax.hist(arr, bins=60, color=c, alpha=0.6, label=f"{lab} (n={arr.size})")
    if vmin is not None:
        ax.axvspan(vmin, vmax, color="0.85", zorder=0,
                   label=f"teacher range [{vmin:.1f}, {vmax:.1f}]")
        n_cross = int((E_exc <= vmax).sum() + (E_inh >= vmin).sum())
        if n_cross:
            ax.text(0.02, 0.86, f"{n_cross} reversal(s) inside the range\n"
                                "-> driving force flips sign there",
                    transform=ax.transAxes, fontsize=7.5, color="tab:red")
    ax.set_xlabel("reversal potential"); ax.set_ylabel("count")
    ax.legend(fontsize=7, frameon=False)

    # (b) conductance vs the teacher's |W|
    ax = axs[0, 1]
    if W_t is not None:
        wt = np.abs(np.asarray(W_t, dtype=np.float64).ravel())
        n = min(wt.size, W_s.size)
        ax.hexbin(wt[:n], W_s[:n], gridsize=60, bins="log", cmap="viridis", mincnt=1)
        ax.set_xlabel(r"teacher $|W_{ij}|$"); ax.set_ylabel(r"student $W_{ij}^2$ (conductance)")
        r = np.corrcoef(wt[:n], W_s[:n])[0, 1]
        ax.text(0.03, 0.93, f"r = {r:.3f}\nnot a recovery: the teacher\nhas no conductance",
                transform=ax.transAxes, fontsize=7.5, va="top")

    # (c, d) tau and V_rest ARE comparable -- both models share them
    for ax, s_arr, t_arr, name in ((axs[1, 0], tau_s, tau_t, r"$\tau$"),
                                   (axs[1, 1], vr_s, vr_t, r"$V_{rest}$")):
        if t_arr is None:
            ax.set_visible(False); continue
        t = np.asarray(t_arr, dtype=np.float64).ravel()
        n = min(t.size, s_arr.size)
        ax.scatter(t[:n], s_arr[:n], s=3, color=COLOR_PRED, alpha=0.35, linewidths=0)
        lo = float(min(t[:n].min(), s_arr[:n].min()))
        hi = float(max(t[:n].max(), s_arr[:n].max()))
        ax.plot([lo, hi], [lo, hi], color=COLOR_TRUE, lw=1.0, ls="--")
        ok = np.isfinite(t[:n]) & np.isfinite(s_arr[:n])
        r2 = (1 - ((s_arr[:n][ok] - t[:n][ok]) ** 2).sum()
              / max(((t[:n][ok] - t[:n][ok].mean()) ** 2).sum(), 1e-30))
        ax.set_xlabel(f"teacher {name}"); ax.set_ylabel(f"student {name}")
        ax.text(0.03, 0.93, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
                fontsize=8, va="top")

    for ax, lbl in zip(axs.ravel(), "abcd"):
        if not ax.get_visible():
            continue
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.text(-0.02, 1.05, lbl, transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="right", va="bottom")

    out = os.path.join(log_dir, "results", out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out + ".png"
