"""The figures drawn from a RecoveredParams: the scatters and the error panels.

ONE IMPLEMENTATION, TWO CALLERS. `-o plot` draws these at the end of a run and
the trainer draws them at every checkpoint, and until this module existed they
were the same picture from two bodies of code: the plot pass got three decimals,
the outlier count, the fitted-edge line and the provenance stamp, while
tmp_training kept the older drawing and quietly disagreed with results/ about the
same run. Anything about how a recovered quantity is DRAWN belongs here.

The thresholds come from `metrics._thresh_for`, so the band a figure filters on is
the band results/metrics.txt filtered on, decided once by the config.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome_gnn.metrics import (
    recovery_param_metrics, is_degenerate_gt, r2_scatter_text, _thresh_for,
    W_OUTLIER_THRESH, TAU_OUTLIER_THRESH, VREST_OUTLIER_THRESH,
)
from connectome_gnn.results_layout import fig_out as _fig_out

def _finite_range(values, fallback):
    """min/max over the finite entries, falling back when there are none.

    An all-NaN array means the quantity was never measured; matplotlib rejects
    NaN axis limits, so the fallback keeps the (empty) panel drawable instead of
    raising in a plotting path.
    """
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return float(fallback[0]), float(fallback[1])
    return float(finite.min()), float(finite.max())


# The four headline scatters, and the one thing they all must satisfy: they are
# drawn from the SAME RecoveredParams that results/metrics.txt is written from.
# Before this, the tau and V_rest figures came from an earlier f_theta-slope
# computation and the W figures from the gain-correction chain, so a directory
# could report Wij_R2 0.942 beside a scatter drawn from a different estimator
# entirely -- and nothing on either said so.
#
# axis: (x label, y label, limits or None for data-driven, ticks or None)
_KEY_FOR = {"W": "Wij", "E_ij": "Eij", "tau": "tau", "V_rest": "V_rest",
            "msg_i": "msg_i"}
_LABEL_FOR = {"W": "W_ij", "E_ij": "E_ij", "tau": "tau", "V_rest": "V_rest",
              "msg_i": "msg_i"}
# The same names set in math, for the axis labels the scatters already use.
_TEX_FOR = {"W": r"W_{ij}", "E_ij": r"E_{ij}", "tau": r"\tau", "V_rest": r"V_{rest}"}
# And with the estimate's hat on the SYMBOL, not on the whole subscripted name:
# \hat{V_{rest}} draws one accent spanning "V_rest", which is not what it means.
_HAT_TEX_FOR = {"W": r"\hat{W}_{ij}", "E_ij": r"\hat{E}_{ij}",
                "tau": r"\hat{\tau}", "V_rest": r"\hat{V}_{rest}"}

_SCATTER_SPEC = {
    "W":      dict(out="Wij_comparison.png",    thresh=W_OUTLIER_THRESH,
                   xlabel=r"true $W_{ij}$",     ylabel=r"learned $W_{ij}$"),
    "tau":    dict(out="tau_comparison.png",    thresh=TAU_OUTLIER_THRESH,
                   xlabel=r"true $\tau$",       ylabel=r"learned $\tau$",
                   lim=(-0.025, 0.5), ticks=([0.0, 0.25, 0.5], ["0.0", "0.25", "0.5"])),
    "V_rest": dict(out="V_rest_comparison.png", thresh=VREST_OUTLIER_THRESH,
                   xlabel=r"true $V_{rest}$",   ylabel=r"learned $V_{rest}$"),
    "E_ij":   dict(out="Eij_comparison.png",    thresh=5.0,
                   xlabel=r"true $E_{ij}$",     ylabel=r"learned $E_{ij}$"),
    # THE AGGREGATE THE TRAJECTORY ACTUALLY DEPENDS ON. W and E trade off inside
    # it -- W wrong by 3x with E wrong by 1/3 lands msg_i on the identity line --
    # so a run whose W scatter is a mess and whose msg_i scatter is not has a
    # gauge problem, not a connectivity problem. No outlier threshold: this one
    # is a population of neuron-frames, not a per-neuron parameter.
    "msg_i":  dict(out="msg_i_comparison.png",  thresh=None,
                   xlabel=r"true $\mathrm{msg}_i$",
                   ylabel=r"learned $\mathrm{msg}_i$"),
}


def _plot_parameter_error(rec, scored, log_dir, quantities=("tau", "V_rest", "W"),
                          out_path=None):
    """The distribution of each quantity's ERROR, learned minus true, one panel each.

    WHAT A SCATTER CANNOT SHOW. On 434,112 edges a scatter is a black cloud and
    the eye reads its outline -- set by the few worst points -- rather than where
    the mass is. The error distribution says what the cloud and the R2 cannot:
    whether the residual is centred on zero or biased off it, and whether it is
    one tight mode or a mode plus a population the model never recovered.

    SIGNED AND IN THE QUANTITY'S OWN UNITS, not a ratio. learned - true keeps the
    sign, so a systematic under-estimate is visible as a shifted mode instead of
    being folded into |.|, and it keeps volts and seconds rather than turning
    every panel into the same dimensionless number -- which on a near-zero true
    value explodes for a reason that has nothing to do with recovery.

    THE AXIS IS CLIPPED AND SAYS SO. Each panel spans +/- the 90th percentile of
    |error|, so roughly a tenth of the sample falls outside and is piled into the
    two edge bins, drawn as the spikes at the ends; the exact share is printed as
    "N% off-scale" beside the IQR. Without the clip a handful of residuals orders
    of magnitude out would compress every real one onto the centre line.

    GEOMETRY MATCHES `_plot_recovered_scatter`: one panel is 10 x 9 inches at
    300 dpi, so the three panels side by side are the same object the three
    W/tau/V_rest scatters are and the two figures can sit in one row of a paper
    without either being resized.
    """
    panels = [(q, rec.pairs.get(q)) for q in quantities]
    panels = [(q, p) for q, p in panels if p is not None and len(p[0]) > 1]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(10 * len(panels), 9))
    if len(panels) == 1:
        axes = [axes]
    for col, (ax, (q, pair)) in enumerate(zip(axes, panels)):
        gt = np.asarray(pair[0], dtype=float).ravel()
        learned = np.asarray(pair[1], dtype=float).ravel()
        ok = np.isfinite(gt) & np.isfinite(learned)
        err = learned[ok] - gt[ok]
        if err.size < 2:
            ax.axis("off")
            continue
        lim = float(np.percentile(np.abs(err), 90.0))
        if not np.isfinite(lim) or lim <= 0:
            lim = float(max(np.max(np.abs(err)), 1e-12))
        off = 100.0 * np.mean(np.abs(err) > lim)
        q1, q3 = np.percentile(err, [25.0, 75.0])
        iqr = float(q3 - q1)
        # Clipped, not dropped: the out-of-range residuals are the point of the
        # "off-scale" number, so they are counted in the two edge bins rather
        # than silently removed from a histogram that reports a fraction.
        bins = np.linspace(-lim, lim, 61)
        ax.hist(np.clip(err, -lim, lim), bins=bins,
                weights=np.full(err.size, 1.0 / err.size),
                histtype="step", color="k", linewidth=1.8,
                label=f"IQR {iqr:.1e}   {off:.0f}% off-scale   n = {err.size:,}")
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.5)
        # Ticks in units of the leading power of ten, which is then named once in
        # the axis label -- five numbers on the axis instead of five exponents.
        exp = int(np.floor(np.log10(lim)))
        scale = 10.0 ** exp
        _ticks = [-lim, -lim / 2, 0.0, lim / 2, lim]
        ax.set_xticks(_ticks)
        ax.set_xticklabels([f"{t / scale:.1f}".rstrip("0").rstrip(".") if t else "0.0"
                            for t in _ticks], fontsize=30)
        ax.set_xlim(-lim * 1.02, lim * 1.02)
        _sym, _hat = _TEX_FOR.get(q, q), _HAT_TEX_FOR.get(q, q)
        ax.set_xlabel(rf"${_hat} - {_sym}$   ($\times 10^{{{exp}}}$)", fontsize=40)
        ax.set_ylabel("fraction", fontsize=40)
        ax.tick_params(axis="y", labelsize=30)
        ax.legend(loc="upper left", fontsize=22, frameon=False, handlelength=1.2)
        ax.text(-0.09, 1.02, "abcdefgh"[col], transform=ax.transAxes,
                fontsize=44, fontweight="bold", va="bottom", ha="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    # out_path is what the trainer passes to drop a stamped copy into
    # tmp_training/recovery/ instead of overwriting results/.
    out = out_path or _fig_out(log_dir, "parameter_error.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _plot_recovered_scatter(rec, scored, quantity, log_dir, mc="k", config=None,
                            out_path=None):
    """One quantity, learned against true, on the wo-outliers template.

    Outliers -- |learned - true| above the quantity's threshold -- are drawn in
    red and excluded from the R2 and slope, with the full-sample R2 in
    parentheses beside it and the share that was dropped underneath, so the
    figure states its own filtering rather than hiding it. A quantity with no
    threshold (the reversal) draws every point and reports one R2.

    Returns the path written, or None when the run does not have this quantity.
    """
    pair = rec.pairs.get(quantity) if rec is not None else None
    spec = _SCATTER_SPEC.get(quantity)
    if pair is None or spec is None:
        return None
    gt = np.asarray(pair[0], dtype=float).ravel()
    learned = np.asarray(pair[1], dtype=float).ravel()
    ok = np.isfinite(gt) & np.isfinite(learned)
    gt, learned = gt[ok], learned[ok]
    if gt.size < 2:
        return None

    # ONE BAND PER QUANTITY, decided by metrics._thresh_for so that the figure and
    # results/metrics.txt cannot disagree. _SCATTER_SPEC's own value is the
    # fallback for a caller with no config: before this, a run that overrode
    # recovery.W_outlier_thresh got one threshold in the file and another on the
    # picture, and E_ij was filtered in the file and unfiltered on the figure.
    from connectome_gnn.metrics import _thresh_for
    thresh = _thresh_for(quantity, config) if config is not None else spec.get("thresh")
    m = recovery_param_metrics(gt, learned, thresh)
    if thresh is None:
        out_mask = np.zeros(gt.size, dtype=bool)
        r2_head, slope = m["r2"], m["slope"]
        r2_all = None
    else:
        out_mask = np.abs(learned - gt) > thresh
        r2_head, slope, r2_all = m["r2_clean"], m["slope_clean"], m["r2"]
    pct_out = 100.0 * out_mask.sum() / gt.size

    fig = plt.figure(figsize=(10, 9))
    # edgecolors="none": at s=1 a marker edge is the same size as the marker, so
    # it only blurs the mark and darkens the dense centre.
    plt.scatter(gt[~out_mask], learned[~out_mask], c=mc, s=1, alpha=0.3,
                edgecolors="none", rasterized=True)
    if out_mask.any():
        # SAME SIZE AS THE INLIERS. At s=6 against s=1 each red point covered
        # about six times the area of a black one, so a population of 4 edges in
        # 434,112 -- 0.001% -- drew the eye as if it were a visible fraction of
        # the cloud. The share is stated in the "outliers: N%" text; the marks
        # only have to be findable, not loud, so colour does the separating and
        # alpha 0.6 keeps them visible on top of the dense centre.
        plt.scatter(gt[out_mask], learned[out_mask], c="red", s=1, alpha=0.6,
                    edgecolors="none", rasterized=True)
    lim = spec.get("lim")
    if lim is None:
        # From the TRUE values plus the inlier spread: a handful of learned
        # values orders of magnitude off would otherwise flatten every real
        # point onto one line, and they are already counted as outliers.
        _lo = float(min(gt.min(), np.percentile(learned, 1)))
        _hi = float(max(gt.max(), np.percentile(learned, 99)))
        _pad = 0.05 * max(_hi - _lo, 1e-9)
        lim = (_lo - _pad, _hi + _pad)
    _line = np.linspace(lim[0], lim[1], 2)
    plt.plot(_line, _line, "--", color="gray", linewidth=1, alpha=0.6)
    if thresh is not None:
        plt.plot(_line, _line + thresh, ":", color="gray", linewidth=1, alpha=0.5)
        plt.plot(_line, _line - thresh, ":", color="gray", linewidth=1, alpha=0.5)

    ax = plt.gca()
    # THREE DECIMALS, because two turn 0.995 into "1.00". The figure and
    # metrics.txt are drawn from one array and must not disagree on the number:
    # at two decimals every R2 above 0.995 reads as perfect recovery, which is
    # the one claim this scatter exists to support or refute.
    if is_degenerate_gt(gt):
        _txt = r2_scatter_text(gt, learned)
    elif r2_all is None:
        _txt = f"R²: {r2_head:.3f}\nslope: {slope:.3f}"
    else:
        _txt = f"R²: {r2_head:.3f} ({r2_all:.3f})\nslope: {slope:.3f}"
    ax.text(0.05, 0.95, _txt, transform=ax.transAxes, verticalalignment="top",
            fontsize=32)
    # The count beside the share: "0.0%" of 434,112 edges is anything from none
    # to a few hundred, and which it is changes what the R2 means.
    ax.text(0.05, 0.78,
            f"outliers: {int(out_mask.sum()):,} / {gt.size:,} ({pct_out:.3f}%)",
            transform=ax.transAxes, verticalalignment="top", fontsize=26)
    # HOW MANY EDGES THE SCATTER IS OF. The per-edge fit leaves some edges
    # unmeasured -- their presynaptic cell is rarely above the activity floor --
    # and the cloud shows only the ones that were fitted. A reader comparing two
    # runs' W scatters is otherwise comparing two different subsets of the
    # connectome without being told.
    _full = rec.diagnostics.get("_W_learned_full") if rec is not None else None
    if quantity in ("W", "E_ij") and _full is not None:
        _tot = int(np.asarray(_full).size)
        if _tot:
            ax.text(0.05, 0.70,
                    f"fitted: {gt.size:,} / {_tot:,} edges ({100.0 * gt.size / _tot:.1f}%)",
                    transform=ax.transAxes, verticalalignment="top", fontsize=26)
    plt.xlabel(spec["xlabel"], fontsize=56)
    plt.ylabel(spec["ylabel"], fontsize=56)
    plt.xlim(*lim)
    plt.ylim(*lim)
    # Ticks at 30, not the 51 the older paper figures used: at that size three
    # numbers span the axis and read as a second label.
    ticks = spec.get("ticks")
    if ticks:
        plt.xticks(ticks[0], ticks[1], fontsize=30)
        plt.yticks(ticks[0], ticks[1], fontsize=30)
    else:
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
    plt.tight_layout()
    out = out_path or _fig_out(log_dir, spec["out"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_form_comparison(rec, log_dir, out_path=None, reference=None,
                         ref_label="reference run"):
    """results/form_comparison.png: the two forms' R2 distributions, and the test.

    LEFT, THE DISTRIBUTIONS. One histogram per family over the per-edge R2 of
    the same messages, on a log count axis because the bulk of 434,112 edges
    piles into the top bin and everything that distinguishes the families lives
    in the tail that would otherwise be invisible. Red and blue for the two
    forms, which are two sources rather than a truth and a prediction.

    RIGHT, WHAT THE TEST SAYS, as text. The numbers and the picture are written
    by the same call so a figure can never carry a statistic the terminal did
    not print, and the caveat about autocorrelated frames travels with them.

    `reference` is the per-edge gain array of another run -- a model KNOWN to
    have no driving force is the empirical null -- which adds the across-run
    Kolmogorov-Smirnov test, the one comparison here whose null is not false by
    construction.
    """
    from connectome_gnn.metrics import form_comparison_stats

    d = getattr(rec, "diagnostics", {}) or {}
    cond = np.asarray(d.get("_form_cond_r2_full", []), dtype=np.float64)
    cur = np.asarray(d.get("_form_cur_r2_full", []), dtype=np.float64)
    if cond.size == 0 or cur.size == 0:
        return None, []
    stats, lines = form_comparison_stats(d)

    gain = cond - cur
    gain = gain[np.isfinite(gain)]
    if reference is not None and np.asarray(reference).size:
        ref = np.asarray(reference, dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        if ref.size:
            from scipy.stats import ks_2samp
            ks = ks_2samp(gain, ref)
            stats["ks_D"] = float(ks.statistic)
            stats["ks_p"] = float(ks.pvalue)
            lines.append(
                f"   against {ref_label}, whose message has no driving force: "
                f"KS D = {ks.statistic:.3f}, p = {ks.pvalue:.3g} "
                f"({gain.size:,} vs {ref.size:,} edges)")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})

    # PANEL A, ON 1 - R2 AND NOT ON R2. Every edge of a network that obeys its
    # own form sits between 0.98 and 1.0000, which is one bin: on the R2 axis
    # the two families are one spike and the picture says nothing. The
    # unexplained fraction on a log axis separates 0.99 from 0.9999 by two
    # decades, and that is where the families differ.
    ax = axes[0]
    ra = 1.0 - cond[np.isfinite(cond)]
    rb = 1.0 - cur[np.isfinite(cur)]
    _floor = 1e-8
    bins = np.logspace(np.log10(_floor), 0, 90)
    ax.hist(np.clip(rb, _floor, 1.0), bins=bins, color="tab:red", alpha=0.6,
            label="current form   $W u + C$")
    ax.hist(np.clip(ra, _floor, 1.0), bins=bins, color="tab:blue", alpha=0.6,
            label="conductance form   $b_1 u + b_2\\,u v_i + b_3$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("unexplained fraction of the message, $1 - R^2$", fontsize=13)
    ax.set_ylabel("edges", fontsize=13)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.text(0.0, 1.02, "a   the same messages, fitted inside each family",
            transform=ax.transAxes, va="bottom", fontsize=12, fontweight="bold")

    ax = axes[1]
    ax.axis("off")
    ax.text(0.0, 1.02, "b   is the gap more than an extra column buys by chance?",
            transform=ax.transAxes, va="bottom", fontsize=12, fontweight="bold")
    # Wrapped to the panel rather than trusting the lines to fit: they carry run
    # numbers whose width is not known when they are written, and an unwrapped
    # line ran across the panel beside it.
    import textwrap
    body = []
    for l in lines:
        ind = "   " if l.startswith("   ") else ""
        body += textwrap.wrap(l.strip(), 74, subsequent_indent=ind + "  ",
                              initial_indent=ind) or [""]
    ax.text(0.0, 0.94, "\n".join(body), transform=ax.transAxes, va="top",
            ha="left", fontsize=9.5, family="monospace")

    # PANEL C, WHERE THE FITTED REVERSAL LANDS. The ratio quoted in panel b is a
    # median of this, and a median cannot show that the population has two
    # modes: edges whose reversal sits within the voltage the cells reach, and
    # edges where the slope the reversal divides by is noise, which send |E| off
    # to hundreds or thousands. Log axis for that reason -- the modes are orders
    # apart. The data's own |v_i| is the line that decides which is which: a
    # reversal beyond it was never visited, so nothing in the data constrains it.
    E = np.abs(np.asarray(d.get("_form_E_cond_full", []), dtype=np.float64))
    E = E[np.isfinite(E) & (E > 0)]
    ax = axes[2]
    if E.size:
        _hi = float(np.quantile(E, 0.999))
        bins = np.logspace(np.log10(max(E.min(), 1e-3)), np.log10(max(_hi, 10.0)), 90)
        ax.hist(np.clip(E, bins[0], bins[-1]), bins=bins, color="tab:purple", alpha=0.75)
        ax.set_xscale("log")
        ax.set_yscale("log")
        vi = float(d.get("vi_abs_p99", float("nan")))
        _leg = []
        if vi == vi:
            ax.axvline(vi, color="k", lw=2)
            _leg.append(f"black: the voltage the data reaches, $|v_i| = {vi:.2f}$")
        gt = rec.pairs.get("E_ij", (None, None))[0] if hasattr(rec, "pairs") else None
        if gt is not None:
            g = np.unique(np.round(np.abs(np.asarray(gt, dtype=np.float64)), 3))
            g = g[np.isfinite(g) & (g > 0)][:4]
            for v in g:
                ax.axvline(v, color="tab:green", lw=1.5, ls="--")
            if g.size:
                _leg.append("green: the generator's own $|E|$")
        if _leg:
            ax.text(0.98, 0.96, "\n".join(_leg), transform=ax.transAxes, ha="right",
                    va="top", fontsize=9.5)
        ax.set_xlabel("$|E|$ the conductance form asks for, per edge", fontsize=13)
        ax.set_ylabel("edges", fontsize=13)
    else:
        ax.axis("off")
    ax.text(0.0, 1.02, "c   where the fitted reversal lands",
            transform=ax.transAxes, va="bottom", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = out_path or _fig_out(log_dir, "form_comparison.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, lines


def write_form_arrays(rec, log_dir, out_path=None):
    """results/form_comparison.npz: the per-edge arrays the test is run on.

    Four float32 arrays over the run's edges -- both forms' R2, the t statistic
    on the driving-force slope, and the frames each edge was fitted on -- about
    7 MB for 434,112 edges. They are kept because the comparison that matters
    across runs, one model's gain distribution against a model known to have no
    driving force, cannot be made from summary numbers, and refitting a finished
    run to get them back costs a GPU hour.
    """
    d = getattr(rec, "diagnostics", {}) or {}
    cond = d.get("_form_cond_r2_full")
    if cond is None:
        return None
    out = out_path or _fig_out(log_dir, "form_comparison.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        conductance_form_r2=np.asarray(cond, dtype=np.float32),
        current_form_r2=np.asarray(d.get("_form_cur_r2_full"), dtype=np.float32),
        t_driving_force=np.asarray(d.get("_form_t_b2_full"), dtype=np.float32),
        frames_per_edge=np.asarray(d.get("_form_n_used_full"), dtype=np.float32))
    return out
