"""The paper's three result tables from the campaign's metrics.txt files.

    python scripts/make_tables.py            (from experiments/paper)

Writes tables/cv_table_gnn_vs_baselines.tex (Tab. 1), tables/flybrid_inliers.tex
(Tab. 2) and tables/cv_table_gnn_cross_noise.tex (Supp. Tab. 4), plus
tables/table_numbers.json with every aggregated value.

Each GNN row is the general-form GNN of the campaign (g_phi = MLP(a_i, a_j,
v_i, v_j) under a group lasso of 25, "cond_l25" in experiments/), five folds
cv00..cv04, read from <GNN_OUTPUT_ROOT>/log/fly/<run>/results/metrics.txt:
one_step_r, rollout_r, Wij_R2 (template readout over the fitted edges),
tau_R2 and V_rest_R2 with their outlier percentages, clustering_accuracy.
Mean +- SD over folds (SD with ddof = 0, as the published tables).

Rows with no campaign run yet keep the published numbers and are printed in
red; rows the campaign ran with a different training scheme say so in their
label. Captions are green (colour `revised`, defined by edit_tex.py) where they describe new content.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import LOG_ROOT, TAB_DIR, read_metrics  # noqa: E402

FOLDS = [f"cv{i:02d}" for i in range(5)]
GOOD, BAD = 0.9, 0.3


# ----------------------------------------------------------------------------- aggregation
def aggregate(pattern, folds=FOLDS):
    """pattern has {fold}; returns {metric: (mean, sd)} and n folds found."""
    acc = {}
    n = 0
    for f in folds:
        m = read_metrics(os.path.join(LOG_ROOT, pattern.format(fold=f)))
        if "Wij_R2" not in m:
            continue
        n += 1
        for k in ("one_step_r", "rollout_r", "Wij_R2", "tau_R2", "tau_pct_outliers",
                  "V_rest_R2", "V_rest_pct_outliers", "clustering_accuracy"):
            acc.setdefault(k, []).append(float(m.get(k, np.nan)))
    out = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in acc.items()}
    out["n"] = n
    return out


def cell(ms, red=False):
    """$m{\\pm}s$, green above 0.9, orange below 0.3; red overrides."""
    m, s = ms
    if np.isnan(m):
        return "$\\cdot$"
    body = f"${m:.2f}{{\\pm}}{s:.2f}$"
    if red:
        return f"\\textcolor{{red}}{{{body}}}"
    if m > GOOD:
        return f"\\good{{{body}}}"
    if m < BAD:
        return f"\\bad{{{body}}}"
    return body


def pct(ms, red=False, like=None):
    """$\\,(x.x)$ outlier percentage, coloured like its R^2 cell."""
    m, _ = ms
    body = f"$\\,({m:.1f})$"
    if red:
        return f"\\textcolor{{red}}{{{body}}}"
    if like is not None and like > GOOD:
        return f"\\good{{{body}}}"
    if like is not None and like < BAD:
        return f"\\bad{{{body}}}"
    return body


def metric_cells(a, red=False):
    """The seven metric cells of a GNN / Known-ODE row (split tau / V_rest)."""
    return (f"  & {cell(a['one_step_r'], red)} & {cell(a['rollout_r'], red)}\n"
            f"  & {cell(a['Wij_R2'], red)}\n"
            f"  & {cell(a['tau_R2'], red)} & {pct(a['tau_pct_outliers'], red, a['tau_R2'][0])}\n"
            f"  & {cell(a['V_rest_R2'], red)} & {pct(a['V_rest_pct_outliers'], red, a['V_rest_R2'][0])}\n")


def published(one, roll, W, tau, tau_out, V, V_out, cl=None):
    """A row of the published paper, as (mean, sd) tuples."""
    d = {"one_step_r": one, "rollout_r": roll, "Wij_R2": W, "tau_R2": tau,
         "tau_pct_outliers": (tau_out, 0.0), "V_rest_R2": V, "V_rest_pct_outliers": (V_out, 0.0)}
    if cl is not None:
        d["clustering_accuracy"] = cl
    d["n"] = 5
    return d


def red_text(s):
    return f"\\textcolor{{red}}{{{s}}}"


# ----------------------------------------------------------------------------- Table 1
def table1(numbers):
    gnn = {
        "noise_free": aggregate("flyvis_noise_free_blank50_condl25_{fold}"),
        "noise_005": aggregate("flyvis_noise_005_blank50_condl25_{fold}"),
        "noise_05": aggregate("flyvis_noise_05_blank50_condl25_{fold}"),
    }
    numbers["table1_gnn"] = gnn
    kode15 = {k: aggregate(f"flyvis_{k}_blank50_kode217_{{fold}}") for k in ("noise_free", "noise_005", "noise_05")}
    kode_pub = {   # published Known-ODE rows, used only until experiment 15's re-analysis has landed
        "noise_free": published((1.0, 0), (1.0, 0), (0.96, 0), (1.0, 0), 1.9, (0.97, 0), 10.2, (0.92, 0)),
        "noise_005": published((1.0, 0), (1.0, 0), (0.99, 0), (1.0, 0), 0.0, (0.99, 0), 4.7, (0.93, 0)),
        "noise_05": published((1.0, 0), (1.0, 0), (1.0, 0), (1.0, 0), 0.0, (1.0, 0), 0.0, (0.93, 0)),
    }
    kode = {k: (kode15[k] if kode15[k]["n"] == 5 else kode_pub[k]) for k in kode15}
    numbers["table1_known_ode"] = kode
    rows = []
    labels = [("noise_free", "noise-free", "0"), ("noise_005", "low noise", "0.05"), ("noise_05", "high noise", "0.5")]
    for i, (k, lab, sig) in enumerate(labels):
        a = kode[k]
        rows.append(f"{'Known ODE' if i == 0 else '':<10} & {lab:<11} & ${sig}$\n" + metric_cells(a)
                    + f"  & {cell(a['clustering_accuracy'])} \\\\")
    rows.append("\\midrule")
    for i, (k, lab, sig) in enumerate(labels):
        a = gnn[k]
        rows.append(f"{'GNN (ours)' if i == 0 else '':<10} & {lab:<11} & ${sig}$\n" + metric_cells(a)
                    + f"  & {cell(a['clustering_accuracy'])} \\\\")
    baselines = r"""
\midrule
Recurrent MLP & noise-free & $0$  &\good{$0.95{\pm}0.04$} & \good{$0.96{\pm}0.05$} & \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & low-noise & $0.05$ & \good{$0.92{\pm}0.07$} & \good{$0.93{\pm}0.11$} & \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & high-noise & $0.5$ & $0.69{\pm}0.18$ & $0.78{\pm}0.24$ & \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
\midrule
EED Model & noise-free & $0$ & \good{$0.92{\pm}0.08$} & \good{$0.97{\pm}0.03$} & \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & low-noise & $0.05$ & $0.90{\pm}0.10$ & \good{$0.97{\pm}0.04$} & \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & high-noise & $0.5$ & $0.60{\pm}0.20$ & $0.77{\pm}0.25$ & \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
"""
    caption = (r"{\color{revised}\textbf{GNN and baselines on Flyvis-217} across three training-noise levels "
               r"($\sigma\in\{0,0.05,0.5\}$); 5-fold CV (mean~$\pm$~SD). GNN rows: the general-form GNN, "
               r"$g_\phi=\mathrm{MLP}(\mathbf{a}_i,\mathbf{a}_j,v_i,v_j)$ under a group lasso of $25$ on its "
               r"input groups, one-step training (experiment 2 of the campaign, runs "
               r"\texttt{flyvis\_noise\_*\_blank50\_condl25\_cv0N}). Known-ODE rows: the published "
               r"Known-ODE runs re-analysed with the current analysis code (experiment 15). ML-baseline rows are "
               r"the published ones. Prediction metrics on noise-free held-out stimuli ($8{,}000$ frames). "
               r"Parameter recovery (possible only for Known-ODE and GNN): $R^2_{\hat{W}}$ by the per-edge "
               r"template readout over the fitted edges of the $434{,}112$; $R^2_{\hat{\tau}}$ and "
               r"$R^2_{\hat{V}^{\mathrm{rest}}}$ over all $13{,}741$ neurons, outlier-corrected "
               r"(Appendix~\ref{app:metrics}); GMM clustering accuracy over $65$ cell types. \good{Green}: $>0.9$.}")
    tex = (r"""% Tab. 1 -- written by scripts/make_tables.py; GNN rows = conductance lasso 25 (experiment 2)
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:cv_gnn_vs_baselines}
\tiny
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} lp{1cm}lrrrr@{}rr@{}rr@{}}
\toprule
& & model &  \multicolumn{2}{c}{prediction} & \multicolumn{4}{c}{parameter recovery} \\
model & condition & noise $\sigma$
  & \multicolumn{1}{c}{one-step $r$} & \multicolumn{1}{c}{rollout $r$}
  & \multicolumn{1}{c}{$R^2_{\hat{W}}$}
  & \multicolumn{2}{c}{$R^2_{\hat{\tau}}  (\text{out.} \%)$}
  & \multicolumn{2}{c}{$R^2_{\hat{V}^{\mathrm{rest}}}  (\text{out.} \%)$}
  & \multicolumn{1}{c}{cluster\ acc.} \\
\midrule
""" + "\n".join(rows) + baselines + r"""\bottomrule
\end{tabular*}
\end{table}
""")
    open(os.path.join(TAB_DIR, "cv_table_gnn_vs_baselines.tex"), "w").write(tex)


# ----------------------------------------------------------------------------- Table 2 (hybrid)
VARIANTS = [
    ("e8_flywireRF_noise_005", "het.\\ RF", "13\\,741", "327\\,358", "e8 hybrid", ""),
    ("e8_flywireRF_proximal_nulls_noise_005", "+ n.e.", "13\\,741", "2\\,418\\,403", "e8 hybrid", ""),
    ("full_eye_flywireRF_noise_005", "het.\\ RF", "50\\,412", "1\\,266\\,378", "FlyWire eye", "\\textit{larger}"),
    ("full_eye_flywireRF_proximal_nulls_noise_005", "+ n.e.", "50\\,412", "9\\,642\\,335", "FlyWire eye", "\\textit{visual field}"),
]


def table2(numbers):
    def block(model_label, pattern):
        lines = []
        agg = {}
        for i, (v, cond, n, e, name, side) in enumerate(VARIANTS):
            a = aggregate(pattern.format(variant=v, fold="{fold}"))
            agg[v] = a
            first = model_label if i == 0 else side
            lines.append(f"{first:<24} & {cond:<10} & ${n}$ & ${e}$ & {name}\n" + metric_cells(a).rstrip("\n") + " \\\\")
            if i == 1:
                lines.append("  \\cmidrule(l{0pt}r{0pt}){2-12}")
        return lines, agg
    ko_lines, ko = block("Known ODE", "{variant}_blank50_kode_{fold}")
    gnn_lines, gnn = block("GNN", "{variant}_blank50_condl25_{fold}")
    numbers["table2_known_ode"] = ko
    numbers["table2_gnn"] = gnn
    baselines = r"""
MLP & het.\ RF   & $13\,741$ & $327\,358$ & e8 hybrid & \good{$0.90{\pm}0.09$} & \good{$0.97{\pm}0.05$}
& \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} \\
\cmidrule(l{0pt}r{0pt}){2-12}
MLP \textit{l.v.f.} & het.\ RF   & $50\,412$ & $1\,266\,378$ & FlyWire eye & $0.86{\pm}0.10$ & \good{$0.96{\pm}0.06$}
& \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} \\
\midrule
EED  & het.\ RF   &  $13\,741$ & $327\,358$ & e8 hybrid & $0.88{\pm}0.12$ & \good{$0.97{\pm}0.05$}
& \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} \\
\cmidrule(l{0pt}r{0pt}){2-12}
EED \textit{l.v.f.} & het.\ RF  & $50\,412$ & $1\,266\,378$ & FlyWire eye & $0.83{\pm}0.14$ & \good{$0.95{\pm}0.06$}
& \multicolumn{1}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} \\
"""
    caption = (r"{\color{revised}\textbf{GNN recovery on hybrid connectome variants} under connectivity uncertainty "
               r"(null-edge augmentation); low model noise $\sigma = 0.05$, 5-fold CV (mean $\pm$ SD). Variant is "
               r"either e8 hybrid ($13{,}741$ neurons) or FlyWire eye ($50{,}412$ neurons). Known-ODE and GNN rows "
               r"are the campaign's re-runs (experiment 7, runs \texttt{<variant>\_blank50\_kode\_cv0N} and "
               r"\texttt{<variant>\_blank50\_condl25\_cv0N}): the GNN is the general-form GNN, "
               r"$g_\phi=\mathrm{MLP}(\mathbf{a}_i,\mathbf{a}_j,v_i,v_j)$ under a group lasso of $25$; ML baselines "
               r"are the published controls. Prediction metrics on noise-free held-out stimuli ($8{,}000$ frames). "
               r"Parameter recovery: $R^2_{\widehat{W}}$ by the template readout over the fitted non-zero edges; "
               r"$R^2_{\widehat{\tau}}$ and $R^2_{\widehat{V}^{\mathrm{rest}}}$ over all neurons, outlier-corrected "
               r"(Appendix~\ref{app:metrics}). \textcolor{green!50!black}{Green}: $>0.9$.}")
    tex = (r"""% Tab. 2 -- written by scripts/make_tables.py; Known ODE + GNN rows = experiment 7 (kode, condl25)
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:zero_edge_inliers}
\tiny
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} lrrrrrrrr@{}rr@{}r@{}}
\toprule
& & & & & \multicolumn{2}{c}{prediction} & \multicolumn{3}{c}{parameter recovery} \\
model & \multicolumn{1}{c}{condition} & \multicolumn{1}{c}{neurons} & \multicolumn{1}{c}{edges} & \multicolumn{1}{c}{variant}
  & \multicolumn{1}{c}{one-step $r$} & \multicolumn{1}{c}{rollout $r$}
  & \multicolumn{1}{c}{$R^2_{\widehat{W}}$}
  & \multicolumn{2}{c}{$R^2_{\widehat{\tau}}$\,(out.\ \%)}
  & \multicolumn{2}{c}{$R^2_{\widehat{V}^{\mathrm{rest}}}$\,(out.\ \%)} \\
\midrule
""" + "\n".join(ko_lines) + "\n\\midrule\n" + "\n".join(gnn_lines) + "\n  \\midrule" + baselines + r"""\bottomrule
\end{tabular*}
\end{table}
""")
    open(os.path.join(TAB_DIR, "flybrid_inliers.tex"), "w").write(tex)


# ----------------------------------------------------------------------------- Supp. Tab. 4 (cross-noise)
def table_cross(numbers):
    rows = []
    agg = {}

    def add(label, gamma, edges, pattern=None, pub=None, red=False, folds=FOLDS):
        if pattern is not None:
            a = aggregate(pattern, folds)
        else:
            a = pub
        agg[label] = a
        rows.append(f"{label:<40} & ${gamma}$ & ${edges}$\n" + metric_cells(a, red)
                    + f"  & {cell(a['clustering_accuracy'], red)} \\\\")

    add("low model noise", "0", "434\\,112", "flyvis_noise_005_blank50_condl25_{fold}")
    add("low meas.\\ noise, 20-step recurrent", "0.1", "434\\,112", "flyvis_noise_005_010_condl25rc20_{fold}")
    add("mid meas.\\ noise, 20-step recurrent", "0.2", "434\\,112", "flyvis_noise_005_020_condl25rc20_{fold}")
    add("unknown stimulus", "0", "434\\,112", "flyvis_noise_005_INR_davis_blank50_condl251s_{fold}")
    add("$+400\\%$ null edges", "0", "2\\,170\\,560", "flyvis_noise_005_null400_condl25_{fold}")
    add("$-20\\%$ edges removed", "0", "347\\,000", "flyvis_noise_005_rm20_condl251s_{fold}")
    add("$-50\\%$ edges removed", "0", "217\\,056", "flyvis_noise_005_rm50_condl251s_{fold}")
    add("$1/5$ frames, rollout horizon 6", "0", "434\\,112", "flyvis_noise_005_s5h06_condl25_{fold}")
    add("$1/5$ frames, rollout horizon 21", "0", "434\\,112", "flyvis_noise_005_s5h21_condl25_{fold}")
    add("$10\\%$ hidden", "0", "434\\,112", "flyvis_noise_005_hid10_condl251s_{fold}")
    add("$20\\%$ hidden", "0", "434\\,112", "flyvis_noise_005_hid20_none_condl251s_{fold}")
    numbers["table_cross"] = agg
    caption = (r"{\color{revised}\textbf{GNN evaluation across degraded Flyvis-217}; model noise $\sigma = 0.05$, "
               r"5-fold CV (mean~$\pm$~SD) with variable measurement noise ($\gamma$), unknown stimulus, "
               r"added/removed edges, sub-sampled frames, and hidden neurons. Every row is the general-form GNN "
               r"($g_\phi=\mathrm{MLP}(\mathbf{a}_i,\mathbf{a}_j,v_i,v_j)$, group lasso $25$) of the campaign: "
               r"experiments 2 (low model noise), 3 (measurement noise, trained with the 20-step recurrent rollout "
               r"since one-step training was not run for this form), 11 (unknown stimulus), 4 (null edges), "
               r"14 (edges removed, $10\%$ hidden), 5 ($1/5$ frames, the rollout scored on the observed frames "
               r"only, horizons 6 and 21) and 9 ($20\%$ hidden). Prediction metrics on noise-free held-out stimuli "
               r"($8{,}000$ frames). Parameter recovery: $R^2_{\hat{W}}$ by the template readout over the fitted "
               r"edges; $R^2_{\hat{\tau}}$ and $R^2_{\hat{V}^{\mathrm{rest}}}$ over all $13{,}741$ neurons, "
               r"outlier-corrected (Appendix~\ref{app:metrics}); GMM clustering accuracy over $65$ cell types. "
               r"\good{Green}: $>0.9$. \bad{Orange}: $<0.3$.}")
    tex = (r"""% Supp. Tab. 4 -- written by scripts/make_tables.py; conductance lasso 25 rows from experiments 2, 3, 4, 5, 9, 11
\begin{table}[h!]
\centering
\caption{""" + caption + r"""}
\label{tab:cv_cross_noise}
\tiny
\setlength{\tabcolsep}{1.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} @{}llrrrrr@{}rr@{}rr@{}}
\toprule
& \makebox[0pt][l]{measurement} & & \multicolumn{2}{c}{prediction} & \multicolumn{4}{c}{parameter recovery} \\
condition & noise $\gamma$ & \multicolumn{1}{c}{edges}
  & \multicolumn{1}{c}{one-step $r$} & \multicolumn{1}{c}{rollout $r$}
  & \multicolumn{1}{c}{$R^2_{\hat{W}}$}
  & \multicolumn{2}{c}{$R^2_{\hat{\tau}} (\text{out.} \%)$}
  & \multicolumn{2}{c}{$R^2_{\hat{V}^{\mathrm{rest}}} (\text{out.} \%)$}
  & \multicolumn{1}{c}{cluster.\ acc.} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")
    open(os.path.join(TAB_DIR, "cv_table_gnn_cross_noise.tex"), "w").write(tex)


# ----------------------------------------------------------------------------- Supp. Tab. 7 (Known-ODE)
KO_ROWS = [
    ("noise-free", "0", "0", "434\\,112", "flyvis_noise_free"),
    ("low model noise", "0.05", "0", "434\\,112", "flyvis_noise_005"),
    ("high model noise", "0.5", "0", "434\\,112", "flyvis_noise_05"),
    ("low meas.\\ noise", "0.05", "0.1", "434\\,112", "flyvis_noise_005_010"),
    ("mid meas.\\ noise", "0.05", "0.2", "434\\,112", "flyvis_noise_005_020"),
    ("$+400\\%$ null edges", "0.05", "0", "2\\,170\\,560", "flyvis_noise_005_null_edges_pc_400"),
    ("$-20\\%$ edges removed", "0.05", "0", "347\\,000", "flyvis_noise_005_removed_pc_20"),
    ("$-50\\%$ edges removed", "0.05", "0", "217\\,056", "flyvis_noise_005_removed_pc_50"),
]


def table_known_ode(numbers):
    rows, agg = [], {}
    for lab, sig, gam, edges, base in KO_ROWS:
        a = aggregate(f"{base}_blank50_kode217_{{fold}}")
        if a["n"] == 0:
            print(f"  [Supp. Tab. 7] {base}: experiment 15 not landed, table left as published")
            return
        agg[base] = a
        rows.append(f"{lab:<24} & ${sig}$ & ${gam}$ & ${edges}$\n" + metric_cells(a)
                    + f"  & {cell(a['clustering_accuracy'])} \\\\")
    numbers["table_known_ode"] = agg
    caption = (r"{\color{revised}\textbf{Known-ODE evaluation across degraded versions} of the Flyvis training "
               r"data: model and measurement noise, added/removed connectivity edges. The published Known-ODE "
               r"runs, re-analysed with the paper's current analysis code (experiment 15). Five-fold "
               r"cross-validation (mean~$\pm$~SD). Prediction: metrics computed on noise-free data with "
               r"held-out stimuli ($8{,}000$ frames). Parameter recovery: $R^2_{\hat{W}}$ by the template readout over the "
               r"fitted edges, as for the GNN rows; $R^2_{\hat{\tau}}$ and $R^2_{\hat{V}^{\mathrm{rest}}}$ over "
               r"all neurons, outlier-corrected (outlier fraction in parentheses, Appendix~\ref{app:metrics}); "
               r"GMM clustering accuracy over $65$ cell types. "
               r"\good{Green}: value $> 0.9$. \bad{Orange}: value $< 0.3$.}")
    tex = (r"""% Supp. Tab. 7 -- written by scripts/make_tables.py from experiment 15 (kode217 runs)
\begin{table}[h!]
\centering
\caption{""" + caption + r"""}
\label{tab:cv_known_ode}
\tiny
\setlength{\tabcolsep}{1.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} @{}lllrrrrr@{}rr@{}rr@{}}
\toprule
& model & measurement & & \multicolumn{2}{c}{prediction} & \multicolumn{4}{c}{parameter recovery} \\
condition & noise $\sigma$ & noise $\gamma$ & \multicolumn{1}{c}{edges}
  & \multicolumn{1}{c}{one-step $r$} & \multicolumn{1}{c}{rollout $r$}
  & \multicolumn{1}{c}{$R^2_{\hat{W}}$}
  & \multicolumn{2}{c}{$R^2_{\hat{\tau}} (\text{out.} \%)$}
  & \multicolumn{2}{c}{$R^2_{\hat{V}^{\mathrm{rest}}} (\text{out.} \%)$}
  & \multicolumn{1}{c}{cluster\ acc.} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")
    open(os.path.join(TAB_DIR, "cv_table_known_ode_conditions.tex"), "w").write(tex)


def main():
    numbers = {}
    table1(numbers)
    table_known_ode(numbers)
    table2(numbers)
    table_cross(numbers)
    json.dump(numbers, open(os.path.join(TAB_DIR, "table_numbers.json"), "w"), indent=1)
    for name, d in numbers.items():
        print(f"== {name}")
        for k, a in d.items():
            print(f"  {k:<48} n={a['n']}  one {a['one_step_r'][0]:.3f}  roll {a['rollout_r'][0]:.3f}  "
                  f"W {a['Wij_R2'][0]:.3f}  tau {a['tau_R2'][0]:.3f} ({a['tau_pct_outliers'][0]:.1f})  "
                  f"V {a['V_rest_R2'][0]:.3f} ({a['V_rest_pct_outliers'][0]:.1f})  "
                  f"cl {a.get('clustering_accuracy', (np.nan,))[0]:.3f}")


if __name__ == "__main__":
    main()
