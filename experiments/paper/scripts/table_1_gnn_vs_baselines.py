"""Tab. 1: Known-ODE, general-form GNN and ML baselines on Flyvis-217 at three
model-noise levels -> tables/cv_table_gnn_vs_baselines.tex

    python scripts/table_1_gnn_vs_baselines.py

GNN rows: experiment 2 (flyvis_noise_*_blank50_condl25_cv0N); Known-ODE rows:
experiment 15 (flyvis_noise_*_blank50_kode217_cv0N); ML-baseline rows are the
published values, typed here and labelled so in the caption.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import (FOLDS, TAB_DIR, aggregate, cell, frames_tex, metric_cells, pct,  # noqa: E402,F401
                          print_rows, save_numbers)


# ----------------------------------------------------------------------------- Table 1
def table1(numbers):
    gnn = {
        "noise_free": aggregate("flyvis_noise_free_blank50_condl25_{fold}"),
        "noise_005": aggregate("flyvis_noise_005_blank50_condl25_{fold}"),
        "noise_05": aggregate("flyvis_noise_05_blank50_condl25_{fold}"),
    }
    numbers["table1_gnn"] = gnn
    # the published Known-ODE checkpoints re-analysed by experiment 15; all five folds required
    kode = {k: aggregate(f"flyvis_{k}_blank50_kode217_{{fold}}") for k in ("noise_free", "noise_005", "noise_05")}
    assert all(a["n"] == 5 for a in list(kode.values()) + list(gnn.values())), "a Table 1 row lacks folds"
    numbers["table1_known_ode"] = kode
    rows = []
    labels = [("noise_free", "noise-free", "0"), ("noise_005", "low noise", "0.05"), ("noise_05", "high noise", "0.5")]
    for i, (k, lab, sig) in enumerate(labels):
        a = kode[k]
        rows.append(f"{'Known ODE' if i == 0 else '':<10} & {lab:<11} & ${sig}$\n" + metric_cells(a, w_outliers=True)
                    + f"  & {cell(a['clustering_accuracy'])} \\\\")
    rows.append("\\midrule")
    for i, (k, lab, sig) in enumerate(labels):
        a = gnn[k]
        rows.append(f"{'GNN (ours)' if i == 0 else '':<10} & {lab:<11} & ${sig}$\n" + metric_cells(a, w_outliers=True)
                    + f"  & {cell(a['clustering_accuracy'])} \\\\")
    baselines = r"""
\midrule
Recurrent MLP & noise-free & $0$  &\good{$0.95{\pm}0.04$} & \good{$0.96{\pm}0.05$} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & low-noise & $0.05$ & \good{$0.92{\pm}0.07$} & \good{$0.93{\pm}0.11$} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & high-noise & $0.5$ & $0.69{\pm}0.18$ & $0.78{\pm}0.24$ & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
\midrule
EED Model & noise-free & $0$ & \good{$0.92{\pm}0.08$} & \good{$0.97{\pm}0.03$} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & low-noise & $0.05$ & $0.90{\pm}0.10$ & \good{$0.97{\pm}0.04$} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
 & high-noise & $0.5$ & $0.60{\pm}0.20$ & $0.77{\pm}0.25$ & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{2}{c}{---} & \multicolumn{1}{c}{---} \\
"""
    # the published caption (overleaf/tables/cv_table_gnn_vs_baselines.tex); green = the words that changed
    caption = (r"\textbf{GNN and baselines on Flyvis-217} across three {\color{revised}model-noise} levels "
               r"($\sigma\in\{0,0.05,0.5\}$); 5-fold CV (mean~$\pm$~SD). Prediction metrics on noise-free held-out "
               r"stimuli. "
               r"Parameter recovery (possible only for Known-ODE and GNN): $R^2_{\hat{W}}$ over "
               r"{\color{revised}the fitted edges of the} $434{,}112$ {\color{revised}edges, outlier-corrected}; "
               r"$R^2_{\hat{\tau}}$ and $R^2_{\hat{V}^{\mathrm{rest}}}$ over all $13{,}741$ neurons, outlier-corrected "
               r"(Appendix~\ref{app:metrics}); GMM clustering accuracy over $65$ cell types. \good{Green}: $>0.9$. "
               r"{\color{revised}(At $\sigma=0.5$ the rollout from the initial voltages diverged ($r<0.5$) in 2 of the 10 "
               r"folds we trained, cv04 and cv09; the GNN row uses cv00--cv03 and cv05, cv04 replaced by the next fold that is not one.)}")
    tex = (r"""% Tab. 1 -- written by scripts/make_tables.py; GNN rows = conductance lasso 25 (experiment 2)
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:cv_gnn_vs_baselines}
\tiny
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcrrr@{}lr@{}lr@{}lr}
\toprule
& & model &  \multicolumn{2}{c}{prediction} & \multicolumn{6}{c}{parameter recovery} \\
model & condition & noise $\sigma$
  & \multicolumn{1}{c}{one-step $r$} & \multicolumn{1}{c}{rollout $r$}
  & \multicolumn{2}{c}{$R^2_{\hat{W}}  (\text{out.} \%)$}
  & \multicolumn{2}{c}{$R^2_{\hat{\tau}}  (\text{out.} \%)$}
  & \multicolumn{2}{c}{$R^2_{\hat{V}^{\mathrm{rest}}}  (\text{out.} \%)$}
  & \multicolumn{1}{c}{cluster\ acc.} \\
\midrule
""" + "\n".join(rows) + baselines + r"""\bottomrule
\end{tabular}}
\end{table}
""")
    open(os.path.join(TAB_DIR, "cv_table_gnn_vs_baselines.tex"), "w").write(tex)


def main():
    numbers = {}
    table1(numbers)
    for key, value in numbers.items():
        save_numbers(key, value)
        print_rows(key, value)


if __name__ == "__main__":
    main()
