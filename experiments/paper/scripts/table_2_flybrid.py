"""Tab. 2: GNN and Known-ODE on the hybrid FlyWire connectomes, with and
without proximal null edges -> tables/flybrid_inliers.tex

    python scripts/table_2_flybrid.py

Rows: experiment 7 (<variant>_blank50_kode_cv0N, <variant>_blank50_condl25_cv0N);
ML-baseline rows are the published controls, typed here and labelled so.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import (FOLDS, TAB_DIR, aggregate, cell, frames_tex, metric_cells, pct,  # noqa: E402,F401
                          print_rows, save_numbers)


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
               r"are the published controls. Prediction metrics on noise-free held-out stimuli (" + frames_tex(*ko.values(), *gnn.values()) + r" frames by fold for the GNN and Known-ODE rows). "
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


def main():
    numbers = {}
    table2(numbers)
    for key, value in numbers.items():
        save_numbers(key, value)
        print_rows(key, value)


if __name__ == "__main__":
    main()
