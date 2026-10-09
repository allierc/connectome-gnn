"""Supp. Tab. 7: the Known-ODE across degraded Flyvis-217 data
-> tables/cv_table_known_ode_conditions.tex

    python scripts/table_s7_known_ode.py

Rows: experiment 15 (<condition>_blank50_kode217_cv0N), all five folds required.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import (FOLDS, TAB_DIR, aggregate, cell, frames_tex, metric_cells, pct,  # noqa: E402,F401
                          print_rows, save_numbers)


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
        assert a["n"] == 5, f"Supp. Tab. 7: {base} has {a['n']} of 5 folds (experiment 15)"
        agg[base] = a
        rows.append(f"{lab:<24} & ${sig}$ & ${gam}$ & ${edges}$\n" + metric_cells(a)
                    + f"  & {cell(a['clustering_accuracy'])} \\\\")
    numbers["table_known_ode"] = agg
    caption = (r"{\color{revised}\textbf{Known-ODE evaluation across degraded versions} of the Flyvis training "
               r"data: model and measurement noise, added/removed connectivity edges. The published Known-ODE "
               r"runs, re-analysed with the paper's current analysis code (experiment 15). Five-fold "
               r"cross-validation (mean~$\pm$~SD). Prediction: metrics computed on noise-free data with "
               r"held-out stimuli (" + frames_tex(*agg.values()) + r" frames by fold). Parameter recovery: $R^2_{\hat{W}}$ by the template readout over the "
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
    table_known_ode(numbers)
    for key, value in numbers.items():
        save_numbers(key, value)
        print_rows(key, value)


if __name__ == "__main__":
    main()
