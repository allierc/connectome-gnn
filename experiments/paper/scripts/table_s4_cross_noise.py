"""Supp. Tab. 4: the general-form GNN across degraded Flyvis-217 data
-> tables/cv_table_gnn_cross_noise.tex

    python scripts/table_s4_cross_noise.py

Rows: experiments 2, 3, 4, 5, 9, 11 and 14 (run patterns in the add() calls).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import (FOLDS, TAB_DIR, aggregate, cell, frames_tex, metric_cells, pct,  # noqa: E402,F401
                          print_rows, save_numbers)


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
               r"(" + frames_tex(*[a for k, a in agg.items() if k != "unknown stimulus"]) + r" frames by fold; the unknown-stimulus row on the first " + frames_tex(agg["unknown stimulus"]) + r" training frames, which the SIREN was fitted on). Parameter recovery: $R^2_{\hat{W}}$ by the template readout over the fitted "
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


def main():
    numbers = {}
    table_cross(numbers)
    for key, value in numbers.items():
        save_numbers(key, value)
        print_rows(key, value)


if __name__ == "__main__":
    main()
