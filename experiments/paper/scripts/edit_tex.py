"""Derive experiments/paper/main.tex from the Overleaf export (overleaf/main.tex):

  * figure paths figure/ -> figures/
  * the captions of the regenerated figures (Fig. 1, Supp. Figs. rollout,
    clustering, SIREN, FlyWire parameters) rewritten, in BLUE;
  * figures that still show the published current-form model (Fig. 2 d-f, the
    50% edge-ablation rollout) get a RED note in their caption;
  * every passage whose numbers or claims rest on the current-form GNN is
    wrapped in RED: it is to be rewritten by the authors, not here.

    python scripts/edit_tex.py
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "overleaf", "main.tex")
DST = os.path.join(HERE, "main.tex")

tex = open(SRC).read().replace("{figure/", "{figures/")

RED_OPEN = "{\\color{red}%\n"
RED_CLOSE = "}%\n"
BLUE = "\\color{blue}"


def once(hay, needle):
    assert hay.count(needle) == 1, (hay.count(needle), needle[:60])


def red_block(start, end):
    """Wrap everything from `start` (inclusive) to `end` (exclusive) in red."""
    global tex
    once(tex, start); once(tex, end)
    i = tex.index(start); j = tex.index(end, i)
    tex = tex[:i] + RED_OPEN + tex[i:j].rstrip("\n") + "\n" + RED_CLOSE + "\n" + tex[j:]


def red_sentence(sentence):
    global tex
    once(tex, sentence)
    tex = tex.replace(sentence, "\\textcolor{red}{" + sentence + "}")


def replace_figure(label, new_block):
    """Replace the whole figure environment carrying \\label{label}."""
    global tex
    pat = re.compile(r"\\begin\{figure\}(?:\[[^\]]*\])?\s*\\centering.*?\\end\{figure\}", re.S)
    hits = [m for m in pat.finditer(tex) if f"\\label{{{label}}}" in m.group(0)]
    assert len(hits) == 1, (label, len(hits))
    m = hits[0]
    tex = tex[:m.start()] + new_block.strip("\n") + tex[m.end():]


# ----------------------------------------------------------------------------- figures (blue captions)
replace_figure("fig:gnn_params_3col_noise_comparison", r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_gnn_params_3col_noise_comparison.pdf}
  \caption{%
    {\color{blue}\textbf{Circuit parameter extraction} from the general-form GNN
    ($g_\phi=\mathrm{MLP}(\mathbf{a}_i,\mathbf{a}_j,v_i,v_j)$, group lasso $25$) on Flyvis-217 under
    three model-noise regimes ($\sigma = 0$, $0.05$, $0.5$; fold cv00 of experiment 2).
    \textbf{(a, c, e)}~Learned against true synaptic weight $\hat{W}_{ij}$, one point per edge fitted by
    the template readout ($77\%$, $76\%$ and $96\%$ of the $434{,}112$ edges at $\sigma = 0$, $0.05$, $0.5$; the rest have a
    presynaptic neuron that never rises above the activity floor).
    \textbf{(b, d, f)}~Learned latent embeddings $\mathbf{a}_i \in \mathbb{R}^2$ of all neurons, coloured
    by ground-truth cell type.
    \textbf{(g, i, k)}~The aggregated message $m_i = \sum_j \hat W_{ij}\, g_\phi(\cdot)$ each neuron
    receives, learned against the generator's $\sum_j W_{ij}\,\mathrm{ReLU}(v_j)$, on the frames the
    metric $R^2_{m}$ is scored on (the general-form counterpart of the published $f_\theta$ panel).
    \textbf{(h, j, l)}~The per-edge message at observed $(v_i, v_j)$: learned
    $k_i \hat W_{ij}\, g_\phi(\mathbf{a}_i, \mathbf{a}_j, v_i, v_j)$ against the generator's
    $W_{ij}\,\mathrm{ReLU}(v_j)$ on $1{,}024$ random edges $\times$ $64$ random frames, $k_i$ the template
    readout's per-neuron gauge (the counterpart of the published $g_\phi$ panel).
    \textbf{(m, o, q)}~Resting potentials $V_i^{\mathrm{rest}}$.
    \textbf{(n, p, r)}~Time constants $\tau_i$.
    Each scatter is annotated with the $R^2$ and slope of learned against true quoted in
    \cref{tab:cv_gnn_vs_baselines} (identity-line $R^2$; for $\tau$ and $V^{\mathrm{rest}}$ the inlier
    $R^2$ with the all-neuron $R^2$ in parentheses, outliers beyond the dotted $\pm 0.1$~s / $\pm 0.2$
    bands in red and their fraction printed); the per-neuron $\tau$, $V^{\mathrm{rest}}$ and per-edge
    $\hat W_{ij}$ are the template readout's (Appendix~\ref{app:extraction}).}
  }
  \label{fig:gnn_params_3col_noise_comparison}
\end{figure}
""")

replace_figure("fig:rollout_3col_noise_comparison", r"""
\begin{figure}[ht!]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_rollout_3col_noise_comparison.pdf}
\caption{{\color{blue}\textbf{Rollout prediction.} General-form GNN (group lasso $25$) evaluated on
    central $217$-column Flyvis data under three model noise regimes ($\sigma = 0$, $\sigma = 0.05$,
    $\sigma = 0.5$; fold cv00 of experiment 2), tested on held-out stimuli, and compared against the
    noise-free Flyvis simulation of the same stimuli. Top row (\textbf{a}, \textbf{b}, \textbf{c}): GNN
    rollout traces for $12$ representative cell types over a $20$~s window ($1{,}000$ frames at
    $\Delta t = 20$~ms). Green: noise-free ground-truth voltage; black: GNN rollout prediction; red: the
    visual input of the photoreceptor row. Bottom row (\textbf{d}, \textbf{e}, \textbf{f}): rollout
    voltage against noise-free ground-truth voltage, pooled over all $(\text{neuron}, \text{frame})$
    pairs of the $8{,}000$-frame rollout as a log-density image. Pearson~$r$: per-neuron, Fisher-$z$
    pooled, on the full rollout.}
}
\label{fig:rollout_3col_noise_comparison}
\end{figure}
""")

replace_figure("fig:clustering_appendix", r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/fig_clustering_appendix.pdf}
\caption{{\color{blue}\textbf{Cell-type clusterability across feature spaces} of the general-form GNN
    (group lasso $25$, $\sigma = 0.05$, fold cv00 of experiment 2).
    Each panel scatters the $13{,}741$ neurons of the central $217$-column Flyvis model coloured by
    ground-truth cell type ($65$ classes). Gaussian Mixture Model (GMM) accuracy, Adjusted Rand Index
    (ARI) and Normalized Mutual Information (NMI) are quoted top-left of each panel, with $k$ the number
    of populated mixture components (out of $100$ requested). The GMM is fit on the $z$-scored raw
    features; UMAP is shown for visualisation only when the feature space is $>2$D.
    (\textbf{a}) Ground-truth $(\tau, V_{\mathrm{rest}})$ plus $8$ connectivity statistics of the true
    $\mathbf{W}$ ($10$D).
    (\textbf{b}) Same statistics computed from the learned $\hat\tau$, $\hat V_{\mathrm{rest}}$ and the
    template-readout $\hat{\mathbf{W}}$ over its fitted edges ($10$D).
    (\textbf{c}) Learned 2D node embedding $\mathbf{a}_i$ shown directly (no projection).
    (\textbf{d}) $\mathbf{a}_i$ concatenated with the learned
    $(\hat\tau, \hat V_{\mathrm{rest}}, \hat{\mathbf{W}}\text{-stats})$ ($12$D).}
}
    \label{fig:clustering_appendix}
\end{figure}
""")

replace_figure("fig:stim_rollout_inr", r"""
\begin{figure}[ht!]
\centering
\includegraphics[width=\textwidth]{figures/fig_stim_rollout_inr.pdf}
\caption{{\color{blue}\textbf{Joint GNN+INR (visual SIREN)}: the general-form GNN (group lasso $25$,
    one-step training) evaluated on central $217$-column Flyvis data with low model noise
    ($\sigma = 0.05$), $64\,000$ training frames (fold cv00 of experiment 11); rollout evaluated on the
    first $8\,000$ training frames since the SIREN cannot extrapolate to test-time indices. The SIREN
    output is shown with its mean and SD matched to the true stimulus, sign from the correlation (its raw
    output is defined up to the sign and scale the GNN's input weights absorb; the tester's least-squares
    correction would compress the learned range by the factor $r$).
    \textbf{(a)} INR-reconstructed visual stimulus on the R1 photoreceptor lattice ($217$ columns), ten
    frames spaced by $80$~ms from $10{,}000$~ms, $z$-scored per frame; top: ground truth; middle: INR;
    bottom: residual.
    \textbf{(b)} INR stimulus (black) against the true one (green) for $12$ photoreceptors (R1--R8 of one
    column, R1--R4 of another) over a $20$~s window ($1{,}000$ frames at $\Delta t = 20$~ms); the header
    gives the per-photoreceptor Pearson~$r$ over the $8{,}000$ frames, Fisher-$z$ pooled over the
    $1{,}736$ photoreceptors (mean $\pm$ SD). The SIREN's output enters the GNN squared, so it cannot be
    negative, and $f_\theta$ is free to read it with either sign; this run learned the inverted code
    (bright pixels $\to$ small output), so its zero floor becomes a ceiling at $0.73$ of the true range:
    the $7\%$ of photoreceptor--frames brighter than it all receive the same input, visible as the
    flattened bright plateaus. The sign is set at random by the training (four of the five folds learned
    it inverted).
    \textbf{(c)} GNN voltage rollout (black) against the noise-free ground truth (green) for $12$
    representative cell types over the same window; the header gives the per-neuron Pearson~$r$,
    Fisher-$z$ pooled over all $13{,}741$ neurons (mean $\pm$ SD).}
}
\label{fig:stim_rollout_inr}
\end{figure}
""")

replace_figure("fig:gnn_params_4col_flywire_comparison", r"""
\begin{figure}[ht!]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/fig_gnn_params_4col_flywire_comparison.pdf}
  \caption{%
    {\color{blue}\textbf{Circuit parameter extraction from the general-form GNN} (group lasso $25$)
    \textbf{on the four FlyWire connectome variants} with low model noise ($\sigma = 0.05$; fold cv00
    of experiment 7): e8 hybrid, e8 hybrid with proximal null edges (n.e.), FlyWire eye, and FlyWire eye
    with proximal null edges.
    \textbf{(a--d)}~Learned against true synaptic weight $\hat{W}_{ij}$ over the edges fitted by the
    template readout ($60\%$, $61\%$, $59\%$ and $62\%$ of the edges).
    \textbf{(e--h)}~Learned latent embeddings $\mathbf{a}_i \in \mathbb{R}^2$ of all neurons, coloured by
    ground-truth cell type.
    \textbf{(i--l)}~Resting potentials $V_i^{\mathrm{rest}}$ and \textbf{(m--p)}~time constants
    $\tau_i$ of the template readout, outliers beyond the dotted $\pm 0.2$ / $\pm 0.1$~s bands in red;
    $R^2$ (inlier, all-neuron in parentheses), slope and outlier fraction as in
    \cref{tab:zero_edge_inliers}.}
  }
  \label{fig:gnn_params_4col_flywire_comparison}
\end{figure}
""")

# Supp. Fig. Known-ODE rollout: redrawn in the published conventions from the published runs' bundles
# (scripts/fig_rollout_3col.py --known-ode); the published figure follows it for comparison.
_ko_old = tex[tex.index(r"\begin{figure}[ht!]" + "\n" + r"  \centering" + "\n" + r"  \includegraphics[width=\textwidth]{figures/fig_rollout_3col_noise_comparison_known_ode_nf_green.png}"):]
_ko_old = _ko_old[:_ko_old.index(r"\end{figure}") + len(r"\end{figure}")]
assert r"\label{fig:know_ode_rollout}" in _ko_old
_ko_new = r"""\begin{figure}[ht!]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_rollout_3col_noise_comparison_known_ode.pdf}
\caption{{\color{blue}\textbf{Known-ODE rollout prediction.} Known-ODE models (the published runs, fold
    cv00) evaluated on central $217$-column Flyvis data under three model noise regimes ($\sigma = 0$,
    $\sigma = 0.05$, $\sigma = 0.5$), tested on held-out stimuli, and compared against the noise-free
    Flyvis simulation of the same stimuli. Top row (\textbf{a}, \textbf{b}, \textbf{c}): Known-ODE rollout
    traces for $12$ representative cell types over a $20$~s window ($1{,}000$ frames at
    $\Delta t = 20$~ms). Green: noise-free ground-truth voltage; black: Known-ODE rollout prediction; red:
    the visual input of the photoreceptor row. Bottom row (\textbf{d}, \textbf{e}, \textbf{f}): rollout
    voltage against noise-free ground-truth voltage, pooled over all $(\text{neuron}, \text{frame})$ pairs
    of the $8{,}000$-frame rollout as a log-density image. Pearson~$r$: per-neuron, Fisher-$z$ pooled, on
    the full rollout.}
}
\label{fig:know_ode_rollout}
\end{figure}
"""
tex = tex.replace(_ko_old, _ko_new)          # the published figure, shown beside it for comparison, removed 2026-10-08

# ----------------------------------------------------------------------------- figures kept, red note
# Fig. 2: the published panels a-d with e, f redrawn from experiment 7 (scripts/fig_flywire_hybrid.py);
# the published figure follows it for comparison until Cedric drops it.
once(tex, r"\includegraphics[width=\textwidth]{figures/fig_flywire_hybrid.png}")
tex = tex.replace(r"\includegraphics[width=\textwidth]{figures/fig_flywire_hybrid.png}",
                  r"\includegraphics[width=\textwidth]{figures/fig_flywire_hybrid_new.png}")
once(tex, r"\textbf{(d-f)} Rollout comparisons to 20{,}000~ms unseen DAVIS stimuli. \textbf{(d)} Rollouts of FlyWire eye model (teal) vs. Flyvis rollout (orange). \textbf{(e)} Rollouts of GNN (black) vs. FlyWire eye simulations (green). \textbf{(f)} Rollouts of GNN with $761\%$ false-positive edges (black, FlyWire eye~+~n.e.) vs. FlyWire eye simulations (green).")
tex = tex.replace(r"\textbf{(d-f)} Rollout comparisons to 20{,}000~ms unseen DAVIS stimuli. \textbf{(d)} Rollouts of FlyWire eye model (teal) vs. Flyvis rollout (orange). \textbf{(e)} Rollouts of GNN (black) vs. FlyWire eye simulations (green). \textbf{(f)} Rollouts of GNN with $761\%$ false-positive edges (black, FlyWire eye~+~n.e.) vs. FlyWire eye simulations (green).",
                  r"\textbf{(d-f)} Rollout comparisons to 20{,}000~ms unseen DAVIS stimuli. \textbf{(d)} Rollouts of "
                  r"FlyWire eye model (teal) vs. Flyvis rollout (orange). {\color{blue}\textbf{(e)} Rollouts of the "
                  r"general-form GNN ($g_\phi=\mathrm{MLP}(\mathbf{a}_i,\mathbf{a}_j,v_i,v_j)$, group lasso $25$; "
                  r"black) vs. FlyWire eye simulations (green), fold cv00 of experiment 7, with the Fisher-$z$ pooled "
                  r"Pearson $r$ over all $50{,}412$ neurons and $8{,}000$ frames. \textbf{(f)} The same GNN trained "
                  r"with $661\%$ false-positive edges (FlyWire eye~+~n.e.) vs. FlyWire eye simulations.}")
once(tex, r"\label{fig:flywire}" + "\n\\end{figure}")
tex = tex.replace(r"\label{fig:flywire}" + "\n\\end{figure}", r"\label{fig:flywire}" + "\n\\end{figure}" + r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_flywire_hybrid.png}
  \caption{\textcolor{orange}{[For comparison only, to be removed] The published Fig.~2, panels e--f with the
    current-form GNN.}}
  \label{fig:flywire_published}
\end{figure}
""")
if os.path.exists(os.path.join(HERE, "figures", "fig_rollout_3col_noise_comparison_ablation50.pdf")):
    replace_figure("fig:rollout_3col_noise_comparison_ablation50", r"""
\begin{figure}[ht!]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_rollout_3col_noise_comparison_ablation50.pdf}
\caption{{\color{blue}\textbf{Rollout prediction under $50\%$ edge ablation.} General-form GNN (group
    lasso $25$; fold cv00 of experiment 2) evaluated on central $217$-column Flyvis data under three model
    noise regimes (noise-free $\sigma = 0$, $\sigma = 0.05$, $\sigma = 0.5$), tested on held-out stimuli,
    and compared against the noise-free ablated Flyvis simulation. For each test the simulator
    regenerates voltage traces with $50\%$ of the synaptic edges zeroed, and the same edge mask is applied
    to the trained GNN's learned weights $\hat{\mathbf{W}}$ before rollout (no retraining). Top row
    (\textbf{a}, \textbf{b}, \textbf{c}): GNN rollout traces for $12$ representative cell types over a
    $20$~s window ($1{,}000$ frames at $\Delta t = 20$~ms). Green: ablated ground-truth voltage; black: GNN
    rollout prediction; red: the visual input of the photoreceptor row. Bottom row (\textbf{d}, \textbf{e},
    \textbf{f}): rollout voltage against the noise-free ablated voltage, pooled over all
    $(\text{neuron}, \text{frame})$ pairs of the $8{,}000$-frame rollout as a log-density image.
    Pearson~$r$: per-neuron, Fisher-$z$ pooled, on the full rollout.}
}
  \label{fig:rollout_3col_noise_comparison_ablation50}
\end{figure}
""")
else:
    once(tex, r"\caption{\textbf{Rollout prediction under $50\%$ edge ablation.} GNN")
    tex = tex.replace(r"\caption{\textbf{Rollout prediction under $50\%$ edge ablation.} GNN",
                      r"\caption{\textcolor{red}{[Published current-form figure. The ablation rollouts of the "
                      r"general-form GNN (group lasso $25$) on the three \texttt{mask\_50} datasets have not "
                      r"been run: new test jobs needed.]} \textbf{Rollout prediction under $50\%$ edge "
                      r"ablation.} GNN")

# ----------------------------------------------------------------------------- red: to be rewritten
red_sentence(r"Second, the connectome serves well as \emph{binary} structural prior: the GNN tolerates a "
             r"400\% inflation of the edge set with random null edges and still recovers true synaptic "
             r"weights at $R^2=0.96$, while removing 50\% of true edges breaks recovery.")
red_sentence(r"Finally, measurement noise on voltage traces remains the principal open bottleneck: a "
             r"$\gamma = 0.1$ measurement noise leaves rollouts intact but degrades synaptic-weight "
             r"recovery from $R^2 = 0.99$ to $0.63$, and we flag this as the central challenge for "
             r"connectome-inverse methods on real data.")
# experiment 15 re-analysed the Known-ODE: noise-free V_rest R2 0.90 (1.7% outliers), so "R2 > 0.95" no longer holds
red_sentence("The parameter recovery over $> 4.5\\cdot 10^5$\nvalues is comparably accurate even at high model noise "
             "($R^2>0.95$, outliers $<10\\%$, \\cref{fig:know_ode_rollout,fig:known_ode_params_3col_noise_comparison}).")
red_block(r"\textbf{Graph neural network model}\label{method:GNN}", r"\textbf{Benchmark}")
red_block(r"\textbf{Results with and without the ODE given.}", "\\begin{figure}[t]\n  \\centering\n  \\includegraphics[width=\\textwidth]{figures/fig_flywire_hybrid.png}")
red_block(r"\textbf{Edge ablation as a mechanistic test.}", r"\textbf{Data degradation (measurement noise, partial data, unknown stimulus).}")
red_block(r"\textbf{Data degradation (measurement noise, partial data, unknown stimulus).}", r"% \paragraph{Robustness to measurement noise.}")
red_block(r"\textbf{GNN recovers fly visual system simulation with real synaptic connectivity.}", r"\textbf{GNN dynamical system inference overcomes false connectivity measurements.}")
red_block(r"\textbf{GNN dynamical system inference overcomes false connectivity measurements.}", r"\section{Discussion}")
red_block(r"We asked how and when interpretable, parameter-resolved circuit models can be recovered", r"\section{Limitations}")
red_block(r"Several idealizations of the simulator do not transfer to real recordings.", r"\begin{ack}")
red_block(r"The optimised networks of the GNN are $g_\phi$ and $f_\theta$ of", r"\input{tables/agent_GNN_HPO}")
red_block(r"The GNN of \cref{eq:gnn_ct} does not expose the membrane time", r"\subsection{Joint stimulus recovery with SIREN.}")
red_block(r"The unkown-stimulus results of \cref{tab:cv_cross_noise} is obtained by hiding the visual stimulus", r"% \subsection{Joint hidden-activity recovery with Instant-NGP}")

# a legend after the title
once(tex, r"\maketitle")
tex = tex.replace(r"\maketitle", r"\maketitle" + "\n" + r"\begin{center}\small\textcolor{blue}{Blue: tables and figures regenerated with the general-form GNN (group lasso 25) of the 2026 experiment campaign.} \textcolor{red}{Red: text, or a figure, still carrying the published current-form results, to be rewritten.}\end{center}" + "\n")

open(DST, "w").write(tex)
print("wrote", DST, len(tex.splitlines()), "lines")
