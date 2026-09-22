# Handoff — recomputing the recurrent-MLP Jacobian row (Vzfg Q4)

Reviewer Vzfg asked us to apply the Supp. Fig. 18 Jacobian test to *every* model,
not only the recurrent MLP. The GNN and Known-ODE rows are done; the MLP row is
blank in the rebuttal because its checkpoints are not reachable from this
filesystem. This note is what's needed to fill it.

## What we need from you

The two MLP baseline runs from the submitted experiments, currently under
`/groups/saalfeld/home/kumarv4/repos/connectome-gnn/log/fly/`:

| run | used for |
|---|---|
| `flyvis_noise_free_mlp_blank50_l1_0` | the Supp. Fig. 18 no-regularisation panel |
| `flyvis_noise_free_mlp_blank50_l1_1em6` | the L1 panel |

For each, only two things are needed:

```
log/fly/<run>/config.yaml
log/fly/<run>/models/best_model_with_0_graphs_*.pt
```

Copy them anywhere readable under `$GNN_OUTPUT_ROOT/log/fly/<run>/`, keeping that
layout — the loader finds the config and picks the highest-`sort_key` checkpoint,
the same selection GNN_PlotFigure calls "best". Nothing else from the original
tree is required, and no retraining is involved.

## Then run

```bash
cd /workspace/connectome-gnn-cx
GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
python figures/flyvis/fig_jacobian_gnn_knownode.py \
    --n-frames 200 \
    --mlp-config flyvis_noise_free_mlp_blank50_l1_0 \
    --mlp-frames 20
```

Output: `figures/flyvis/results_jacobian_gnn_knownode.json`, with
`edges.recurrent_mlp`, `diagonal.recurrent_mlp` and `off_graph_fraction` added.
Those three go straight into the Q4 table in `reply_all.tex`.

## Why `--mlp-frames` is separate

The GNN and Known-ODE Jacobians are supported on the connectome, so only the
434,112 edge entries are ever computed and 200 frames is cheap. The MLP is
unconstrained, so its Jacobian is dense (13,741 x 13,741) and has to be built by
a chunked backward pass per frame. That is the whole point of the comparison —
the fraction of |J| landing on unconnected pairs is zero by construction for our
two models and is the quantity that distinguishes the MLP — but it costs roughly
a minute per frame. Start at 20 and raise it if the numbers look unstable.

## What is being computed

At each held-out frame, for every model:

```
J_ij = d(dv_i/dt) / dv_j
```

Ground truth from Eq. 1 is `J_ij = W_ij * ReLU'(v_j) / tau_i`, `J_ii = -1/tau_i`.
Known-ODE uses the same closed form in the learned `W_hat, tau_hat`. GNN and MLP
are differentiated by autograd — **not** finite differences, which the ReLU kinks
make unstable. All models are evaluated at the *same* sampled frames (fixed seed)
so the comparison is like-for-like.

Reported per model: R^2 against the ground-truth J, Pearson r, sign agreement,
and for the MLP the off-graph fraction of `sum_{i != j} |J_ij|`.

## Current numbers (200 frames, noise-free blank50, fold cv00)

| block | model | R² vs GT J | Pearson r | sign agr. |
|---|---|---|---|---|
| off-diagonal | Known-ODE | 0.977 | 0.988 | 0.882 |
| off-diagonal | GNN | 0.970 | 0.988 | 0.878 |
| diagonal | Known-ODE | 0.944 | 0.975 | 1.000 |
| diagonal | GNN | 0.865 | 0.963 | 1.000 |

One caveat worth knowing before reading the MLP row against these: the condition
is noise-free (sigma = 0) to match Supp. Fig. 18's panels, and that is the
degenerate regime the rebuttal explicitly restricts against elsewhere. Matching
Supp. Fig. 18 is what the reviewer asked for, so the comparison stays there.

Script: `figures/flyvis/fig_jacobian_gnn_knownode.py`
Context: `neurips_review/reply_all.tex`, Reviewer Vzfg, Q4.
