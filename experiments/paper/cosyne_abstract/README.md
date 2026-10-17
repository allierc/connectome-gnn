# Cosyne abstract: material

Source paper: "Graph neural networks recover interpretable circuit models from neural activity"
(NeurIPS 2026 submission). Everything below is in `experiments/paper/`, paths relative to it.

Write the abstract in this folder.

## The paper

| what | file |
|---|---|
| compiled paper (built 2026-10-14, commit 5092617a) | `main.pdf` |
| LaTeX source | `main.tex` (figures from `figures/`, tables from `tables/`) |
| bibliography | `references.bib` |
| figure style rules | `FIGURE_STYLE.md` |

## Main text

| item | file | made by |
|---|---|---|
| Figure 1: circuit parameters (W, tau, V_rest) recovered by the general-form GNN, model noise 0.05 | `figures/fig_gnn_params_noise_005.pdf` | `scripts/fig_gnn_params_noise_005.py` |
| Figure 2: hybrid Flyvis-FlyWire connectome, model inference | `figures/fig_flywire_hybrid_new.{pdf,png}` | `scripts/fig_flywire_hybrid.py` |
| Table 1: GNN against baselines, three model-noise levels | `tables/cv_table_gnn_vs_baselines.tex` | `scripts/table_1_gnn_vs_baselines.py` |
| Table 2: GNN recovery on hybrid connectome variants (null-edge augmentation) | `tables/flybrid_inliers.tex` | `scripts/table_2_flybrid.py` |

Every number in Tables 1, 2 and Supp. Tables 4 and 7 is in `tables/table_numbers.json`, as
[mean, SD] over the five cross-validation folds (SD with ddof 0).

## Supplementary

| item | file |
|---|---|
| Supp. Fig. 3: visual input and per-cell-type activity on the hexagonal lattice | `figures/fig_stimulus_hexagons.png` |
| Supp. Fig. 4: simulated activity across noise regimes | `figures/fig_simulations.png` |
| Supp. Fig. 5: ground-truth parameter distributions | `figures/fig_ground_truth_distributions.png` |
| Supp. Fig. 6: rollout, three model-noise levels | `figures/fig_rollout_3col_noise_comparison.pdf` |
| Supp. Fig. 7: cell-type clusterability across feature spaces | `figures/fig_clustering_appendix.pdf` |
| Supp. Fig. 8: rollout under 50% edge ablation | `figures/fig_rollout_3col_noise_comparison_ablation50.pdf` |
| Supp. Fig. 9: joint GNN + visual SIREN (unknown stimulus) | `figures/fig_stim_rollout_inr.pdf` |
| Supp. Fig. 10: parameters on the four FlyWire connectome variants | `figures/fig_gnn_params_4col_flywire_comparison.pdf` |
| Supp. Fig. 11: hybrid model heterogeneous connectivity against Flyvis templated connectivity | `figures/fig_flywire_hybrid_SI.png` |
| Supp. Fig. 12: per-neuron least-squares parameter recovery, noise-free | `figures/fig_lstsq_param_recovery_flyvis_noise_free_blank50_cv00.pdf` |
| Supp. Fig. 13: architectures of the three black-box baselines | `figures/fig_baseline_model_arch.pdf` |
| Supp. Fig. 14: Known-ODE rollout | `figures/fig_rollout_3col_noise_comparison_known_ode.pdf` |
| Supp. Fig. 15: Known-ODE parameters | `figures/fig_known_ode_params_3col_noise_comparison_new.pdf` |
| Supp. Fig. 16: stimulus context length against baseline fidelity | `figures/fig_stimulus_ctx_pearson.pdf` |
| Supp. Fig. 17: rollout, MLP baseline | `figures/fig_rollout_3col_noise_comparison_mlp.pdf` |
| Supp. Fig. 18: Jacobian of the MLP baseline against the true dynamics | `figures/fig_jacobian_l1_comparison.pdf` |
| Supp. Fig. 19: low-rank reconstruction of the activity matrix | `figures/fig_svd_variance_explained.pdf` |
| Supp. Fig. 20: rollout, EED baseline | `figures/fig_rollout_3col_noise_comparison_eed.pdf` |
| Supp. Tab. 3: coregistration of Flyvis and FlyWire (FAFB v783) | `tables/flyrewire_stats.tex` |
| Supp. Tab. 4: GNN across degraded data, model noise 0.05 | `tables/cv_table_gnn_cross_noise.tex` |
| Supp. Tab. 5: training parameters of the general-form GNN | `tables/agent_GNN_HPO.tex` |
| Supp. Tab. 6: training parameters of the Known-ODE | `tables/known_ode_training.tex` |
| Supp. Tab. 7: Known-ODE across degraded data | `tables/cv_table_known_ode_conditions.tex` |
| Supp. Tab. 8: baseline models, three model-noise levels | `tables/blank50_baselines_table.tex` |
| Supp. Tab. 9: compute resources | `tables/table_compute_resources.tex` |

Other files in `figures/` are earlier versions or unused panels. They are kept for reference and
are not in the paper. `figures/fig_jacobian_l1_comparison.png` (6.5 MB) is not in git; its PDF is.

## Rebuilding

The scripts read the trained runs under `/groups/saalfeld/home/allierc/GraphData/log/fly/`. That
data is on the Janelia cluster file system and not in git, so the scripts run only where
`/groups` is mounted. `scripts/build.sh` rebuilds the tables and compiles `main.pdf`.

## Status of the numbers (2026-10-17)

- The parameter fit reads every sampled frame (no activity filter). R2 of W is computed over the
  edges the fit could fit (97-100% of the edges); its parenthesis in the tables is the percentage of
  outliers.
- Three rows were trained on ten folds (cv00-cv09) because some folds have a rollout outlier: the
  rollout from the initial voltages diverges (rollout r below 0.5) while the one-step r stays at 0.99
  or more. Outlier folds among the ten: Table 1 GNN at model noise 0.5, 2 (cv04, cv09); Supp. Tab. 4
  unknown stimulus, 3 (cv03, cv04, cv06); Supp. Tab. 4 +400% null edges, 2 (cv02, cv08).
- The tables keep five folds per row: each outlier fold of cv00-cv04 is replaced by the next of
  cv05-cv09 that is not one (`FOLD_SWAPS` in `scripts/table_common.py`), and the captions and the
  Results text say so.
