"""Supp. Fig.: held-out autoregressive rollout of the general-form GNN (group
lasso 25, experiment 2, fold cv00) at the three model-noise levels, against the
noise-free simulation of the same held-out stimuli (7,207 frames).

    python scripts/fig_rollout_3col.py

Data: <run>/results/rollout_bundle.npz and <run>/results_rollout.log (r).
Output: figures/fig_rollout_3col_noise_comparison.{pdf,png}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import LOG_ROOT  # noqa: E402
from rollout_panels import draw_figure  # noqa: E402

RUNS = ["flyvis_noise_free_blank50_condl25_cv00", "flyvis_noise_005_blank50_condl25_cv00",
        "flyvis_noise_05_blank50_condl25_cv00"]


def main():
    draw_figure([os.path.join(LOG_ROOT, r, *('results', 'rollout_bundle.npz')) for r in RUNS],
                [os.path.join(LOG_ROOT, r, *('results_rollout.log',)) for r in RUNS],
                'fig_rollout_3col_noise_comparison', truth='noise-free')


if __name__ == "__main__":
    main()
