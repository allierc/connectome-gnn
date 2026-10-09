"""Supp. Fig.: rollout of the same GNN with 50% of the edges ablated (no
retraining), against the noise-free simulation of the ablated circuit
(7,999 frames).

    python scripts/fig_rollout_ablation50.py

The tester zeroed the masked half of the learned W (ablation_mask.pt of
flyvis_noise_free_mask_50) and rolled out on that dataset's noise-free
simulation; its bundle and log are kept in <run>/ablation50/ (a later -o test
clears results/*_on_*).
Data: <run>/ablation50/rollout_bundle_on_noise_free_mask_50.npz and
<run>/ablation50/results_rollout_on_noise_free_mask_50.log (r).
Output: figures/fig_rollout_3col_noise_comparison_ablation50.{pdf,png}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import LOG_ROOT  # noqa: E402
from rollout_panels import draw_figure  # noqa: E402

RUNS = ["flyvis_noise_free_blank50_condl25_cv00", "flyvis_noise_005_blank50_condl25_cv00",
        "flyvis_noise_05_blank50_condl25_cv00"]


def main():
    draw_figure([os.path.join(LOG_ROOT, r, *('ablation50', 'rollout_bundle_on_noise_free_mask_50.npz')) for r in RUNS],
                [os.path.join(LOG_ROOT, r, *('ablation50', 'results_rollout_on_noise_free_mask_50.log')) for r in RUNS],
                'fig_rollout_3col_noise_comparison_ablation50', truth='noise-free ablated')


if __name__ == "__main__":
    main()
