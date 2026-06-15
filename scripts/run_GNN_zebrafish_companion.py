#!/usr/bin/env python3
"""Zebrafish companion runner — relaunch the incomplete zebrafish_hd runs.

Thin wrapper over run_GNN_drosophila_cx_companion: it reuses that runner's
entire SSH/bsub submit + batched-bjobs monitor machinery, only overriding
GROUP="zebrafish" (so config/zebrafish/, log/zebrafish/ resolve) and the
default config set.

The default set is the 79 zebrafish runs that did NOT reach n_epochs in the
2026-06-13..15 batch (all 36 GNN runs OOM'd before the gradient-checkpoint
fix; the RNN runs were cut short). 5 GNN er_ei_6..10 runs are omitted — they
have no config in config/zebrafish/ so cannot be launched by name. Baked as a
static list because the faulty log folders are deleted before relaunch
(dynamic discovery from log/ would then find nothing).

Usage:
    python scripts/run_GNN_zebrafish_companion.py --cluster a100
    python scripts/run_GNN_zebrafish_companion.py --no-monitor
    python scripts/run_GNN_zebrafish_companion.py --config <names...>   # override
    python scripts/run_GNN_zebrafish_companion.py --all                 # every config/zebrafish/*.yaml
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_GNN_drosophila_cx_companion as base  # noqa: E402

# Retarget the shared runner at the zebrafish domain.
base.GROUP = "zebrafish"
base._ORDER_HINT = []  # no drosophila-specific ordering

# The 79 incomplete zebrafish runs to relaunch (config present in repo).
base._RERUN_GNN_FIXED = [
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_clustermove",
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_clustermove_1",
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_clustermove_2",
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_clustermove_3",
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_clustermove_4",
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_clustermove_5",
    "zebrafish_hd_si_gnn_ipn_917_v1_position_2d_leaky_clustermove",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_distance",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_distance_leaky",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_position_2d",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_position_2d_leaky",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_both",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_both_leaky",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_clustermove",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_clustermove_1",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_clustermove_2",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_clustermove_3",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_clustermove_4",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_clustermove_5",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_1",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_2",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_3",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_4",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_ei_1",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_ei_2",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_ei_3",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_ei_4",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_er_ei_5",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_freezeemb",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation_vfwd",
    "zebrafish_hd_si_ipn_917_v1_position_2d",
    "zebrafish_hd_si_ipn_917_v1_position_2d_cos000",
    "zebrafish_hd_si_ipn_917_v1_position_2d_cos025",
    "zebrafish_hd_si_ipn_917_v1_position_2d_cos050",
    "zebrafish_hd_si_ipn_917_v1_position_2d_cos075",
    "zebrafish_hd_si_ipn_917_v1_position_2d_cos100",
    "zebrafish_hd_si_ipn_917_v1_position_2d_leaky",
    "zebrafish_hd_si_ipn_917_v1_propriocep_distance",
    "zebrafish_hd_si_ipn_917_v1_propriocep_distance_leaky",
    "zebrafish_hd_si_ipn_917_v1_propriocep_mismatch",
    "zebrafish_hd_si_ipn_917_v1_propriocep_position_2d",
    "zebrafish_hd_si_ipn_917_v1_propriocep_position_2d_leaky",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_cos000",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_cos025",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_cos050",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_cos075",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_cos100",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_leaky",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_bs_1",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_bs_2",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_bs_3",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_bs_4",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_bs_5",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_cos000",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_cos025",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_cos050",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_cos075",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_cos100",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_kbins",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_mlpdec",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_mlpdec_1",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_mlpdec_2",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_mlpdec_3",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_mlpdec_4",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_mlpdec_5",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_bs_1",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_bs_2",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_bs_3",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_bs_4",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_bs_5",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_er",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_rep_1",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_rep_2",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_rep_3",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_rep_4",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_rep_5",
]

if __name__ == "__main__":
    raise SystemExit(base.main())
