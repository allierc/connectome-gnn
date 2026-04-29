"""GNN YT-only cross-check with UNIFIED-winner HPs on 50%-blank-prefix data.

Expanded from the 3-condition prototype to all 8 canonical conditions
plus 3 new variants (noise_005_020, removed_pc_50, hidden_020_ngp).

Goal: test whether 50% blank-prefix per video sequence recovers V_rest_R²
from the ~0 ceiling we hit on the standard YT CV table without blanks.

Config files used (relative to repo config/fly/):

  simulation-block sources (one per condition):
    flyvis_noise_free.yaml
    flyvis_noise_005.yaml
    flyvis_noise_05.yaml
    flyvis_noise_005_010.yaml
    flyvis_noise_005_020.yaml               # NEW — template: flyvis_noise_005_010.yaml
    flyvis_noise_005_null_edges_pc_400.yaml
    flyvis_noise_005_removed_pc_20.yaml
    flyvis_noise_005_removed_pc_50.yaml     # NEW — template: flyvis_noise_05_removed_pc_20.yaml
    flyvis_noise_005_stride_5.yaml
    flyvis_noise_005_hidden_010_ngp.yaml
    flyvis_noise_005_hidden_020_ngp.yaml    # NEW — template: flyvis_noise_005_hidden_010_ngp.yaml

  HP yaml (graph_model + training blocks, applied to every condition):
    flyvis_unified_winner.yaml

  emitted CV yamls (55 total, written to <output_root>/config/fly/):
    {base}_blank50_unified_cv{00..04}.yaml

    datasets: <output_root>/graphs_data/fly/<base>_blank50_cv{00..04}/
    tex out : <output_root>/log/cv_blank50_unified_rows.tex

Wall-clock per GNN: ~1 h on a100.
Total training units: 11 conditions × 5 folds = 55 GNNs.

The 3 NEW conditions (noise_005_020, removed_pc_50, hidden_020_ngp) also need
to be registered in CONDITIONS in src/connectome_gnn/cross/yaml_io.py so
the emitter picks them up. Until their base yamls + registry entries
exist, those rows will be skipped with a warning rather than crash.

This script does NOT generate data — it fails fast if the datasets are
missing. Run run_generate_YT_data_blank50.py first (or use the bash
wrapper run_GNN_blank50_pipeline.sh).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions, emit_summary_md
from connectome_gnn.utils import load_data_root_from_json

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--retrain', action='store_true',
                     help='Re-run TRAIN: wipe models/, results/, tmp_training/ '
                          'and cluster_train.log per fold, then retrain from '
                          'scratch. Without this flag the [skip] guard keeps '
                          'every already-trained fold and only fills in folds '
                          'whose models/ is missing.')
_parser.add_argument('--retest', action='store_true',
                     help='Re-run TEST (rollout): wipe results_rollout.log per '
                          'fold and rerun data_test. Independent of --retrain '
                          'and --replot.')
_parser.add_argument('--replot', action='store_true',
                     help='Re-run PLOT (parameter scatters): wipe '
                          'results/metrics.txt per fold and rerun data_plot. '
                          'Independent of --retrain and --retest.')
_parser.add_argument('--redo-all', dest='redo_all', action='store_true',
                     help='Shortcut for --retrain --retest --replot. Wipes '
                          'every phase artefact and runs the full pipeline '
                          'from scratch.')
_parser.add_argument('--skip-test-plot', dest='skip_test_plot', action='store_true',
                     help='Submit the training wave only — suppress the '
                          'test+plot wave entirely. Combine with --retrain to '
                          'retrain from scratch without immediately rolling '
                          'out / plotting the new model.')
_parser.add_argument('--cluster', choices=['a100', 'l4'], default=None,
                     help='Override the LSF GPU queue for ALL conditions '
                          '(a100 or l4). When unset, uses the per-script '
                          'default.')
_parser.add_argument('--cv00-only', dest='cv00_only', action='store_true',
                     help='Limit the CV grid to fold 0 (cv00) only — useful '
                          'when iterating on plot rendering with --replot, '
                          'so a single fold is re-rolled out per condition '
                          'instead of all five.')
_args = _parser.parse_args()

_force_train = bool(_args.retrain or _args.redo_all)
_force_test  = bool(_args.retest  or _args.redo_all)
_force_plot  = bool(_args.replot  or _args.redo_all)
_n_folds     = 1 if _args.cv00_only else 5


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
    # Match the standalone flyvis_noise_005_blank50 run (which uses the config
    # default True); emit_one() otherwise forces False for the YT-VOS pipelines.
    'skip_short_videos': True,
}

#  condition name                        ->  LSF queue (gpu_<node>)
# Comment out a row to drop it from the run. The dict is the single source of
# truth — CONDITION_FILTER is derived from its keys, NODE_PER_CONDITION from
# the full mapping.
CONDITION_NODES = {
    # === AR(1) measurement-noise sweep (6-point dose-response, blank50 + gamma=0.10) ===
    # Low rho brackets the indicator-kinetics regime (ASAP3 ~ 0.25,
    # GCaMP6f rise ~ 0.50, GCaMP6f decay ~ 0.75); high rho probes the
    # asymptote toward the noise_005 ceiling (per-frame derivative noise
    # scales as (1-rho), so at rho=0.99 it is 1% of the i.i.d. case).
    # The rho=0 control is the existing flyvis_noise_005_010 condition
    # under blank50 overrides (commented out below; uncomment if needed).
    # 'flyvis_noise_005_010_blank50_ar1_rho25': 'l4',
    # 'flyvis_noise_005_010_blank50_ar1_rho50': 'l4',
    # 'flyvis_noise_005_010_blank50_ar1_rho75': 'l4',
    # 'flyvis_noise_005_010_blank50_ar1_rho90': 'l4',
    # 'flyvis_noise_005_010_blank50_ar1_rho95': 'l4',
    # 'flyvis_noise_005_010_blank50_ar1_rho99': 'l4',
    # === stride_5 (speed) baseline — re-enabled with AR(1) sweep for the
    # learned_ode_params.pt re-emit pass. Uncomment more rows below to expand
    # to the full canonical baseline set in a follow-up run.

    # --- other non-AR(1) baselines (paused; uncomment to add to this run) ---
    'flyvis_noise_free':                    'a100',
    'flyvis_noise_005':                     'a100',
    'flyvis_noise_05':                      'a100',
    # 'flyvis_noise_005_010':                 'a100',  # = AR(1) rho=0 control under blank50 overrides
    # 'flyvis_noise_005_020':                 'a100',
    # 'flyvis_noise_005_null_edges_pc_400':   'a100',
    # 'flyvis_noise_005_removed_pc_20':       'a100',
    # 'flyvis_noise_005_removed_pc_50':       'a100',
    # 'flyvis_noise_005_hidden_010_ngp':      'a100',
    # 'flyvis_noise_005_hidden_020_ngp':      'a100',
    # 'flyvis_noise_005_stride_5':             'a100',
}

CONDITION_FILTER     = list(CONDITION_NODES.keys())
NODE_PER_CONDITION   = CONDITION_NODES


# Per-condition DAL overrides. Expensive conditions (5x edges / NGP encoder)
# need a smaller gradient budget to keep wall time under ~6h per fold.
# Others use the default data_augmentation_loop=500.
DAL_OVERRIDES = {
    'flyvis_noise_005_null_edges_pc_400': 100,   # 2.17M edges -> ~5.8h instead of ~29h
    'flyvis_noise_005_hidden_010_ngp':    100,   # NGP encoder + anchors loss; matches winner DAL
    'flyvis_noise_005_hidden_020_ngp':    100,   # same as _010_ngp
}

# Per-condition HP yaml overrides. stride_5 (BPTT with bs=1, coeff_g_phi_diff=9000,
# coeff_g_phi_norm=0.1) and the hidden_*_ngp conditions (NGP-T + anchors training:
# lr_NNR_f, coeff_anchor_voltage, n_anchor, alternate_training) have structurally
# different training recipes that the uniform noise_005-style HP yaml cannot
# represent — route them to their own winner yamls instead. Makes
# `python run_GNN_unified_blank50.py` equivalent to running
# patch_blank50_pending_cv_yamls.py after the uniform emit.
HP_YAML_OVERRIDES = {
    'flyvis_noise_005_stride_5':       'flyvis_noise_005_stride_5_winner',
    'flyvis_noise_005_hidden_010_ngp': 'flyvis_noise_005_hidden_010_ngp_anchors_winner',
    'flyvis_noise_005_hidden_020_ngp': 'flyvis_noise_005_hidden_020_ngp_anchors_winner',
}


run_all_conditions(
    hp_source='uniform',
    suffix='blank50_unified',
    hp_yaml='flyvis_unified_blank50_winner',
    node_name=_args.cluster or 'a100',
    hard_runtime_limit_min=2880,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    data_augmentation_loop_overrides=DAL_OVERRIDES,
    hp_yaml_overrides=HP_YAML_OVERRIDES,
    force_train=_force_train,
    force_test=_force_test,
    force_plot=_force_plot,
    skip_test_plot=_args.skip_test_plot,
    n_folds=_n_folds,
    conditions_per_wave=3,
    emit_tex=False,
)

# Per-condition CV summary markdown (rows per fold + mean±SD), written to
# <output_root>/log/cv_blank50_unified_summary.md.
emit_summary_md('blank50_unified',
                output_root=os.environ.get('GNN_OUTPUT_ROOT')
                            or load_data_root_from_json(),
                n_folds=_n_folds)


# ---------------------------------------------------------------------------
# Example invocations  (run from /workspace/connectome-gnn)
# ---------------------------------------------------------------------------
#
# # Default: resume — skip already-trained folds, run test+plot only on
# # folds whose results_rollout.log / results/metrics.txt is missing.
# python run_GNN_unified_blank50.py --cluster a100
#
# # Training partially done but plots are missing: do NOT pass --retrain
# # (the [skip] guard will keep the trained folds and retrain only the
# # wiped ones). Force the test+plot wave for every fold so all parameter
# # PNGs land:
# python run_GNN_unified_blank50.py --retest --replot --cluster a100
#
# # Only re-render the parameter plots (rollouts already present):
# python run_GNN_unified_blank50.py --replot --cluster a100
#
# # Full clean re-run (wipe + retrain + retest + replot every fold):
# python run_GNN_unified_blank50.py --redo-all --cluster a100
#
# # Train only — suppress the rollout/plot wave entirely:
# python run_GNN_unified_blank50.py --skip-test-plot --cluster a100
#
# # Force fresh training but defer plotting to a later --replot pass:
# python run_GNN_unified_blank50.py --retrain --skip-test-plot --cluster a100
# python run_GNN_unified_blank50.py --replot --cv00-only --cluster a100
