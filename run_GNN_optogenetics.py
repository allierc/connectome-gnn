"""GNN training over the optogenetics sweep, 5-fold CV.

Mirrors run_GNN_unified_blank50.py: TARGETS / WAVEFORMS lists where commenting
a row drops conditions, per-fold CV YAMLs, argparse front-end. Reuses
connectome_gnn.cross.pipeline.submit_training_wave to dispatch jobs to LSF
(same path as the unified blank50 runner — no duplicated bsub logic here).

Pre-requisite: the corresponding dataset must already exist at
    graphs_data/fly/<flyvis_noise_free_blank50_opto_<cond>_cv<XX>>/
(Run run_generate_optogenetics.py first.)

Usage:
    python run_GNN_optogenetics.py                  # submit all to LSF (default)
    python run_GNN_optogenetics.py --cv00-only      # cv00 only
    python run_GNN_optogenetics.py --retrain        # wipe models/ and retrain
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.cross.pipeline import submit_training_wave  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path, get_data_root, load_data_root_from_json, set_data_root,
)
from _opto_cv_yaml import emit_fold_yaml, fold_dataset_name  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cv00-only', dest='cv00_only', action='store_true',
                    help='Train only fold 0 (cv00).')
parser.add_argument('--cluster', choices=['a100', 'l4'], default='l4',
                    help='LSF GPU queue (default l4).')
parser.add_argument('--runtime-min', type=int, default=1440,
                    help='Cluster runtime cap in minutes (default 1440 = 24h).')
parser.add_argument('--retrain', action='store_true',
                    help='Wipe models/, results/, tmp_training/ per fold and retrain.')
parser.add_argument('--max-parallel', type=int, default=20,
                    help='Maximum (cond, fold) jobs to submit per wave; each '
                         'wave blocks until it finishes before the next submits. '
                         'Default 20 (gpu_l4 has higher throughput than a100).')
args = parser.parse_args()


try:
    set_data_root(load_data_root_from_json())
except Exception:
    pass

OUTPUT_ROOT = get_data_root()


FOLDS = [0] if args.cv00_only else [0, 1, 2, 3, 4]


# Top-9 positive controls by null_dim. Comment a target row to drop both
# its variants; comment a waveform to drop that waveform for every target.
# Multi-type entries via '+' (e.g. 'TmY15+Mi1') target both simultaneously.
# Aliases (e.g. 'retina') expand per scripts/generate_opto_configs.py.
TARGETS = [
    # ──────────────────────────────────────────────────────────────────────
    # Single-cell-type runs — testing the kernel-prediction hypothesis
    # ──────────────────────────────────────────────────────────────────────

    'TmY15',       # null_dim=43,299 (rank  1) | R²_W=0.95 (rank 20)
                   # 217 cells (1.58%) | 14.05% kernel coverage | 9× leverage
                   # Tests: high-count, low-amplitude degeneracy.
                   # Predicted: many edges lifted, modest global R²_W change.
                   # Biological: wide-RF input to T4 (suppressive in PD).
                   # ★ T4 INPUT — pairs with Tm1 for within-motif contrast.

    'Mi1',         # null_dim=25,834 (rank  2) | R²_W=0.73 (rank  5)
                   # 217 cells (1.58%) | 8.38% kernel coverage | 5× leverage
                   # Tests: high count AND high amplitude — only top-5 on both.
                   # Predicted: substantial lift on both global R²_W and
                   #   per-edge stratified recovery.
                   # Biological: strongest excitatory input to T4 in motion-
                   #   detection circuit; mature Gal4 driver lines exist.
                   # ★ T4 INPUT — headline single-type experiment.

    # 'Tm3',       # null_dim=20,471 (rank  3) | R²_W=0.97 (rank 27)
                   # Skipped: high count but small amplitude (similar profile
                   #   to TmY15 but less extreme). TmY15 covers this regime.
                   # NOTE: Tm3 IS a T4 input — would strengthen the motif story
                   #   if added later, but redundant with TmY15 leverage-wise.

    'Tm4',         # null_dim=15,971 (rank  4) | R²_W=0.28 (rank  1)
                   # ~700 cells (~5%) | 5.18% kernel coverage
                   # Tests: largest weight-space ambiguity in the network.
                   # Predicted: largest single-type lift on global R²_W in
                   #   Table 1; fewer total edges affected than TmY15.
                   # Biological: NOT a T4 input — isolates the methodological
                   #   claim from the T4 biological story. Useful as a
                   #   "leverage works regardless of motif" demonstration.

    'Tm1',         # null_dim=15,525 (rank  5) | R²_W=0.74 (rank  6)
                   # ~700 cells (~5%) | 5.04% kernel coverage
                   # Tests: consistent top-6 on both rankings.
                   # ★ NEGATIVE CONTROL FOR T4 MOTIF — Tm1 is NOT a T4 input
                   #   (verify against fig 1a input current list: Mi1, Tm3,
                   #   T4c, T4b, T4a, T5c, C3, Mi10, Mi9, CT1(M10), Mi4,
                   #   TmY15 — no Tm1). Has near-identical leverage profile
                   #   to Mi1 (rank 2/5 vs 5/6 on the two metrics) but no
                   #   biological connection to motion detection.
                   # Predicted: global R²_W lifts (kernel still recovered),
                   #   but stratified R²_W on T4-postsynaptic edges does
                   #   NOT lift selectively. This contrast is what
                   #   distinguishes "kernel breaking helps recovery" from
                   #   "T4 input perturbation specifically helps T4 recovery".

    'Mi4',         # null_dim=14,439 (rank  6) | R²_W=0.91 (rank 13)
                   # 217 cells (1.58%) | 4.69% kernel coverage
                   # Tests: high count, modest amplitude.
                   # Biological: strongest suppressive input to T4 (paired
                   #   with Mi1 excitatory).
                   # ★ T4 INPUT — pairs with Mi1 to cover both ON/OFF
                   #   channels of the motion-detection input.

    # 'T4c',       # null_dim=12,564 (rank  7) | R²_W=0.99 (rank 32)
                   # Skipped: T4c is a *target* in the motion-detection story,
                   #   not an input. Perturbing T4 directly conflates "is T4
                   #   identifiable" with "are T4 inputs identifiable" —
                   #   biologically muddy. Run only if explicitly testing
                   #   identifiability of motion-output neurons themselves.

    # 'Mi9',       # null_dim=11,889 (rank  8) | R²_W=0.94 (rank 16)
                   # Skipped: Mi9 IS a T4 input but ranks below Mi1/Mi4 on
                   #   both metrics. Add later if Mi1+Mi4 leaves residual
                   #   T4-edge degeneracy and additional motif coverage helps.

    # 'Tm2',       # null_dim=11,068 (rank  9) | R²_W=0.81 (rank  9)
                   # Skipped: rank-9 on both metrics — most consistent
                   #   non-leader. Worth adding if you want a cleaner OED
                   #   frontier point in the middle of the leverage scatter.

    # ──────────────────────────────────────────────────────────────────────
    # Combined targeting — testing additivity and biological motif
    # ──────────────────────────────────────────────────────────────────────

    'TmY15+Mi1',   # joint perturbation: 434 cells (3.16%)
                   # Combined kernel coverage: 22.4% (43,299 + 25,834 / 308,160)
                   # Tests: do the two top-null_dim types stack additively in
                   #   recovery, or do they overlap on the same edges?
                   # Predicted: super-additive on T4-postsynaptic edges
                   #   (both are T4 inputs — covers wide-RF suppressive
                   #   TmY15 and narrow-RF excitatory Mi1 simultaneously),
                   #   additive elsewhere.
                   # ★ Headline combined experiment — covers the dominant T4
                   #   input motif while perturbing only ~3% of neurons.

    # ──────────────────────────────────────────────────────────────────────
    # Negative controls — testing what should NOT lift recovery
    # ──────────────────────────────────────────────────────────────────────

    'retina',      # R1–R8 photoreceptors: ~1,736 cells (~12.6% of network)
                   # All eight R-types are in the IDENTIFIABLE bucket
                   #   (k=1 everywhere — every R→postsynaptic edge is a
                   #   singleton). Combined null_dim contribution: 0.
                   # ★ NEGATIVE CONTROL FOR KERNEL HYPOTHESIS. Perturbing
                   #   identifiable cells should give ΔR²_W ≈ 0 despite
                   #   perturbing ~12.6% of neurons — *more* cells than any
                   #   single non-retina run.
                   # Predicted: no recovery lift. If a lift IS observed,
                   #   something is wrong (over-parameterization, training
                   #   confound, or kernel analysis incorrect).
                   # Biological: matches "drive every photoreceptor with
                   #   patterned light" — closest opto analogue to the
                   #   visual stimulus itself.

    # ──────────────────────────────────────────────────────────────────────
    # Mid-rank positive controls — testing the OED leverage scatter
    # ──────────────────────────────────────────────────────────────────────

    'L4',          # null_dim=6,136 (rank 13) | R²_W=0.73 (rank  4)
                   # ~700 cells (~5%) | 1.99% kernel coverage
                   # Tests: low count, high amplitude — inverse profile to
                   #   TmY15. Few degenerate directions but each is "deep"
                   #   in weight-space.
                   # Predicted: small lift in count of recovered edges,
                   #   moderate lift in global R²_W.
                   # Biological: NOT a T4 input — additional non-motif
                   #   data point for the leverage scatter.

    'L5',          # null_dim=7,463 (rank 11) | R²_W=0.62 (rank  3)
                   # ~700 cells (~5%) | 2.42% kernel coverage
                   # Tests: same low-count high-amplitude regime as L4.
                   # Predicted: similar to L4 — useful as a methodological
                   #   replicate. Together L4+L5 establish that the
                   #   amplitude-only regime is a real and reproducible
                   #   experimental signature.
                   # Biological: NOT a T4 input.
]


# ──────────────────────────────────────────────────────────────────────────
# CONTRAST GROUPS — for downstream stratified analysis
# ──────────────────────────────────────────────────────────────────────────
# The recovery-analysis script should compute stratified R²_W separately
# over edges sourced from each contrast group, for each trained opto model.
# The mechanistic claim of the paper rests on the differential pattern
# across these strata.

T4_INPUT_TYPES = [
    'Mi1', 'Tm3', 'T4a', 'T4b', 'T4c', 'T5c', 'C3', 'Mi10', 'Mi9',
    'CT1(M10)', 'Mi4', 'TmY15',
]
# Source: input-current panel of the FlyVis T4c figure (Lappalainen 2024 fig 1a).
# Edges with src ∈ T4_INPUT_TYPES and dst ∈ {T4a, T4b, T4c, T4d} form the
# "T4-motif edges" stratum.

NON_T4_INPUT_LEVERAGE_MATCHED_TYPES = [
    'Tm1',          # leverage profile near-identical to Mi1 — the key contrast
    'Tm4',          # high-amplitude, no T4 connection
    'L4', 'L5',     # low-count high-amplitude, no T4 connection
]
# These are the cell types whose perturbation is predicted to lift global
# R²_W but NOT preferentially lift T4-motif edges. The contrast between
# (T4 inputs lifting T4 edges) and (non-T4 inputs not lifting T4 edges) is
# the mechanism-specific claim.

IDENTIFIABLE_TYPES = [
    'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',  # 'retina'
    'Am', 'Lawf1', 'Mi3', 'T1', 'Tm5Y',
]
# Cell types in the identifiable bucket (no degenerate groups). Perturbing
# these should produce ΔR²_W ≈ 0. Used for negative-control runs.
WAVEFORMS = [
    '05',            # white_noise, σ = 0.5  (matches flyvis_noise_05 convention)
    'heaviside',     # 35 ON / 35 OFF stochastic square wave (column-distinct)
    'heaviside_05',  # heaviside + white-noise σ = 0.5 layered on top
]

CONDITIONS = [f'{t}_{w}' for t in TARGETS for w in WAVEFORMS]


def _dataset_voltage(cond: str, fold: int) -> str:
    return os.path.join(
        graphs_data_path('fly', fold_dataset_name(cond, fold)),
        'x_list_train', 'voltage.zarr',
    )


# Step 1: emit per-fold CV YAMLs into <output_root>/config/fly/ (idempotent).
print(f'emitting CV YAMLs for {len(CONDITIONS)} condition(s) × {len(FOLDS)} fold(s) ...')
yt_cfgs = []
for cond in CONDITIONS:
    for fold in FOLDS:
        ds_voltage = _dataset_voltage(cond, fold)
        if not os.path.isdir(ds_voltage):
            print(f'  SKIP {fold_dataset_name(cond, fold)}: dataset missing on disk')
            continue
        yaml_path = emit_fold_yaml(cond, fold)
        cfg = NeuralGraphConfig.from_yaml(yaml_path)
        # Mirror GNN_Main.py:249-250 — prefix dataset/config_file with fly/.
        if not cfg.dataset.startswith('fly/'):
            cfg.dataset = 'fly/' + cfg.dataset
        if not cfg.config_file.startswith('fly/'):
            cfg.config_file = 'fly/' + cfg.config_file
        yt_cfgs.append(cfg)

if not yt_cfgs:
    sys.exit('No runnable jobs — every dataset is missing on disk. '
             'Run run_generate_optogenetics.py first.')


# Step 2: submit jobs in waves of up to --max-parallel via the same LSF
# dispatcher used by run_GNN_unified_blank50.py. Each wave blocks on
# wait_for_cluster_jobs_with_metrics before the next wave submits.
n = len(yt_cfgs)
wave_size = max(1, args.max_parallel)
n_waves = (n + wave_size - 1) // wave_size
print(f'\nsubmitting {n} jobs to LSF in {n_waves} wave(s) of up to '
      f'{wave_size} (queue=gpu_{args.cluster}, runtime≤{args.runtime_min}min) ...')

for wave_i in range(n_waves):
    chunk = yt_cfgs[wave_i * wave_size:(wave_i + 1) * wave_size]
    print(f'\n=== wave {wave_i + 1}/{n_waves}: {len(chunk)} job(s) ===')
    submit_training_wave(
        yt_cfgs=chunk,
        output_root=OUTPUT_ROOT,
        node_name=args.cluster,
        hard_runtime_limit_min=args.runtime_min,
        force_train=args.retrain,
    )
