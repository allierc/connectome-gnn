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
    python run_GNN_optogenetics.py --max-parallel 80 --cluster l4

═══════════════════════════════════════════════════════════════════════════
EXPERIMENTAL RATIONALE
═══════════════════════════════════════════════════════════════════════════

This sweep tests a single methodological claim and its mechanistic basis:

    Targeted optogenetic perturbation of analytically-predicted cell types
    recovers the synaptic weights that naturalistic visual stimulation
    leaves unidentifiable, even when the perturbation is restricted to a
    small fraction of neurons.

The claim has three components, each tested by a different group of runs.

───────────────────────────────────────────────────────────────────────────
  WHAT'S ALREADY KNOWN (Table 1, the noise sweep)
───────────────────────────────────────────────────────────────────────────

The current paper shows that adding per-neuron Gaussian noise σ·ξ_i(t) to
EVERY neuron in the simulator lifts GNN weight recovery from R²_W = 0.89
(noise-free) to R²_W ≈ 1.00 (σ = 0.5). This establishes that activity
diversity beyond the visual manifold is the binding constraint on
identifiability — but it does so with a perturbation (per-neuron iid
stochastic noise on every cell) that no wet-lab can deliver.

The structural-nullspace analysis (app:degeneracy, scripts/structural_-
nullspace_table.py) shows that 71% of the connectome's edges live in a
columnar sum-zero kernel: dim ker(H) = 308,160 / 434,112 edges. The
kernel decomposes by cell type, with 9 cell types contributing > 10,000
free directions each.

What the existing paper does NOT establish:

  (a) Whether targeted, deterministic, biologically realizable
      perturbation (i.e. optogenetics) suffices to break the kernel.
  (b) Whether the analytical kernel decomposition predicts which cell
      types must be perturbed to lift recovery.
  (c) Which mechanism — spatial decorrelation across columns, or
      temporal richness of the drive — actually does the work.
  (d) Whether the resulting recovery is mechanism-specific (lifts the
      analytically-predicted edges) or a generic perturbation effect.

This sweep is designed to answer all four.

───────────────────────────────────────────────────────────────────────────
  WHAT THIS SWEEP ADDS
───────────────────────────────────────────────────────────────────────────

The 9 targets × 3 waveforms × 5 folds = 135 trained models populate a 2-D
design where each axis isolates one variable of the central claim:

  TARGET axis (cell-type selection) ─→ tests claim (b) and (d):
    • TmY15, Mi1, Tm4, Tm1, Mi4, TmY15+Mi1, L4, L5
        Vary the cell type perturbed at fixed waveform and amplitude.
        Predicts: ΔR²_W correlates with the cell type's null_dim
        contribution and/or its R²_W variant-rollout signature.
    • retina (R1–R8): NEGATIVE CONTROL for claim (b).
        Photoreceptors are in the IDENTIFIABLE bucket — k=1 everywhere,
        zero null_dim contribution. Perturbing 12.6% of the network this
        way should give ΔR²_W ≈ 0. A null result here is required to
        rule out "any opto helps" as a confound.
    • Tm1 vs Mi1 / Mi4 / TmY15: WITHIN-MOTIF NEGATIVE CONTROL for
      claim (d). Tm1 has a leverage profile near-identical to Mi1
      (rank 5/6 vs 2/5 across the two ranking metrics) but is NOT a
      T4 input. The mechanism-specific prediction is that Mi1 / Mi4 /
      TmY15 lift R²_W on T4-postsynaptic edges preferentially, while
      Tm1 lifts R²_W globally without the T4-edge selectivity.

  WAVEFORM axis (perturbation structure) ─→ tests claim (c):
    • '05' (white_noise σ=0.5):
        Per-neuron iid broadband. Both spatially decorrelated AND
        temporally rich. The "everything on" baseline.
    • 'heaviside' (column-distinct random telegraph):
        Per-neuron piecewise-constant signal with per-neuron transition
        times and per-neuron Unif(0,1) values. Spatially decorrelated
        but temporally piecewise-constant (low-frequency dominated).
        ★ Disentangles spatial decorrelation from temporal richness.
        The earlier column-COHERENT Heaviside (TmY15, R²_W = 0.87) gave
        no lift. If column-DISTINCT Heaviside lifts to ~0.91 (matching
        white noise), spatial decorrelation alone is the binding
        constraint. If it lands intermediate, both ingredients matter.
    • 'heaviside_05' (heaviside + σ=0.5 white noise additive):
        Step structure plus broadband, on the same column-distinct
        targets. Tests whether layered drives super-add over either
        component alone.

  CV FOLDS (5) provide error bars. The 0.02–0.05 effect sizes seen in
  TmY15 white-noise pilot are at the edge of fold-to-fold variance; 5
  folds plus a paired test is the minimum for statistical claims.

───────────────────────────────────────────────────────────────────────────
  EXPECTED OUTCOMES
───────────────────────────────────────────────────────────────────────────

Quantitative predictions, derived from the analytical kernel and the
TmY15 pilot, in order of confidence:

  HIGH CONFIDENCE (the paper's main claims rest on these):
    • Retina × white_noise: ΔR²_W ≈ 0 ± 0.02 across all five folds.
    • TmY15 × white_noise: R²_W ∈ [0.90, 0.93], V_rest ∈ [0.85, 0.92].
        (Pilot gave 0.91, 0.88. Sweep should reproduce.)

  MEDIUM CONFIDENCE (predictions follow from kernel analysis but
  haven't been observed yet):
    • Mi1 × white_noise: R²_W comparable-to-or-better-than TmY15
        (rank 2/5 on both metrics — the "consistent leader").
    • Tm4 × white_noise: largest single-type lift on global R²_W
        (rank 1 by R²_W variant — largest weight-space ambiguity).
    • TmY15+Mi1 × white_noise: lifts T4-postsynaptic edges
        super-additively; non-T4 edges roughly additively.
    • Stratified R²_W on T4-postsynaptic edges: lifts substantially
        for {Mi1, Mi4, TmY15, TmY15+Mi1}; lifts modestly or not at
        all for {Tm1, Tm4, L4, L5, retina}.

  LOW CONFIDENCE (genuinely open questions this sweep is designed
  to resolve):
    • TmY15 × heaviside: outcome decides spatial-vs-temporal
        question. Three regimes possible (see waveform axis above).
    • Tm1 lift size relative to Mi1: tests whether T4-motif
        membership matters beyond raw kernel coverage.
    • L4, L5 behavior: tests whether low-count, high-amplitude
        targets work as predicted by the leverage scatter.
    • heaviside_05 vs '05': tests whether step transients add
        information beyond what broadband noise provides.

───────────────────────────────────────────────────────────────────────────
  WHAT TO ANALYZE WHEN IT'S DONE
───────────────────────────────────────────────────────────────────────────

The trained models output per-fold metrics into results/metrics.txt; the
markdown summary (cv_blank50_opto_summary.md) aggregates across folds.
Beyond reading those tables, the following analyses are required for the
paper's claims to hold:

  1. STRATIFIED R²_W. Global R²_W is partly architecturally bounded
     (Known-ODE noise-free hits 0.96, GNN noise-free hits 0.89; the
     0.07 gap is not perturbation-recoverable). The interesting
     signal is in WHICH edges lift. Compute R²_W separately over:
       (a) all edges
       (b) singleton-group edges (k=1, identifiable from dynamics alone)
       (c) degenerate-group edges, source ∈ targeted cell type
       (d) degenerate-group edges, source ∉ targeted cell type
       (e) T4-postsynaptic edges (using T4_INPUT_TYPES below)
     Predicted: (c) lifts most for each opto target; (e) lifts only
     for T4-input opto targets; (a)–(d)–(e) lift for retina ≈ 0.

  2. PAIRED PER-FOLD TESTS. The effect sizes (0.02–0.05 on R²_W,
     0.10–0.15 on V_rest) are within fold-to-fold standard deviation
     but should be CONSISTENT across folds. For each (cond, baseline)
     pair, compute Δ_fold = R²(opto, fold) − R²(baseline, fold) and
     test sign(Δ) with a Wilcoxon signed-rank or paired-t. This is
     the right significance test, not the absolute SD.

  3. OUTLIER-RATE AUDIT. The "out%" columns in V_rest / τ R² report
     outlier-corrected scores. If outlier rate also changes between
     conditions, some apparent gains may be artifacts of more-
     aggressive exclusion. Recompute both R²s on the INTERSECTION
     of clean masks before drawing conclusions.

  4. LEVERAGE SCATTER. Plot ΔR²_W (and ΔR²_Vrest) per cell type
     against (i) null_dim (count) and (ii) 1 − R²_W^variant
     (amplitude). If the count axis dominates, ranking by null_dim
     was right. If the amplitude axis dominates, Tm4-style targeting
     wins. If both contribute, the OED frontier is two-dimensional.

  5. WAVEFORM CONTRAST PANEL. For TmY15 specifically: plot
     R²_W and R²_Vrest across {noise-free, heaviside-coherent (old),
     heaviside-distinct (new), white_noise, heaviside_05}. The
     ordering across waveforms decides claim (c).

═══════════════════════════════════════════════════════════════════════════

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
    python run_GNN_optogenetics.py --max-parallel 80 --cluster l4


"""

import argparse
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.cross.pipeline import submit_training_wave  # noqa: E402
from connectome_gnn.cross.summary_md import (  # noqa: E402
    COLUMNS, _collect_fold, _fmt_cell, _fmt_meansd,
)
from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path, get_data_root, load_data_root_from_json, log_path,
    set_data_root,
)
from _opto_cv_yaml import emit_fold_yaml, fold_dataset_name  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cv00-only', dest='cv00_only', action='store_true',
                    help='Train only fold 0 (cv00).')
parser.add_argument('--cluster', choices=['a100', 'l4'], default='l4',
                    help='LSF GPU queue (default l4).')
parser.add_argument('--runtime-min', type=int, default=2880,
                    help='Cluster runtime cap in minutes (default 2880 = 48h).')
parser.add_argument('--retrain', action='store_true',
                    help='Wipe models/, results/, tmp_training/ per fold and retrain.')
parser.add_argument('--max-parallel', type=int, default=128,
                    help='Maximum (cond, fold) jobs to submit per wave; each '
                         'wave blocks until it finishes before the next submits. '
                         'Default 128 (gpu_l4 has higher throughput than a100).')
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


_DS_REQUIRED = (
    'x_list_train/voltage.zarr',
    'x_list_test/voltage.zarr',
    'y_list_train.zarr',
    'y_list_test.zarr',
)


def _dataset_status(out_ds: str) -> str:
    """Classify dataset on disk: 'complete' | 'partial' | 'missing'.

    Matches run_generate_optogenetics._opto_dataset_complete: a dataset is
    'complete' iff all four zarrs exist. 'partial' = folder exists but at
    least one zarr is missing (likely currently generating, or aborted run).
    """
    base = graphs_data_path('fly', out_ds)
    if not os.path.isdir(base):
        return 'missing'
    if all(os.path.isdir(os.path.join(base, sub)) for sub in _DS_REQUIRED):
        return 'complete'
    return 'partial'


def _is_trained(config_file: str) -> bool:
    """True iff a best_model_with_*.pt checkpoint exists for this fold.

    Mirrors connectome_gnn.cross.pipeline._have_model so the runner skips
    the same folds the dispatcher would silently skip — but reports them
    up-front in the status summary instead.
    """
    return bool(glob.glob(
        os.path.join(log_path(config_file), 'models', 'best_model_with_*.pt')
    ))


def _emit_opto_summary_md():
    """Write <output_root>/log/cv_blank50_opto_summary.md.

    Mirrors connectome_gnn.cross.summary_md.emit_summary_md but iterates
    over our per-condition opto log dirs (which are not in CONDITION_BASES).
    Per-fold metrics are pulled from results/metrics.txt + results_test.log
    + results_rollout.log; folds without those artefacts render as '–'.
    Safe to call repeatedly — overwrites the file in place.
    """
    parent_dir = os.path.join(OUTPUT_ROOT, 'log', 'fly')
    sections = []
    for cond in CONDITIONS:
        base_name = f'flyvis_noise_free_blank50_opto_{cond}'
        fold_dirs, fold_data = [], []
        for i in FOLDS:
            fd = os.path.join(parent_dir, f'{base_name}_cv{i:02d}')
            if os.path.isdir(fd):
                fold_dirs.append(fd)
                fold_data.append(_collect_fold(fd))
        if not fold_dirs:
            continue
        lines = [f'## opto: `{cond}`', '',
                 f'**Log dir:** `{parent_dir}/{base_name}_cv*`', '']
        headers = ['Fold'] + [label for _, label, _ in COLUMNS]
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
        for fd, vals in zip(fold_dirs, fold_data):
            cv_tag = os.path.basename(fd).rsplit('_', 1)[-1]
            cells = [cv_tag] + [_fmt_cell(vals[k], fmt) for k, _, fmt in COLUMNS]
            lines.append('| ' + ' | '.join(cells) + ' |')
        summary_cells = ['**mean ± SD**']
        for k, _, fmt in COLUMNS:
            summary_cells.append('**' + _fmt_meansd([v[k] for v in fold_data], fmt) + '**')
        lines.append('| ' + ' | '.join(summary_cells) + ' |')
        lines.extend(['', '<details><summary>Per-fold log directories</summary>', ''])
        for fd in fold_dirs:
            lines.append(f'- `{fd}`')
        lines.extend(['', '</details>'])
        sections.append('\n'.join(lines))

    out_dir = os.path.join(OUTPUT_ROOT, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cv_blank50_opto_summary.md')
    with open(out_path, 'w') as f:
        f.write('# CV summary — `blank50_opto`\n\n')
        f.write(f'**Output root:** `{OUTPUT_ROOT}`\n\n')
        f.write(f'**Conditions found:** {len(sections)} '
                f'(of {len(CONDITIONS)} declared)\n\n')
        if not sections:
            f.write('_No matching log directories found._\n')
        else:
            f.write('\n\n'.join(sections))
            f.write('\n')
    print(f'  [md  ] {out_path}  ({len(sections)} condition(s))')
    return out_path


def _screen():
    """Classify every (cond, fold) pair and return wave candidates + buckets.

    Re-evaluated before every wave so datasets the generator finishes while
    a wave is running are picked up on the next iteration, and folds that
    just got trained are dropped.
    """
    yt_cfgs, trained, partial, missing = [], [], [], []
    per_cond = {c: {'ready': 0, 'trained': 0, 'partial': 0, 'missing': 0}
                for c in CONDITIONS}
    for cond in CONDITIONS:
        for fold in FOLDS:
            out_ds = fold_dataset_name(cond, fold)
            status = _dataset_status(out_ds)
            if status == 'partial':
                partial.append(out_ds)
                per_cond[cond]['partial'] += 1
                continue
            if status == 'missing':
                missing.append(out_ds)
                per_cond[cond]['missing'] += 1
                continue
            # complete: emit the per-fold YAML so we can check trained-ness via
            # log_path(cfg.config_file) and queue the cfg if untrained.
            yaml_path = emit_fold_yaml(cond, fold)
            cfg = NeuralGraphConfig.from_yaml(yaml_path)
            if not cfg.dataset.startswith('fly/'):
                cfg.dataset = 'fly/' + cfg.dataset
            if not cfg.config_file.startswith('fly/'):
                cfg.config_file = 'fly/' + cfg.config_file
            if not args.retrain and _is_trained(cfg.config_file):
                trained.append(out_ds)
                per_cond[cond]['trained'] += 1
                continue
            per_cond[cond]['ready'] += 1
            yt_cfgs.append(cfg)
    return yt_cfgs, trained, partial, missing, per_cond


def _print_screen(yt_cfgs, trained, partial, missing, per_cond, n_total):
    print(f'  ready to train : {len(yt_cfgs):3d} / {n_total}')
    print(f'  already trained: {len(trained):3d} / {n_total}'
          f'{"  (use --retrain to wipe & re-train)" if trained else ""}')
    print(f'  partial dataset: {len(partial):3d} / {n_total}'
          f'{"  (currently generating or aborted)" if partial else ""}')
    print(f'  not yet started: {len(missing):3d} / {n_total}'
          f'{"  (run run_generate_optogenetics.py)" if missing else ""}')
    print()
    print('per condition (ready / trained / partial / missing  of '
          f'{len(FOLDS)} fold(s)):')
    for cond, counts in per_cond.items():
        flag = ''
        if counts['ready'] == len(FOLDS):
            flag = '  ← full wave'
        elif counts['ready'] == 0 and counts['trained'] == len(FOLDS):
            flag = '  ← all trained'
        elif counts['ready'] == 0:
            flag = '  ← nothing to submit'
        print(f"  {cond:30s} {counts['ready']}/{counts['trained']}/"
              f"{counts['partial']}/{counts['missing']}{flag}")
    if partial:
        print()
        print('partial datasets (skipped — let the generator finish first):')
        for x in partial[:10]:
            print(f'  {x}')
        if len(partial) > 10:
            print(f'  ... +{len(partial) - 10} more')


# Step 1+2: re-screen before every wave so we pick up datasets the generator
# finished mid-loop, and drop folds that just got trained. The `attempted`
# set guards against folds whose training crashed without producing a
# checkpoint — re-submitting them every wave would loop forever.
wave_size = max(1, args.max_parallel)
n_total = len(CONDITIONS) * len(FOLDS)
attempted: set[str] = set()
wave_i = 0

while True:
    wave_i += 1
    print(f'\n--- screening before wave {wave_i} '
          f'({len(CONDITIONS)} cond × {len(FOLDS)} fold = {n_total}) ---')
    yt_cfgs, trained, partial, missing, per_cond = _screen()
    deferred = [c for c in yt_cfgs if c.config_file in attempted]
    yt_cfgs = [c for c in yt_cfgs if c.config_file not in attempted]
    print()
    _print_screen(yt_cfgs, trained, partial, missing, per_cond, n_total)
    if deferred:
        print()
        print(f'  ({len(deferred)} fold(s) submitted earlier this run without '
              f'producing a checkpoint — skipped, re-run script to retry)')

    if not yt_cfgs:
        if partial:
            print(f'\nNo runnable jobs right now — {len(partial)} dataset(s) '
                  f'still generating. Re-run when generation completes.')
        elif missing:
            print(f'\nNo runnable jobs — {len(missing)} dataset(s) not yet '
                  f'started. Run run_generate_optogenetics.py first.')
        else:
            print('\nAll done — every (cond, fold) is trained.')
        _emit_opto_summary_md()
        break

    chunk = yt_cfgs[:wave_size]
    print(f'\n=== wave {wave_i}: {len(chunk)} job(s) '
          f'(queue=gpu_{args.cluster}, runtime≤{args.runtime_min}min) ===')
    submit_training_wave(
        yt_cfgs=chunk,
        output_root=OUTPUT_ROOT,
        node_name=args.cluster,
        hard_runtime_limit_min=args.runtime_min,
        force_train=args.retrain,
    )
    attempted.update(c.config_file for c in chunk)
    # Refresh the markdown after each wave so progress is visible without
    # waiting for the full sweep to finish.
    _emit_opto_summary_md()
