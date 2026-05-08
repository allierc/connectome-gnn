"""Pre-generate the optogenetics sweep datasets across all CV folds.

Mirror of run_generate_blank50.py for the opto experiment. For each enabled
(condition × fold) pair, emits a per-fold YAML into <data_root>/config/fly/
and calls add_optogenetics_stimulus(config) on it.

Final output: enabled_conditions × len(FOLDS) datasets at
    <output_root>/graphs_data/fly/<base>_opto_<target>_<wf>_cv<XX>/

Comment lines in CONDITIONS or shrink FOLDS to skip individual runs.
Downstream training runner: run_GNN_optogenetics.py.

No CLI flags — edit CONDITIONS / FOLDS or scripts/generate_opto_configs.py
to change the sweep.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.optogenetics import add_optogenetics_stimulus  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path, load_data_root_from_json, set_data_root,
)
from _opto_cv_yaml import (  # noqa: E402
    BASELINE_PREFIX, emit_fold_yaml, emit_gen_yaml, fold_dataset_name,
)


FOLDS = [0, 1, 2, 3, 4]

# Top-9 positive controls by null_dim. Comment a target row to drop both its
# variants; comment a waveform string to drop that waveform across all targets.
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


try:
    set_data_root(load_data_root_from_json())
except Exception:
    pass


def _baseline_exists(fold: int) -> bool:
    voltage = os.path.join(
        graphs_data_path('fly', f'{BASELINE_PREFIX}_cv{fold:02d}'),
        'x_list_train', 'voltage.zarr',
    )
    return os.path.isdir(voltage)


def _opto_dataset_complete(out_ds: str) -> bool:
    """True iff <out_ds>/ has both splits' x_list voltage and y_list zarrs.

    These four artefacts together are sufficient for the trainer to load the
    dataset (see load_flyvis_data); their joint presence implies a clean
    generation run completed past both splits' ODE integration and y_writer.
    """
    base = graphs_data_path('fly', out_ds)
    return all(
        os.path.isdir(os.path.join(base, sub))
        for sub in (
            'x_list_train/voltage.zarr',
            'x_list_test/voltage.zarr',
            'y_list_train.zarr',
            'y_list_test.zarr',
        )
    )


missing = [f for f in FOLDS if not _baseline_exists(f)]
if missing:
    sys.exit(
        f"baseline datasets missing for folds {missing} "
        f"(expected at {graphs_data_path('fly', BASELINE_PREFIX + '_cv{XX}')}). "
        f"Run the unified blank50 generator first."
    )


for cond in CONDITIONS:
    # Step 1: emit the per-condition '_gen' template into <data_root>/config/fly/
    # (mirrors generate_all_yt_data's behaviour from run_generate_blank50.py).
    gen_yaml = emit_gen_yaml(cond)
    print(f'\n>>> condition={cond}  gen config: {gen_yaml}', flush=True)

    for fold in FOLDS:
        # Step 2: per-fold safety — re-confirm the source data folder exists.
        src_dir = graphs_data_path('fly', f'{BASELINE_PREFIX}_cv{fold:02d}')
        src_voltage = os.path.join(src_dir, 'x_list_train', 'voltage.zarr')
        if not os.path.isdir(src_voltage):
            print(f'  SKIP cv{fold:02d}: source missing at {src_dir}', flush=True)
            continue

        # Step 3: emit the per-fold _cvXX YAML into <data_root>/config/fly/.
        out_ds = fold_dataset_name(cond, fold)
        if _opto_dataset_complete(out_ds):
            print(f'  [skip] {cond} cv{fold:02d}: already generated at '
                  f'{graphs_data_path("fly", out_ds)}', flush=True)
            continue
        fold_yaml = emit_fold_yaml(cond, fold)
        print(f'\n=== {cond}  cv{fold:02d}  ({out_ds}) ===', flush=True)
        print(f'config: {fold_yaml}', flush=True)
        print(f'source: {src_dir}', flush=True)

        # Step 4: re-simulate with the configured opto current.
        config = NeuralGraphConfig.from_yaml(fold_yaml)
        # Mirror GNN_Main.py:249-250 — prefix dataset with the YAML's parent
        # directory (here always 'fly/') so add_optogenetics_stimulus writes to
        # graphs_data/fly/<dataset>/ rather than graphs_data/<dataset>/.
        if not config.dataset.startswith('fly/'):
            config.dataset = 'fly/' + config.dataset
        add_optogenetics_stimulus(config)
