#!/bin/bash
# Full CV pipeline for the joint GNN+INR (SIREN) model.
#
# Trains N seeds x 2 conditions of GNN+INR and emits the two TeX rows of
# tab:cv_inr (DAVIS and YouTube-VOS).  Replaces run_INR_cv_batch.sh and
# run_INR_DAVIS_cv_batch.sh.
#
# Directory layout produced per seed i in 0..N-1:
#   log/fly/<CFG_NAME>_davis_cv<i>/   DAVIS stimuli   -> Row 1
#   log/fly/<CFG_NAME>_yt_cv<i>/      YouTube-VOS     -> Row 2
#
# Caching — nothing is re-done when its output already exists (per fold):
#   data_generate skipped if <graphs_data>/x_list_train exists
#   data_train    skipped if models/best_model_with_*.pt   exists
#   data_test     skipped if results_rollout.log           exists
#   data_plot     skipped if results/metrics.txt           exists
# So the script is idempotent: interrupt and re-run freely.
#
# Output:
#   TeX rows for tab:cv_inr are written to
#     <DATA_ROOT>/log/fly/<CFG_NAME>/results/cv_inr_table_rows.tex
#   (also echoed to stdout at the end of the run).
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is bash run_GNN_INR.sh
#   # (piping via `< run_GNN_INR.sh` also works — stdin fallback handled below)
#
# Or run interactively:
#   bash run_GNN_INR.sh

set -euo pipefail

# BASH_SOURCE[0] is empty when the script is read from stdin
# (e.g. `bsub -Is < run_GNN_INR.sh`), so dirname returns "." and cd
# lands us in ~/.lsbatch. Detect that case and fall back to the
# cluster_root_dir in data_paths.json.
if [[ -n "${BASH_SOURCE[0]:-}" && -f "$(dirname "${BASH_SOURCE[0]}")/run_GNN_INR.sh" ]]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    REPO_DIR="$(python3 -c 'import json,os; print(json.load(open(os.path.expanduser("~/Graph/connectome-gnn/data_paths.json")))["cluster_root_dir"])' 2>/dev/null)"
    REPO_DIR="${REPO_DIR:-/groups/saalfeld/home/allierc/Graph/connectome-gnn}"
fi
cd "${REPO_DIR}"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
CFG_NAME="flyvis_noise_005_INR"
N_SEEDS=5

echo "============================================================"
echo "GNN+INR full CV — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Config:    ${CFG_NAME}   Seeds: ${N_SEEDS}"
echo "Folds:     ${CFG_NAME}_{davis,yt}_cv00..$((N_SEEDS-1))"
echo "============================================================"

python "${REPO_DIR}/scripts/run_inr_cv.py" \
    --config "${CFG_NAME}" \
    --output_root "${DATA_ROOT}" \
    --n_seeds "${N_SEEDS}" \
    --conditions davis yt

echo ""
echo "============================================================"
echo "Emitting TeX rows for tab:cv_inr — $(date)"
echo "============================================================"
python "${REPO_DIR}/scripts/emit_inr_table_rows.py" \
    --config "${CFG_NAME}" \
    --output_root "${DATA_ROOT}" \
    --n_seeds "${N_SEEDS}"

echo ""
echo "============================================================"
echo "GNN+INR full CV complete — $(date)"
echo "============================================================"
