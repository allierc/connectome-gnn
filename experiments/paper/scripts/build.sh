#!/bin/bash
# Rebuild experiments/paper/main.pdf from the campaign's results.
#   bash scripts/build.sh            (from experiments/paper; needs GNN_OUTPUT_ROOT)
# 1. tables from metrics.txt        2. figures from the runs' npz / checkpoints
# 3. main.tex from overleaf/main.tex (blue captions, red passages)   4. pdflatex + bibtex
set -e
cd "$(dirname "$0")/.."
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
export PATH=/workspace/.conda_envs/neural-graph-linux/bin:$PATH
export GNN_OUTPUT_ROOT=${GNN_OUTPUT_ROOT:-/groups/saalfeld/home/allierc/GraphData}
mkdir -p figures tables
cp -n overleaf/figure/* figures/ 2>/dev/null || true       # published figures kept as they are
cp -n overleaf/tables/*.tex tables/ 2>/dev/null || true    # published tables not regenerated here
$PY scripts/make_tables.py
for f in fig_gnn_params_3col fig_flywire_hybrid fig_rollout_3col "fig_rollout_3col --ablation50" "fig_rollout_3col --known-ode" fig_known_ode_params_3col fig_gnn_params_4col_flywire fig_clustering_appendix fig_stim_rollout_inr; do
  set -- $f; $PY scripts/$1.py "${@:2}" 2>&1 | grep -v "findfont\|UserWarning\|warnings.warn"
done
$PY scripts/edit_tex.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
grep -n "^!" main.log || echo "main.pdf: $(grep -o 'Output written on main.pdf ([0-9]* pages' main.log)"
