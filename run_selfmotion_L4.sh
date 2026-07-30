#!/usr/bin/env bash
# Launch the three self-motion task-target trainings on L4 GPU nodes (Janelia LSF).
#
# bsub is NOT available in the devcontainer, so run this from a cluster submit
# host, from the repo root, with GNN_OUTPUT_ROOT set (shared /groups path).
#
#   cd <repo root on the cluster>
#   export GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData
#   bash run_selfmotion_L4.sh
#
# Submits one job per task-target config (rotation / translation / both), all
# training on the shared 4-ch dataset zebrafish_hd_si_task_selfmotion. Override
# QUEUE / WALL / NCPU via env vars.
set -euo pipefail

QUEUE="${QUEUE:-gpu_l4}"          # L4 GPU queue
WALL="${WALL:-1440}"             # wall-clock minutes
NCPU="${NCPU:-8}"
: "${GNN_OUTPUT_ROOT:?set GNN_OUTPUT_ROOT (e.g. /groups/saalfeld/home/allierc/GraphData)}"

CONFIGS=(
  zebrafish_hd_si_ipn12_selfmotion_rotation
  zebrafish_hd_si_ipn12_selfmotion_translation
  zebrafish_hd_si_ipn12_selfmotion_both
)

for cfg in "${CONFIGS[@]}"; do
  logdir="${GNN_OUTPUT_ROOT}/log/zebrafish/${cfg}"
  mkdir -p "${logdir}"
  # NOTE: relative "python GNN_Main.py ..." per the cluster convention.
  bsub -n "${NCPU}" -gpu "num=1" -q "${QUEUE}" -W "${WALL}" \
       -J "sm_${cfg##*_}" \
       -o "${logdir}/bsub_%J.out" -e "${logdir}/bsub_%J.err" \
       "python GNN_Main.py -o train ${cfg}"
  echo "submitted: ${cfg}  ->  queue=${QUEUE}  log=${logdir}"
done

echo
echo "watch:    bjobs -w | grep sm_"
echo "metrics:  tail -f ${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_ipn12_selfmotion_*/tmp_training/metrics.log"
