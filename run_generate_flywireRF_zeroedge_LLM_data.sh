#!/bin/bash -l
#
# Pre-generate the 8 per-slot LLM datasets for the two flywireRF zero-edge
# explorations on the cluster (where DAVIS is mounted). Submits one bsub
# job per slot to gpu_a100, runs them in parallel.
#
# After all 8 finish, set claude.generate_data: false in both source yamls
# and re-run GNN_LLM.py from this devcontainer — it will skip local
# generation and submit cluster training directly.
#
# Slot yamls expected at:
#   /groups/saalfeld/home/allierc/GraphData/config/fly/
#       flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005_Claude_{00..03}.yaml
#       flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005_Claude_{00..03}.yaml
#
# Output datasets land at:
#   $OUTPUT_ROOT/graphs_data/fly/<base>_<NN>/
#
# Run via ssh from the devcontainer (no copy/scp needed — the script lives
# under the NFS-mounted Graph/ tree which is reachable from login1):
#   ssh allierc@login1 "bash -l -c 'bash /groups/saalfeld/home/allierc/Graph/connectome-gnn/run_generate_flywireRF_zeroedge_LLM_data.sh'"
#
# Each bsub job cd's into $CLUSTER_REPO (default GraphCluster/connectome-gnn,
# the same dir cluster.py uses) before running GNN_Main.py.

set -e

OUTPUT_ROOT="${OUTPUT_ROOT:-/groups/saalfeld/home/allierc/GraphData}"
CONFIG_DIR="$OUTPUT_ROOT/config/fly"
LOG_DIR="$OUTPUT_ROOT/log/_gen_flywireRF_zeroedge_LLM"
CONDA_ENV="${CONDA_ENV:-connectome-gnn}"
NODE="${NODE:-a100}"
WALL_MIN="${WALL_MIN:-360}"   # 6 hours; 1.96M edges x 64k frames is the bottleneck
# Cluster-side repo root that the bsub job will cd into before running
# GNN_Main.py. We default to the NFS-mounted Graph/connectome-gnn clone
# because GraphCluster/connectome-gnn is missing the pre-exported hybrid
# connectome tables under data/hybrid_connectomes/ that data_generate
# needs at simulation time. (Training jobs are unaffected — they only
# read the pre-built dataset, not these tables.)
CLUSTER_REPO="${CLUSTER_REPO:-/groups/saalfeld/home/allierc/Graph/connectome-gnn}"

mkdir -p "$LOG_DIR"

BASES=(
  flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005
  flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005
)

JOBS=()
for base in "${BASES[@]}"; do
  for slot in 00 01 02 03; do
    yaml="$CONFIG_DIR/${base}_Claude_${slot}.yaml"
    if [ ! -f "$yaml" ]; then
      echo "  [skip] missing $yaml"
      continue
    fi
    job_name="gen_${base}_${slot}"
    out_log="$LOG_DIR/${job_name}.out"
    err_log="$LOG_DIR/${job_name}.err"

    # Pre-erase stale completion marker so re-runs don't no-op.
    rm -f "$OUTPUT_ROOT/log/fly/${base}_${slot}/_completed_generate" 2>/dev/null || true

    cmd="cd '$CLUSTER_REPO' && python GNN_Main.py -o generate '$yaml' --output_root '$OUTPUT_ROOT'"
    bsub_cmd="bsub -n 2 -gpu 'num=1' -q gpu_${NODE} -W ${WALL_MIN} \
        -J '${job_name}' \
        -o '${out_log}' -e '${err_log}' \
        bash -l -c \"conda run -n ${CONDA_ENV} bash -c '${cmd}'\""

    echo "  [submit] ${job_name}"
    eval "$bsub_cmd"
  done
done

echo
echo "Submitted ${#BASES[@]} x 4 = $((${#BASES[@]} * 4)) data-generation jobs to gpu_${NODE}."
echo "Watch with: bjobs | grep gen_"
echo "Logs:       $LOG_DIR/"
echo
echo "When ALL finish:"
echo "  1. Edit both source yamls: claude.generate_data: false"
echo "  2. Re-run from devcontainer:"
echo "     python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005 iterations=80 --cluster --resume"
echo "     python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005 iterations=72 --cluster --resume"
