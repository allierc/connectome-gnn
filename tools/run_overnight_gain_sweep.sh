#!/usr/bin/env bash
# Overnight: does coeff_f_theta_msg_gain help, and at what coefficient?
#
# NOT LAUNCHED BY ANYTHING. Run it by hand from the cluster checkout
# (/groups/saalfeld/home/allierc/Graph/connectome-gnn) once the arms below are
# agreed. Every block can be commented out on its own.
#
#   ssh allierc@login1
#   cd /groups/saalfeld/home/allierc/Graph/connectome-gnn && git pull
#   bash tools/run_overnight_gain_sweep.sh            # submits
#   bash tools/run_overnight_gain_sweep.sh --dry-run  # prints, submits nothing
#
# WHAT IS BEING TESTED. The generator puts the message and the neuron's own
# voltage inside one bracket over one tau, (V_rest - v_i + msg_i + I_i) / tau_i,
# so df/dmsg = -df/dv for every neuron whatever its tau. Nothing in the loss said
# so until now, and the message's scale was therefore free: msg/c with a gain of
# c through f_theta is the same trajectory. Measured consequences on the runs in
# hand -- tau*df/dmsg = 0.0404 where the ODE requires 1 on the sigma=0.05
# conductance GNN (a message 24.8x too large), and on sigma=0 the entire weight
# vector down to 1e-8 with f_theta amplifying it back, leaving nothing
# recoverable. coeff_f_theta_msg_gain penalises
#
#     || (df/dmsg + df/dv) / rms_batch(df/dv) ||_2
#
# WHY A SWEEP. The coefficient is unswept. The term is a norm(2) over the batch,
# exactly like the trajectory loss, so the two are directly comparable: at a
# 24.8x gauge error the residual is ~24 per neuron, and over a batch of ~55,000
# that is a norm near 5.6e3 against a trajectory loss of 0.04-0.13. A coefficient
# of 2e-6 therefore contributes ~1e-2, about a quarter of the data loss, when the
# gauge is as wrong as it has been measured to be, and falls away as the gauge
# closes. 2e-7 and 2e-5 bracket that by a decade either side. If every arm at
# 2e-5 trains but none of them moves tau*df/dmsg, the term is too weak and the
# sweep should move up, not down.
#
# WHAT TO READ AFTERWARDS, in this order:
#   tools/pysr_recovery.py <config>   Wij_gain near 1.0 and tmpl_k_median near
#                                     tau*1/tau = 1 is the term working
#   tools/message_gauge.py <config>   a_median toward 1.0, b_median toward 0
#   results/metrics.txt               rollout_r must NOT fall: the term is only
#                                     worth having if the fit survives it
#
# COST: 16 arms x 24 h on a100 (4 current + 12 conductance). Trim before running if that is too much -- the
# two nol1sil blocks are the ones to drop first, since gnnsil is the current
# winner and nol1sil only asks whether W_L1 still matters once the gain is pinned.

set -u

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

QUEUE=gpu_a100
WALL=24:00
NCPU=2
ENV=connectome-gnn
OUT=/groups/saalfeld/home/allierc/Graph/.scratch/night

# conda run, ALWAYS. An interactive bsub -Is inherits the login shell's
# environment and bare `python` is the right one there; a batch job does not, and
# the four current arms submitted without it on 2026-09-14 exited in 0.2 s of CPU
# with no log directory and nothing to read, because bsub without -o keeps the
# output. Hence also -o: a job that dies must leave its reason on disk.
submit() {
  local cfg="$1"
  local cmd="conda run -n ${ENV} python GNN_Main.py -o train ${cfg}"
  mkdir -p "${OUT}" 2>/dev/null || true   # the devcontainer cannot, the cluster can
  if [ "${DRY}" = "1" ]; then
    echo "bsub -n ${NCPU} -gpu \"num=1\" -q ${QUEUE} -W ${WALL} -J ${cfg} -o ${OUT}/${cfg}.out \"${cmd}\""
  else
    bsub -n "${NCPU}" -gpu "num=1" -q "${QUEUE}" -W "${WALL}" -J "${cfg}" \
         -o "${OUT}/${cfg}.out" "${cmd}"
  fi
}

# ------------------------------------------------- the reference current run
# NOT PART OF THE SWEEP: four arms on flyvis_current_noise_005, the recipe that
# already works, asking only whether each change leaves it working. nol drops
# f_theta's weight penalties, sil adds the silent anchor, nol1sil does both minus
# coeff_W_L1, and nol1silgain adds the new term on top at the middle coefficient.
# Submitted by hand on 2026-09-14 as jobs 154285912-15; kept here so the night's
# run is one file rather than a file plus a memory.
submit flyvis_current_noise_005_current_nol
submit flyvis_current_noise_005_current_sil
submit flyvis_current_noise_005_current_nol1sil
submit flyvis_current_noise_005_current_nol1silgain

# ---------------------------------------------------------------- sigma = 0.05
# The reference conductance winner. If the term helps anywhere it should help
# here, where the gauge error is measured and the model is otherwise healthy.
submit flyvis_conductance_noise_005_conductance_gnnsil_g07_cv00
submit flyvis_conductance_noise_005_conductance_gnnsil_g06_cv00
submit flyvis_conductance_noise_005_conductance_gnnsil_g05_cv00

# Same, with coeff_W_L1 off: does the elementwise weight penalty still earn its
# place once the message scale is pinned by something that is not a shrinkage?
submit flyvis_conductance_noise_005_conductance_nol1sil_g07_cv00
submit flyvis_conductance_noise_005_conductance_nol1sil_g06_cv00
submit flyvis_conductance_noise_005_conductance_nol1sil_g05_cv00

# ------------------------------------------------------------------ sigma = 0
# THE CASE THE TERM EXISTS FOR. Without it the sigma=0 gnnsil run put all 434,112
# weights below 1e-6 (largest 3.29e-07) and let f_theta amplify by 255; with the
# gain pinned that route is closed, so "did the weight vector survive" is the
# first thing to check, before any recovery metric.
submit flyvis_conductance_noise_free_conductance_gnnsil_g07_cv00
submit flyvis_conductance_noise_free_conductance_gnnsil_g06_cv00
submit flyvis_conductance_noise_free_conductance_gnnsil_g05_cv00

submit flyvis_conductance_noise_free_conductance_nol1sil_g07_cv00
submit flyvis_conductance_noise_free_conductance_nol1sil_g06_cv00
submit flyvis_conductance_noise_free_conductance_nol1sil_g05_cv00
