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
# WHY A SWEEP, AND WHY THIS DECADE. The coefficient is sized from a measurement,
# not an estimate: on the trained sigma=0.05 conductance model the derivatives
# are df/dv = -26.16 per second and df/dmsg = +1.098, so G = 0.0403 where the ODE
# requires 1 and the relative residual the term charges is 0.78 per neuron,
# steady across finite-difference steps from 1e-6 to 1. The term is a norm(2)
# over the batch exactly like the trajectory loss, so 0.78 over ~13,700 neurons
# is a norm near 91: 1e-4 contributes ~9e-3 against a trajectory loss of
# 0.04-0.13, with 1e-5 and 1e-3 a decade either side. The first bracket tried
# here, 2e-7 to 2e-5, came from an estimate and contributed 3.7e-08 -- four
# orders of magnitude of nothing, which is what this measurement was for.
#
# NOTE ON THE CURRENT ARMS BELOW: the current family has almost no violation to
# punish -- its gauge error sits in tau rather than in G -- so the gain arm there
# tests only that the term is harmless, not that it works.
#
# WHAT TO READ AFTERWARDS, in this order:
#   tools/pysr_recovery.py <config>   Wij_gain near 1.0 and tmpl_k_median near
#                                     tau*1/tau = 1 is the term working
#   tools/message_gauge.py <config>   a_median toward 1.0, b_median toward 0
#   results/metrics.txt               rollout_r must NOT fall: the term is only
#                                     worth having if the fit survives it
#
# EVERY ARM HAS coeff_W_L1 0 AND coeff_W_L2 0. Nothing shrinks W, so whatever
# discipline the weights keep comes from the silent anchor and the gain term
# rather than from shrinkage, and the two control arms say what that base does on
# its own. The five current-family arms are on gpu_l4 and are not in this file.
#
# COST: 12 arms x 24 h on a100. The rc5 pair costs about 3x the network
# evaluations of the others, since the rollout horizon multiplies the cost per
# update and the loop count was kept rather than cut.

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
  # train_test_plot, NOT train. The task string is matched by substring
  # (GNN_Main.py: "train" in task, "test" in task, "plot" in task), so one token
  # runs all three in sequence -- the convention GNN_LLM+ already uses. Training
  # alone leaves no metrics.txt and no figures, and a night that has to be
  # replotted by hand the next morning is a night half spent.
  local cmd="conda run -n ${ENV} python GNN_Main.py -o train_test_plot ${cfg}"
  mkdir -p "${OUT}" 2>/dev/null || true   # the devcontainer cannot, the cluster can
  if [ "${DRY}" = "1" ]; then
    echo "bsub -n ${NCPU} -gpu \"num=1\" -q ${QUEUE} -W ${WALL} -J ${cfg} -o ${OUT}/${cfg}.out \"${cmd}\""
  else
    bsub -n "${NCPU}" -gpu "num=1" -q "${QUEUE}" -W "${WALL}" -J "${cfg}" \
         -o "${OUT}/${cfg}.out" "${cmd}"
  fi
}

# ---------------------------------------------------------------- sigma = 0.05
# The control: no weight penalty, no gauge pinning. Read it FIRST -- it is free
# to inflate or collapse, and it is what the gain arms are measured against.
submit flyvis_conductance_noise_005_conductance_nol12sil_ctrl_cv00

# The gain sweep. G is 0.0403 on this model where the ODE requires 1.
submit flyvis_conductance_noise_005_conductance_nol12sil_g1e5_cv00
submit flyvis_conductance_noise_005_conductance_nol12sil_g1e4_cv00
submit flyvis_conductance_noise_005_conductance_nol12sil_g1e3_cv00

# Rollout curriculum, alone and with the gain term. It cannot break the gauge --
# msg/c with a gain of c reproduces the whole trajectory -- but it can tighten
# tau, which is what limits the BLIND conductance readout.
submit flyvis_conductance_noise_005_conductance_nol12sil_rc5_cv00
submit flyvis_conductance_noise_005_conductance_nol12sil_rc5g1e4_cv00

# ------------------------------------------------------------------ sigma = 0
# THE CASE THE GAIN TERM EXISTS FOR. Without it this model put all 434,112
# weights below 1e-6 (largest 3.29e-07) and let f_theta amplify by 255. With no
# weight penalty at all the control may do worse still, which is the measurement:
# check max W**2 before any recovery metric.
submit flyvis_conductance_noise_free_conductance_nol12sil_ctrl_cv00

submit flyvis_conductance_noise_free_conductance_nol12sil_g1e5_cv00
submit flyvis_conductance_noise_free_conductance_nol12sil_g1e4_cv00
submit flyvis_conductance_noise_free_conductance_nol12sil_g1e3_cv00

submit flyvis_conductance_noise_free_conductance_nol12sil_rc5_cv00
submit flyvis_conductance_noise_free_conductance_nol12sil_rc5g1e4_cv00
