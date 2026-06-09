"""Export a faithful learned-model swim rollout to JSON for the web demo.

Loads the trained zebrafish_hd_si_gnn_ipn12 checkpoint, runs the model on
auto-generated swim-integration stimuli (a few seeds), and dumps:
  - real skeleton geometry (model-order, PCA-aligned so the dIPN ring is in-plane)
  - per-neuron z-scored activity time series (uint8) for each seed
  - traces: HD, omega, L/R, F/B, and swim events
Output: interactive/zebrafish_rollout.json
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import torch
from connectome_gnn.utils import set_data_root, load_data_root_from_json, log_path

# data root: GraphData on /groups
for cand in [os.environ.get("GNN_OUTPUT_ROOT"),
             "/groups/saalfeld/home/allierc/GraphData"]:
    if cand and os.path.isdir(cand):
        set_data_root(cand); print("data_root =", cand); break

import fig_zebrafish_anatomy_3d_voltage_anim as A

CONFIG = "zebrafish_hd_si_gnn_ipn12_v1_cv0"   # trained CV fold of the v1 model
N_STEPS = 3000
SEEDS = [1, 7, 21]
TIME_STRIDE = 4          # export every 4th frame
DOWNSAMPLE = 12          # navis skeleton downsample

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev)
model, config = A._load(CONFIG, dev)
dt = float(getattr(model, "dt", 0.01))
print("dt", dt, "n_steps", N_STEPS)

# ---- circuit registry: body_ids + categories in model order ----
from connectome_gnn.generators.circuits import get_circuit
from connectome_gnn.generators.connectome_loaders import _zhd_category
circuit = get_circuit(config.circuit.name)
model_bodyids = circuit.body_ids
model_categories = np.array(
    [_zhd_category(circuit.type_names[t]) for t in circuit.neuron_types],
    dtype=object)
prov = circuit.provenance
anatomy_dirs = [prov.get("anatomy_dir")]
anatomy_dirs.extend(prov.get("anatomy_extra_dirs", []) or [])
anatomy_dirs = [d for d in anatomy_dirs if d]
print("anatomy_dirs", anatomy_dirs, "N", circuit.N)

neurons, types_str, has_skel = A._load_skeletons_in_model_order(
    anatomy_dirs, model_bodyids, model_categories, downsample=DOWNSAMPLE)
seg_arrays, seg_owner, all_segs = A._extract_per_neuron_segments(neurons, has_skel)
idx_with = np.where(has_skel)[0]
print(f"{len(idx_with)} neurons with skeletons; {all_segs.shape[0]:,} segments")

# ---- PCA-align all skeleton points so the ring lies in the xy-plane ----
pts = all_segs.reshape(-1, 3)
ctr = pts.mean(0)
cov = np.cov((pts - ctr).T)
w, V = np.linalg.eigh(cov)
R = V[:, np.argsort(w)[::-1]]                 # columns: u,v,w (var-descending)
scale = np.percentile(np.abs((pts - ctr) @ R), 99)
def tf(P):  # -> normalized aligned coords
    return ((np.asarray(P) - ctr) @ R) / scale

# per-neuron preferred angle from soma/mean position in ring plane
def neuron_segments(i):
    s = seg_arrays[i]
    if len(s) == 0: return None, None
    a = tf(s.reshape(-1, 3)).reshape(-1, 2, 3)
    mean_xy = a.reshape(-1, 3).mean(0)
    ang = float(math.atan2(mean_xy[1], mean_xy[0]))
    return a, ang

geom = []           # one entry per displayed neuron
ang_list = []
for i in idx_with:
    segs, ang = neuron_segments(i)
    flat = []
    for s in segs:
        flat += [round(float(s[0][0]), 3), round(float(s[0][1]), 3), round(float(s[0][2]), 3),
                 round(float(s[1][0]), 3), round(float(s[1][1]), 3), round(float(s[1][2]), 3)]
    geom.append({"t": str(types_str[i]), "a": round(ang, 4), "s": flat})
    ang_list.append(ang)

# ---- run swims, z-score, export activity + traces ----
z_lo = float(getattr(config.plotting, "anatomy_voltage_z_lo", 0.0))
z_hi = float(getattr(config.plotting, "anatomy_voltage_z_hi", 3.0))
# z_hi=10 in config makes everything dim; use a perceptual default if huge
if z_hi > 5: z_hi = 3.0
print("z_lo,z_hi", z_lo, z_hi)

runs = []
for sd in SEEDS:
    h_traj, theta, omega, decoded_hd, turn_lr, swim_fb, theta_disp = A._run_swim(
        model, N_STEPS, dt, dev, seed=sd)
    mu = h_traj.mean(0, keepdims=True); sg = h_traj.std(0, keepdims=True) + 1e-6
    z = (h_traj - mu) / sg
    lit = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-6), 0, 1)   # [T, Nmodel]
    lit = lit[:, idx_with]                                     # displayed neurons
    lit = lit[::TIME_STRIDE]                                   # time stride
    act_u8 = (lit * 255).astype(np.uint8)
    sub = slice(None, None, TIME_STRIDE)
    runs.append({
        "seed": sd,
        "T": int(act_u8.shape[0]),
        "act": [row.tolist() for row in act_u8],               # [T][Nneu] 0..255
        "hd":  [round(float(x), 4) for x in decoded_hd[sub]],
        "omega": [round(float(x), 3) for x in omega[sub]],
        "lr": [round(float(x), 3) for x in turn_lr[sub]],
        "fb": [round(float(x), 3) for x in swim_fb[sub]],
    })
    print(f"seed {sd}: T={act_u8.shape[0]} lit_med={np.median(lit):.3f} "
          f"HD span {decoded_hd.min():.2f}..{decoded_hd.max():.2f}")

out = {
    "config": CONFIG, "dt": dt * TIME_STRIDE, "n_neurons": len(geom),
    "z_lo": z_lo, "z_hi": z_hi,
    "geom": geom, "runs": runs,
}
outpath = os.path.join(ROOT, "interactive", "zebrafish_rollout.json")
with open(outpath, "w") as f:
    json.dump(out, f)
print("wrote", outpath, round(os.path.getsize(outpath) / 1e6, 2), "MB")
