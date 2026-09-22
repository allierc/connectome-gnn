#!/usr/bin/env python
"""Submit the Vzfg Q2 SNR grid to LSF (gpu_a100) and print the launch table.

Grid: 4 measurement-noise levels x 2 arms (GNN, Known-ODE) = 8 jobs.
Process noise is fixed at sigma = 0.05 -- it is baked into the trajectory by the
generator, so it is not a post-hoc axis; changing it needs a new generation pass.

  python launch_snr_grid.py --table     # print the table, submit nothing
  python launch_snr_grid.py --dry-run   # print bsub commands, submit nothing
  python launch_snr_grid.py             # submit
"""
import argparse, os, subprocess, sys

ROOT = "/groups/saalfeld/home/allierc/GraphData"
CONFIG_DIR = os.path.join(ROOT, "config", "fly")
DATA_DIR = os.path.join(ROOT, "graphs_data", "fly")
CLUSTER_SSH = "allierc@login1"
CLUSTER_ROOT = "/groups/saalfeld/home/allierc/Graph/connectome-gnn-cx"
CONDA_ENV = "connectome-gnn"
QUEUE = "gpu_a100"
NCPU = 4
WALL_MIN = 6000
JOB_LOG_DIR = os.path.join(ROOT, "log", "neurips_review_jobs")
SIGMA_MODEL = float(os.environ.get('GRID_SIGMA', '0.05'))
PREFIX = os.environ.get('GRID_PREFIX', 'nr2_ca_snr')
TAGS = ["g000", "g010", "g030", "g100"]
ARMS = [("unified", "GNN"), ("known_ode", "Known-ODE")]


def meta(tag):
    p = os.path.join(DATA_DIR, f"{PREFIX}_{tag}", ".grid_meta")
    if not os.path.isfile(p):
        return None
    d = {}
    for ln in open(p):
        if '=' in ln:
            k, v = ln.strip().split('=', 1)
            d[k] = v
    return d


def rows():
    out = []
    for tag in TAGS:
        m = meta(tag)
        for arm, label in ARMS:
            out.append({
                "config": f"{PREFIX}_{tag}_{arm}",
                "arm": label,
                "sigma_model": SIGMA_MODEL,
                "gamma_frac": m and m.get("frac"),
                "snr_db": m and m.get("snr_db"),
                "lam": m and m.get("lambda"),
                "r_v": m and m.get("train_r_v"),
                "r_vdot": m and m.get("train_r_vdot"),
            })
    return out


def print_table(rs):
    hdr = f"{'config':30s} {'arm':10s} {'sigma_model':>11s} {'gamma/SD':>9s} {'SNR dB':>7s} {'lambda':>8s} {'r(v)':>7s} {'r(vdot)':>8s}"
    print(hdr); print('-' * len(hdr))
    for r in rs:
        def f(x, w, p=''):
            return (f"{float(x):{w}.{p}f}" if p else f"{str(x):>{w}s}") if x not in (None, '') else f"{'--':>{w}s}"
        snr = r["snr_db"]
        snr_s = "inf" if snr and snr.startswith('inf') else (f"{float(snr):.1f}" if snr else "--")
        print(f"{r['config']:30s} {r['arm']:10s} {r['sigma_model']:>11.2f} "
              f"{(r['gamma_frac'] or '--'):>9s} {snr_s:>7s} {(r['lam'] or '--'):>8s} "
              f"{(r['r_v'] or '--'):>7s} {(r['r_vdot'] or '--'):>8s}")


def bsub_cmd(config):
    cfg_abs = os.path.join(CONFIG_DIR, config + ".yaml")
    out = os.path.join(JOB_LOG_DIR, f"{config}.out")
    err = os.path.join(JOB_LOG_DIR, f"{config}.err")
    payload = (f"conda run -n {CONDA_ENV} python GNN_Main.py -o train_test_plot "
               f"{cfg_abs} --output_root {ROOT} --force")
    return (f"ssh {CLUSTER_SSH} \"cd {CLUSTER_ROOT} && source /etc/profile.d/profile.lsf.sh && "
            f"mkdir -p {JOB_LOG_DIR} && bsub -n {NCPU} -gpu \\\"num=1\\\" -q {QUEUE} "
            f"-W {WALL_MIN} -o {out} -e {err} \\\"{payload}\\\"\"")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    rs = rows()
    print_table(rs)
    missing = [r['config'] for r in rs if not os.path.isfile(os.path.join(CONFIG_DIR, r['config'] + '.yaml'))]
    if missing:
        print(f"\nMISSING CONFIGS ({len(missing)}): {missing}")
    if a.table:
        sys.exit(0)
    if missing:
        print("refusing to submit with missing configs"); sys.exit(1)
    print()
    for r in rs:
        cmd = bsub_cmd(r['config'])
        if a.dry_run:
            print(cmd); continue
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(f"{r['config']}: {p.stdout.strip() or p.stderr.strip()}")
