"""Re-run the plot pass of every GNN run behind Tables 1, 2 and Supp. Table 4 with the template
readout's activity filter OFF, in a sandbox.

    python tools/no_filter_reanalysis.py stage [group]            copy the runs into the sandbox
    python tools/no_filter_reanalysis.py submit [group] [--dry]   one bsub `-o plot` job per run
    python tools/no_filter_reanalysis.py compare                  Table 1: filtered against unfiltered
    python tools/no_filter_reanalysis.py promote [group]          back up each real run's results/ to
                                                                  pre_no_filter/ and put the sandbox's in

group: t1 (Table 1's 15 GNN runs), t2 (the 20 FlyWire-variant GNN runs of Table 2), s4 (Supp. Table 4's
other rows, 45 runs), all (default). The Known-ODE runs read W directly, so the filter does not touch them.

recovery.template_frame_filter: false makes every sampled frame enter every edge's fit
(metrics.extract_template_params). Only the plot pass re-runs, so the model, the sampled frames and
the rollout columns (results_rollout.log, kept by the copy) are those of the real run. SANDBOX/log/fly/<run>
is the run without its results/; its checkpoints are symlinks to the real ones (a plot pass writes
template_fit.pt into models/, the checkpoint files it only reads), SANDBOX/graphs_data a symlink to the
real data. The filtered numbers of `compare` are the real runs' results/metrics.txt before `promote`.
"""
import argparse
import filecmp
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import yaml

REAL = "/groups/saalfeld/home/allierc/GraphData"
SANDBOX = os.path.join(REAL, "sandbox_no_filter")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTER_REPO = REPO.replace("/workspace", "/groups/saalfeld/home/allierc/Graph", 1)
FOLDS = [f"cv{i:02d}" for i in range(5)]
NOISE = ("noise_free", "noise_005", "noise_05")
KEYS = ("Wij_R2", "tmpl_pct_fitted", "tau_R2", "tau_pct_outliers", "V_rest_R2", "V_rest_pct_outliers",
        "msg_i_R2", "clustering_accuracy")


VARIANTS = ("e8_flywireRF_noise_005", "e8_flywireRF_proximal_nulls_noise_005",
            "full_eye_flywireRF_noise_005", "full_eye_flywireRF_proximal_nulls_noise_005")
# Supp. Table 4's rows but the first (Table 1's sigma 0.05 row), without the two rows dropped from the paper
S4 = ("flyvis_noise_005_010_condl25rc20_{fold}", "flyvis_noise_005_020_condl25rc20_{fold}",
      "flyvis_noise_005_INR_davis_blank50_condl251s_{fold}", "flyvis_noise_005_null400_condl25_{fold}",
      "flyvis_noise_005_rm20_condl251s_{fold}", "flyvis_noise_005_rm50_condl251s_{fold}",
      "flyvis_noise_005_s5h21_condl25_{fold}", "flyvis_noise_005_hid10_condl251s_{fold}",
      "flyvis_noise_005_hid20_none_condl251s_{fold}")


def runs(group="all"):
    pats = {"t1": [f"flyvis_{n}_blank50_condl25_{{fold}}" for n in NOISE],
            "t2": [f"{v}_blank50_condl25_{{fold}}" for v in VARIANTS],
            "s4": list(S4)}
    out = []
    for g in (("t1", "t2", "s4") if group == "all" else (group,)):
        out += [p.format(fold=f) for p in pats[g] for f in FOLDS]
    return out


def stage(group="all"):
    os.makedirs(os.path.join(SANDBOX, "log", "fly"), exist_ok=True)
    os.makedirs(os.path.join(SANDBOX, "config", "fly"), exist_ok=True)
    gd = os.path.join(SANDBOX, "graphs_data")
    if not os.path.islink(gd):
        os.symlink(os.path.join(REAL, "graphs_data"), gd)
    for r in runs(group):
        src, dst = os.path.join(REAL, "log", "fly", r), os.path.join(SANDBOX, "log", "fly", r)
        if not os.path.isfile(os.path.join(src, "config.yaml")):
            print(f"MISSING {r}")
            continue
        if os.path.exists(os.path.join(SANDBOX, "config", "fly", r + ".yaml")):   # written last: the run is staged
            continue
        subprocess.run(["rsync", "-a", "--exclude", "/results", "--exclude", "/ablation50",
                        "--exclude", "/superseded", "--exclude", "/tmp_recons", "--exclude", "/models/*.pt",
                        src + "/", dst + "/"], check=True)
        for ck in glob.glob(os.path.join(src, "models", "*.pt")):
            os.symlink(ck, os.path.join(dst, "models", os.path.basename(ck)))
        os.makedirs(os.path.join(dst, "results"), exist_ok=True)
        if os.path.exists(os.path.join(src, "results", "test_metrics.npz")):
            shutil.copy2(os.path.join(src, "results", "test_metrics.npz"), os.path.join(dst, "results", "test_metrics.npz"))
        cfg = yaml.safe_load(open(os.path.join(src, "config.yaml")))
        cfg.setdefault("recovery", {})["template_frame_filter"] = False
        with open(os.path.join(SANDBOX, "config", "fly", r + ".yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"staged {r}")


def submit(dry, group="all"):
    book = os.path.join(SANDBOX, "submitted.txt")
    seen = {line.split()[-1] for line in open(book)} if os.path.exists(book) else set()
    for r in runs(group):
        d = os.path.join(SANDBOX, "log", "fly", r)
        if r in seen or os.path.exists(os.path.join(d, "results", "metrics.txt")):
            continue
        if not os.path.exists(os.path.join(SANDBOX, "config", "fly", r + ".yaml")):
            continue
        payload = (f"conda run -n connectome-gnn python GNN_Main.py -o plot "
                   f"{SANDBOX}/config/fly/{r}.yaml --output_root {SANDBOX}")
        ncpu = 12 if r.startswith(VARIANTS) or 'null400' in r else 8
        remote = (f"cd {shlex.quote(CLUSTER_REPO)} && bsub -n {ncpu} -gpu 'num=1' -q gpu_rtx6000 -W 480 "
                  f"-J nf_{r} -o {shlex.quote(d + '/plot.out')} -e {shlex.quote(d + '/plot.err')} "
                  f"{shlex.quote(payload)}")
        if dry:
            print(remote)
            continue
        res = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "allierc@login1", remote],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL)
        m = re.search(r"Job <(\d+)>", res.stdout)
        print(f"{m.group(1) if m else 'FAILED'}  gpu_rtx6000  {r}" + ("" if m else f"  {res.stderr.strip()[:160]}"))
        if m:
            with open(book, "a") as f:
                f.write(f"{m.group(1)} gpu_rtx6000 {r}\n")


def read(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return out


def compare():
    print("mean +- SD over cv00..cv04 (ddof 0); fitted % = share of edges the fit could fit, "
          "out. % = share of neurons outside the outlier band\n")
    print(f"{'noise':11s} {'metric':20s} {'filtered (paper)':>18s} {'no filter':>18s}")
    for n in NOISE:
        a = {k: [] for k in KEYS}; b = {k: [] for k in KEYS}
        for f in FOLDS:
            r = f"flyvis_{n}_blank50_condl25_{f}"
            ma = read(os.path.join(REAL, "log", "fly", r, "results", "metrics.txt"))
            mb = read(os.path.join(SANDBOX, "log", "fly", r, "results", "metrics.txt"))
            if "Wij_R2" not in mb:
                print(f"  {r}: sandbox plot pass not finished"); continue
            for k in KEYS:
                a[k].append(ma.get(k, np.nan)); b[k].append(mb.get(k, np.nan))
        for k in KEYS:
            if not b[k]:
                continue
            fa = f"{np.nanmean(a[k]):.3f} +- {np.nanstd(a[k]):.3f}"; fb = f"{np.nanmean(b[k]):.3f} +- {np.nanstd(b[k]):.3f}"
            print(f"{n:11s} {k:20s} {fa:>18s} {fb:>18s}")
        print()


def promote(group="all"):
    """Put the sandbox's results/ (and the template_fit*.pt it wrote into models/) into the real run.
    A real file that differs is first copied to <run>/pre_no_filter/<same relative path>."""
    n_new = n_back = 0
    for r in runs(group):
        sb, real = os.path.join(SANDBOX, "log", "fly", r), os.path.join(REAL, "log", "fly", r)
        if not os.path.exists(os.path.join(sb, "results", "metrics.txt")):
            print(f"NOT SCORED {r}")
            continue
        for sub in ("results", "models"):
            for root, _dirs, files in os.walk(os.path.join(sb, sub)):
                for f in files:
                    a = os.path.join(root, f)
                    if os.path.islink(a):
                        continue                                    # a checkpoint: the real file
                    rel = os.path.relpath(a, sb)
                    b = os.path.join(real, rel)
                    if os.path.exists(b) and filecmp.cmp(a, b, shallow=False):
                        continue
                    if os.path.exists(b):
                        bk = os.path.join(real, "pre_no_filter", rel)
                        os.makedirs(os.path.dirname(bk), exist_ok=True)
                        if not os.path.exists(bk):
                            shutil.copy2(b, bk)
                            n_back += 1
                    os.makedirs(os.path.dirname(b), exist_ok=True)
                    shutil.copy2(a, b)
                    n_new += 1
        print(f"promoted {r}")
    print(f"{n_new} files written, {n_back} real files backed up to <run>/pre_no_filter/")


TABLES = ("cv_table_gnn_vs_baselines", "flybrid_inliers", "cv_table_gnn_cross_noise")
SCRIPTS = ("table_1_gnn_vs_baselines", "table_2_flybrid", "table_s4_cross_noise")


def tables():
    """Regenerate Tables 1, 2 and S4 from the sandbox's runs (every other run read from the real log
    folder through symlinks), into a scratch folder, and print each line that differs from the
    tracked table."""
    root = os.path.join(REAL, "sandbox_no_filter_tables")
    lf = os.path.join(root, "log", "fly")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(lf)
    sb = set(os.listdir(os.path.join(SANDBOX, "log", "fly")))
    for r in os.listdir(os.path.join(REAL, "log", "fly")):
        os.symlink(os.path.join(SANDBOX if r in set(runs("all")) else REAL, "log", "fly", r), os.path.join(lf, r))
    scratch = "/tmp/no_filter_tables"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(os.path.join(scratch, "tables")); os.makedirs(os.path.join(scratch, "figures"))
    shutil.copytree(os.path.join(REPO, "experiments", "paper", "scripts"), os.path.join(scratch, "scripts"))
    env = dict(os.environ, GNN_OUTPUT_ROOT=root, PYTHONPATH=os.path.join(REPO, "src"))
    for sc in SCRIPTS:
        res = subprocess.run([sys.executable, f"{sc}.py"], cwd=os.path.join(scratch, "scripts"), env=env,
                             capture_output=True, text=True)
        if res.returncode:
            print(f"{sc} FAILED:\n{res.stderr[-1500:]}")
    for t in TABLES:
        new, ref = os.path.join(scratch, "tables", t + ".tex"), os.path.join(REPO, "experiments", "paper", "tables", t + ".tex")
        if not os.path.exists(new):
            print(f"{t}: not produced"); continue
        a, b = open(ref).read().splitlines(), open(new).read().splitlines()
        print(f"{t}: " + ("byte-identical" if open(ref, "rb").read() == open(new, "rb").read() else "DIFFERS"))
        for k, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  line {k + 1}\n    was {x}\n    now {y}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=("stage", "submit", "compare", "promote", "tables"))
    ap.add_argument("group", nargs="?", default="all", choices=("t1", "t2", "s4", "all"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.verb == "submit":
        submit(a.dry, a.group)
    elif a.verb == "compare":
        compare()
    elif a.verb == "tables":
        tables()
    else:
        {"stage": stage, "promote": promote}[a.verb](a.group)
