"""Re-run the plot pass of every run behind Tables 1, 2 and Supp. Table 4 with this branch's code,
in a sandbox, and regenerate the three tables from it.

    python tools/template_only_verify.py stage            copy the runs into the sandbox
    python tools/template_only_verify.py submit [--dry]   one bsub `-o plot` job per run
    python tools/template_only_verify.py status           which sandbox runs have a metrics.txt
    python tools/template_only_verify.py tables           regenerate the tables, diff them

Written for the removal of the gain-correction chain (branch refactor/template-readout-only,
2026-10-13): the question is whether the tables come out byte-identical when every number in
them is produced by the template readout alone. The sandbox is SANDBOX/log/fly/<run>, a copy of
the run without its results/ (the plot pass clears and rewrites results/, so the real run's must
not be touched), the 2.5 GB ablation50/ folder or superseded/; SANDBOX/graphs_data is a symlink
to the real data. The plot pass reads the rollout numbers from results_rollout.log, which the
copy keeps, so the rollout columns test the copy and the parameter columns test the readout.

`tables` copies experiments/paper/scripts to a scratch folder so the table scripts write into
scratch tables/ (they write next to themselves), runs them with GNN_OUTPUT_ROOT=SANDBOX, and
compares each .tex byte for byte with experiments/paper/tables/, printing the differing lines.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys

REAL = "/groups/saalfeld/home/allierc/GraphData"
SANDBOX = os.path.join(REAL, "sandbox_template_only")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the same tree as seen from a cluster node: /workspace is the NFS export of .../Graph
CLUSTER_REPO = REPO.replace("/workspace", "/groups/saalfeld/home/allierc/Graph", 1)
FOLDS = [f"cv{i:02d}" for i in range(5)]
NOISE = ("noise_free", "noise_005", "noise_05")
VARIANTS = ("e8_flywireRF_noise_005", "e8_flywireRF_proximal_nulls_noise_005",
            "full_eye_flywireRF_noise_005", "full_eye_flywireRF_proximal_nulls_noise_005")
# Supp. Table 4's rows, in table_s4_cross_noise.py's order
S4 = ("flyvis_noise_005_blank50_condl25_{fold}", "flyvis_noise_005_010_condl25rc20_{fold}",
      "flyvis_noise_005_020_condl25rc20_{fold}", "flyvis_noise_005_INR_davis_blank50_condl251s_{fold}",
      "flyvis_noise_005_null400_condl25_{fold}", "flyvis_noise_005_null400_condl100_{fold}",
      "flyvis_noise_005_rm20_condl251s_{fold}", "flyvis_noise_005_rm50_condl251s_{fold}",
      "flyvis_noise_005_s5h06_condl25_{fold}", "flyvis_noise_005_s5h21_condl25_{fold}",
      "flyvis_noise_005_hid10_condl251s_{fold}", "flyvis_noise_005_hid20_none_condl251s_{fold}")
TABLES = ("cv_table_gnn_vs_baselines", "flybrid_inliers", "cv_table_gnn_cross_noise")
SCRIPTS = ("table_1_gnn_vs_baselines", "table_2_flybrid", "table_s4_cross_noise")


def runs():
    pats = ([f"flyvis_{n}_blank50_condl25_{{fold}}" for n in NOISE]
            + [f"flyvis_{n}_blank50_kode217_{{fold}}" for n in NOISE]
            + [f"{v}_blank50_kode_{{fold}}" for v in VARIANTS]
            + [f"{v}_blank50_condl25_{{fold}}" for v in VARIANTS] + list(S4))
    out = []
    for p in pats:
        for f in FOLDS:
            r = p.format(fold=f)
            if r not in out:
                out.append(r)
    return out


def queue(run):
    """(queue suffix, n_cpus): the experiments' own -- 12 slots (240 GB) for the FlyWire
    variants and the five-fold null-edge graphs, the H100 for the SIREN-stimulus runs."""
    if "_INR_" in run:
        return "h100", 8
    if run.startswith(VARIANTS) or "null400" in run:
        return "rtx6000", 12
    return "rtx6000", 8


def stage():
    os.makedirs(os.path.join(SANDBOX, "log", "fly"), exist_ok=True)
    os.makedirs(os.path.join(SANDBOX, "config", "fly"), exist_ok=True)
    gd = os.path.join(SANDBOX, "graphs_data")
    if not os.path.islink(gd):
        os.symlink(os.path.join(REAL, "graphs_data"), gd)
    for r in runs():
        src, dst = os.path.join(REAL, "log", "fly", r), os.path.join(SANDBOX, "log", "fly", r)
        if not os.path.isfile(os.path.join(src, "config.yaml")):
            print(f"MISSING {r}")
            continue
        if os.path.exists(os.path.join(dst, "config.yaml")):
            continue
        subprocess.run(["rsync", "-a", "--exclude", "/results", "--exclude", "/ablation50",
                        "--exclude", "/superseded", "--exclude", "/tmp_recons",
                        src + "/", dst + "/"], check=True)
        # the three files clear_results keeps, so the pass sees the results/ it would see
        os.makedirs(os.path.join(dst, "results"), exist_ok=True)
        for k in ("test_metrics.npz",):
            if os.path.exists(os.path.join(src, "results", k)):
                shutil.copy2(os.path.join(src, "results", k), os.path.join(dst, "results", k))
        shutil.copy2(os.path.join(src, "config.yaml"), os.path.join(SANDBOX, "config", "fly", r + ".yaml"))
        print(f"staged {r}")


def submit(dry):
    # one job per run, ever: the ids go to SANDBOX/submitted.txt and a run listed
    # there is not submitted again (delete its line to resubmit it)
    book = os.path.join(SANDBOX, "submitted.txt")
    seen = {line.split()[-1] for line in open(book)} if os.path.exists(book) else set()
    for r in runs():
        d = os.path.join(SANDBOX, "log", "fly", r)
        if r in seen or os.path.exists(os.path.join(d, "results", "metrics.txt")):
            continue
        if not os.path.exists(os.path.join(SANDBOX, "config", "fly", r + ".yaml")):
            continue                                  # not staged yet
        q, ncpu = queue(r)
        payload = (f"conda run -n connectome-gnn python GNN_Main.py -o plot "
                   f"{SANDBOX}/config/fly/{r}.yaml --output_root {SANDBOX}")
        remote = (f"cd {shlex.quote(CLUSTER_REPO)} && bsub -n {ncpu} -gpu 'num=1' -q gpu_{q} -W 480 "
                  f"-J tv_{r} -o {shlex.quote(d + '/plot.out')} -e {shlex.quote(d + '/plot.err')} "
                  f"{shlex.quote(payload)}")
        if dry:
            print(remote)
            continue
        res = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                              "allierc@login1", remote], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL)
        m = re.search(r"Job <(\d+)>", res.stdout)
        print(f"{m.group(1) if m else 'FAILED'}  gpu_{q}  {r}" + ("" if m else f"  {res.stderr.strip()[:160]}"))
        if m:
            with open(book, "a") as f:
                f.write(f"{m.group(1)} gpu_{q} {r}\n")


def status():
    done, todo = [], []
    for r in runs():
        p = os.path.join(SANDBOX, "log", "fly", r, "results", "metrics.txt")
        ok = os.path.exists(p) and any(line.startswith("Wij_R2:") for line in open(p))
        (done if ok else todo).append(r)
    print(f"{len(done)}/{len(done) + len(todo)} sandbox runs have a scored metrics.txt")
    for r in todo:
        print(f"  pending {r}")


def tables():
    scratch = "/tmp/template_only_tables"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(os.path.join(scratch, "tables"))
    os.makedirs(os.path.join(scratch, "figures"))
    shutil.copytree(os.path.join(REPO, "experiments", "paper", "scripts"), os.path.join(scratch, "scripts"))
    env = dict(os.environ, GNN_OUTPUT_ROOT=SANDBOX, PYTHONPATH=os.path.join(REPO, "src"))
    for s in SCRIPTS:
        res = subprocess.run([sys.executable, f"{s}.py"], cwd=os.path.join(scratch, "scripts"),
                             env=env, capture_output=True, text=True)
        if res.returncode:
            print(f"{s} FAILED:\n{res.stderr[-1500:]}")
    for t in TABLES:
        new, ref = os.path.join(scratch, "tables", t + ".tex"), os.path.join(REPO, "experiments", "paper", "tables", t + ".tex")
        if not os.path.exists(new):
            print(f"{t}: not produced")
            continue
        a, b = open(ref).read().splitlines(), open(new).read().splitlines()
        if a == b and open(ref, "rb").read() == open(new, "rb").read():
            print(f"{t}: byte-identical")
            continue
        print(f"{t}: DIFFERS")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  line {i + 1}\n    was {x}\n    now {y}")
        if len(a) != len(b):
            print(f"  line count {len(a)} -> {len(b)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=("stage", "submit", "status", "tables"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"stage": stage, "status": status, "tables": tables}.get(a.verb, lambda: submit(a.dry))()
