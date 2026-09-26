"""Harness self-test: plant known behaviour changes and check the harness sees them.

Each mutant is a textual edit of a scratch copy of the HEAD implementation
(``src/``, ``GNN_PlotFigure.py``, ``assets/``, APFS-cloned from this checkout).
The fast tier then runs with the mutant as "head" against the base
implementation; a mutant is CAUGHT when at least one cell reports a
difference, and the report names the comparator categories that fired
(A files, B plots, C global state, D exception, E logs).

Some planted edits do not change behaviour in the current code (an
"equivalent mutant"); for those the expected outcome is NOT caught, and the
reason is stated with the mutant. ``also`` lists full-tier cells that are run
as well, to check that an equivalence seen on DAVIS holds on Sintel too.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import cells as C
from . import driver as D
from . import fixtures as FX
from . import manifest as M

V = "src/connectome_gnn/generators/voltage/"
INTEGRATE = V + "integrate.py"
INITIAL = V + "initial.py"
PIPELINE = V + "pipeline.py"
SPEC = V + "spec.py"
DAVIS = "src/connectome_gnn/generators/davis.py"


@dataclass(frozen=True)
class Mutant:
    name: str
    edits: tuple                      # ((relative file, old, new), ...); old must occur exactly once
    expect_fast: str                  # "caught" | "equivalent"
    why: str
    also: tuple = field(default=())   # full-tier cells run as well; same expectation as the fast tier


# Phase 2 moved data_generate_voltage into generators/voltage/, so every mutant
# is planted where the same behaviour lives now (phase 1 planted them in
# graph_data_generator.py at BASE_SHA; same eleven behaviours, same expectations).
MUTANTS = [
    Mutant("eta_xi_draw_order", (
        (INTEGRATE, "                    y = pde(x, edge_index, has_field=False)\n"
                    "                    dv_step = y.squeeze()\n",
                    "                    y = pde(x, edge_index, has_field=False)\n"
                    "                    dv_step = y.squeeze()\n"
                    "                    _xi_first = (torch.randn(n_neurons, dtype=torch.float32, device=device)\n"
                    "                                 if noise_model_level > 0 else None)\n"),
        (INTEGRATE, "                        x.voltage = x.voltage + torch.randn(\n"
                    "                            n_neurons, dtype=torch.float32, device=device\n"
                    "                        ) * noise_model_level\n",
                    "                        x.voltage = x.voltage + _xi_first * noise_model_level\n"),
    ), "caught", "process noise drawn before measurement noise; F3 has both"),
    Mutant("store_finite_difference", (
        (INTEGRATE, "                    writer.write_state(x, record)\n",
                    "                    writer.write_state(x, record)\n"
                    "                    _v_prev = x.voltage.clone()\n"),
        (INTEGRATE, "                    writer.write_drift(record)\n",
                    "                    writer.write_drift(record._replace(\n"
                    "                        drift=((x.voltage - _v_prev) / spec.delta_t).unsqueeze(-1)))\n"),
    ), "caught", "y_list gets (v[t+1]-v[t])/dt instead of f(v[t]); they differ even at sigma=0 (exp. Euler)"),
    Mutant("clone_W_before_save", (
        (PIPELINE, "        self.ode_params.save(folder)\n",
                   "        self.ode_params.W = self.ode_params.W.clone()\n        self.ode_params.save(folder)\n"),
    ), "equivalent", "every saved W owns exactly its storage (measured), so a clone writes the same bytes"),
    Mutant("W_view_of_larger_storage", (
        (PIPELINE, "        self.ode_params.save(folder)\n",
                   "        self.ode_params.W = torch.cat([self.ode_params.W, self.ode_params.W])"
                   "[: self.ode_params.W.numel()]\n"
                   "        self.ode_params.save(folder)\n"),
    ), "caught", "finding 8: equal values, but torch.save writes the whole storage of a view"),
    Mutant("extra_dataset_access", (
        (INITIAL, '    sequences = stimuli.item(0)["lum"]\n    frame = sequences[0][None, None]\n',
                  '    _ = stimuli.item(0)\n    sequences = stimuli.item(0)["lum"]\n'
                  '    frame = sequences[0][None, None]\n'),
    ), "equivalent", "no __getitem__ draws RNG here: DAVIS is built with augment=False, and Sintel's "
                     "jitter/noise/gamma draws are guarded by `if self.<x>_std`, all None in this config "
                     "(n_rot=0, flip_axis=0 are passed explicitly); the Sintel cells confirm it",
        also=("sintel_only_noise", "sintel_mixed")),
    Mutant("init_calcium_after_shuffle", (
        (INITIAL, "    # init neuron state x\n\n    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)\n",
                  "    # init neuron state x\n\n    _init_calcium = None\n"),
        (PIPELINE, "        return dict(split=stimulus_mod.split_videos(self.stimuli))\n",
                   "        _split = stimulus_mod.split_videos(self.stimuli)\n"
                   "        self.x.calcium = torch.rand(self.n_neurons, dtype=torch.float32, device=self.spec.device)\n"
                   "        return dict(split=_split)\n"),
    ), "equivalent", "no torch draw lies between the old and the new position (the shuffle is numpy), "
                     "so the torch stream is unchanged; calcium is not saved (save_calcium False)"),
    Mutant("init_calcium_after_materialize", (
        (INITIAL, "    # init neuron state x\n\n    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)\n",
                  "    # init neuron state x\n\n    _init_calcium = None\n"),
        (PIPELINE, "        return dict(sequences=stimulus_mod.materialize_sequences(self.spec, self.stimuli, self.split))\n",
                   "        _seqs = stimulus_mod.materialize_sequences(self.spec, self.stimuli, self.split)\n"
                   "        self.x.calcium = torch.rand(self.n_neurons, dtype=torch.float32, device=self.spec.device)\n"
                   "        return dict(sequences=_seqs)\n"),
    ), "caught", "the DATA would not show it (materialisation draws no RNG on either path, see "
                 "extra_dataset_access, so the draw still precedes every other torch draw; equivalent in "
                 "phase 1), but the draw now sits in materialize_sequences, declared draws=False, so the "
                 "RNG ledger raises RngLedgerError in every check-mode cell (all but F1_base): D and C",
        also=("sintel_only_noise",)),
    Mutant("break_stimulus_alias", (
        (INITIAL, "        stimulus=net.stimulus().squeeze().to(device),\n",
                  "        stimulus=net.stimulus().squeeze().to(device).clone(),\n"),
    ), "caught", "finding 3: F9's stored stimulus after it=0 is the rendered frame only through the alias"),
    Mutant("drop_davis_random_seed", (
        (DAVIS, "            random.seed(self.shuffle_seed)\n", "            pass\n"),
    ), "caught", "the sequence shuffle then follows the runner's pre-seeded stdlib stream"),
    Mutant("max_test_frames_50", (
        (SPEC, "MAX_TEST_FRAMES = 8000\n", "MAX_TEST_FRAMES = 50\n"),
    ), "caught", "fast-tier test splits have 64 frames"),
    Mutant("max_test_frames_7000", (
        (SPEC, "MAX_TEST_FRAMES = 8000\n", "MAX_TEST_FRAMES = 7000\n"),
    ), "caught", "no data changes (no test split reaches 7000 frames; the largest is 1024), but the cap is "
                 "echoed in the 'generating TEST data (capped at N frames ...)' log line: E only"),
]
MUTANTS_BY_NAME = {m.name: m for m in MUTANTS}


def make_scratch(src_root: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("src", "GNN_PlotFigure.py", "assets"):
        s = Path(src_root) / name
        if sys.platform == "darwin":
            subprocess.run(["cp", "-cR", str(s), str(dst / name)], check=True)
        elif s.is_dir():
            shutil.copytree(s, dst / name)
        else:
            shutil.copy2(s, dst / name)
    for p in (dst / "src").rglob("__pycache__"):
        shutil.rmtree(p)
    return dst


# Per-file scopes an edit is confined to (none since phase 2: every planted file
# is a voltage-generation module; phase 1 needed one inside graph_data_generator.py
# because data_generate_spiking repeats many of the voltage path's lines).
SCOPES: dict = {}


def _scope(rel: str, text: str) -> tuple[int, int]:
    if rel not in SCOPES:
        return 0, len(text)
    a, b = SCOPES[rel]
    return text.index(a), text.index(b)


def check(m: Mutant, root: Path) -> None:
    for rel, old, _ in m.edits:
        text = (root / rel).read_text()
        lo, hi = _scope(rel, text)
        n = text[lo:hi].count(old)
        if n != 1:
            raise AssertionError(f"mutant {m.name}: snippet occurs {n} times in the scope of {rel}:\n{old}")


def apply(m: Mutant, root: Path) -> None:
    check(m, root)
    for rel, old, new in m.edits:
        p = root / rel
        text = p.read_text()
        lo, hi = _scope(rel, text)
        text = text[:lo] + text[lo:hi].replace(old, new) + text[hi:]
        p.unlink()                           # replace the cloned file rather than edit it in place
        p.write_text(text)


def _cats(diffs: list[str]) -> list[str]:
    return sorted({d.strip()[1] for d in diffs if d.strip().startswith("[")})


def main(args) -> int:
    work = FX.work_root()
    base = D.ensure_base_worktree(work)
    names = args.mutants or [m.name for m in MUTANTS]
    fast = [c for c in C.select("fast") if C.requirements_met(c)[0]]
    extra = sorted({n for m in names for n in MUTANTS_BY_NAME[m].also
                    if C.requirements_met(C.CELLS_BY_NAME[n])[0]})
    cells_all = fast + [C.CELLS_BY_NAME[n] for n in extra]
    for n in names:                       # fail before running anything
        check(MUTANTS_BY_NAME[n], D.REPO_ROOT)
    D.prepare(cells_all, work, base)
    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    root = work / "mutants" / label
    base_runs = {}
    specs = [D.RunSpec(impl_root=base, cell=c.name, out=root / "_base" / c.name) for c in cells_all]
    for r in D.run_many(specs, work, jobs=args.jobs, progress=None):
        base_runs[r.spec.cell] = r
    report = []
    for name in names:
        m = MUTANTS_BY_NAME[name]
        impl = make_scratch(D.REPO_ROOT, root / name / "impl")
        apply(m, impl)
        cells_m = fast + [C.CELLS_BY_NAME[n] for n in m.also if n in extra]
        specs = [D.RunSpec(impl_root=impl, cell=c.name, out=root / name / c.name) for c in cells_m]
        t0 = time.time()
        results = D.run_many(specs, work, jobs=args.jobs, progress=None)
        per_cell = {}
        for r in results:
            b = base_runs[r.spec.cell]
            if r.manifest is None:
                per_cell[r.spec.cell] = {"runner_error": r.error[-400:]}
                continue
            diffs = M.compare(b.manifest, r.manifest, run_a=b.spec.out, run_b=r.spec.out, explain=False)
            if diffs:
                per_cell[r.spec.cell] = {"categories": _cats(diffs), "first": diffs[:3]}
        caught_fast = sorted(c for c in per_cell if c in {x.name for x in fast})
        caught_also = sorted(c for c in per_cell if c in m.also)
        entry = {
            "mutant": name, "expect_fast": m.expect_fast, "why": m.why,
            "caught_fast": bool(caught_fast), "fast_cells": caught_fast,
            "also": list(m.also), "caught_also": caught_also,
            "categories": sorted({k for c in per_cell.values() for k in c.get("categories", [])}),
            "details": per_cell, "wall_s": round(time.time() - t0, 1),
        }
        want = m.expect_fast == "caught"
        ok = entry["caught_fast"] == want and (bool(caught_also) == want or not m.also)
        entry["as_expected"] = ok
        report.append(entry)
        print(f"{'OK ' if ok else 'BAD'} {name:32s} fast={'caught' if caught_fast else 'not caught':10s} "
              f"(expected {m.expect_fast}) cats={entry['categories']} cells={caught_fast} also={caught_also}")
        shutil.rmtree(impl, ignore_errors=True)
    (root / "mutants.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {root / 'mutants.json'}")
    return 0 if all(e["as_expected"] for e in report) else 1
