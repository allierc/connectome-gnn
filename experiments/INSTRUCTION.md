# How this folder works

Read this first in any new session that touches `experiments/`.

An experiment is **one markdown file**. It states the question, names the specs
that answer it, and carries the results the cluster sends back. A tool,
`tools/exp.py`, reads that file to launch the jobs, to refresh the numbers, and
to build `report.pdf`. Nothing else knows a spec name — not the launcher's
output, not the PDF, not this document.

**The experiment NUMBER is the only handle.** `exp launch 2`, `exp poll 2`,
`exp report` — you never have to remember `flyvis_noise_005_blank50_condl100_cv03`.

```
experiments/
  INDEX.md              number, name, purpose, landed/total   (written by the tool)
  INSTRUCTION.md        this file                             (written by hand)
  expNN_<name>.md       one per experiment                    (both)
  baseline/             two frozen configs, never edited
  specs/<exp>/fly/*.yaml  the pool of specs
  report.pdf            one slide per experiment              (written by the tool)
```

Branch: `feat/experiment-plan` in `allierc/connectome-gnn`.

---

## The four verbs

```bash
PYTHONPATH=src python tools/exp.py launch  <n> [--arm A] [--where AXIS=V1,V2] [--dry-run]
PYTHONPATH=src python tools/exp.py analyse <n>...      # the SECOND job per run
PYTHONPATH=src python tools/exp.py poll    [n...]      # rewrite the STATUS blocks
PYTHONPATH=src python tools/exp.py report  [n...]      # experiments/report.pdf
```

`launch` stages every spec to `GraphData/config/fly/`, submits one `bsub` job
per run, and records the job ids back into the markdown's front matter.

`analyse` is not optional and not automatic. `-o train` does **not** write
`results/metrics.txt`; only `-o test_plot` does, and that is a second job. A run
whose training has finished but whose analysis has not shows as `trained`, and
its row in the landed table is empty until `analyse` runs. `analyse` skips runs
that already have metrics, so re-running it as more of the grid finishes is safe
and is the normal thing to do.

`poll` reads the log tree and rewrites the `<!-- STATUS -->` block inside each
markdown, plus `INDEX.md`. Run it before reading any table.

`report` builds the deck from the markdown files.

Always `--dry-run` a launch first. Thirty jobs is thirty jobs.

---

## The experiment markdown

YAML front matter the tool reads, then prose for humans, then a STATUS block the
tool owns.

```yaml
number: 2                                   # the handle
name: conductance_lasso                     # slide title, _FIGURES key
title: ...                                  # one line
purpose: ...                                # the QUESTION, printed under the table
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/exp02/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
axes:                                       # the grid; `fold` is what a mean is over
  noise: [noise_free, noise_005, noise_05]
  fold:  [cv00, cv01, cv02, cv03, cv04]
arms:
- id: current
  label: current form, no lasso (experiment 1)
  spec_pattern: flyvis_{noise}_blank50_dtfd_{fold}
  submit: false                             # READ, never run — see below
  differs_by: {}
- id: conductance
  label: conductance form, group lasso 100
  spec_pattern: flyvis_{noise}_blank50_condl100_{fold}
  differs_by:                               # what this arm changes, for the reader
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 100.0
job_ids: {...}                              # written by launch
analyse_job_ids: {...}                      # written by analyse
report:
  arm_order: [current, cond_l25, conductance]
  arm_labels: {cond_l25: conductance}       # display name; the id stays the handle
  arm_columns: {lasso: training.coeff_g_phi_input_group_L1}
  timing: true                              # exp00 only: the wall-clock table
  timing_iters: 1600000
```

Run names are **derived** from `spec_pattern` and the grid point, never stored,
so the spec table in the prose and the thing on disk cannot drift apart.

`axes_only` on an arm restricts it to part of the grid — for a probe that exists
at one noise level only. Widening it later is a front-matter edit plus
`launch --where`.

`submit: false` means the arm's runs are **read from the log tree and never
launched**. Experiment 2's `current` arm is experiment 1's `nominal` runs. Without
this flag `launch` would resubmit them and — because it clears a run directory
before reusing it — delete a running experiment's logs.

---

## Adding an experiment

1. Ask the user for the question. **Write down what they say, do not invent a
   purpose.** The `purpose` line is printed under the table and is the thing the
   experiment is judged against.
2. Generate the specs into `experiments/specs/expNN/fly/` from one of the two
   frozen baselines, changing only what the arms declare in `differs_by`. Verify
   it: parse both YAMLs and assert the diff is exactly the intended keys. Do not
   eyeball a `diff`.
3. Write `expNN_<name>.md` with the front matter above and a spec table listing
   every run.
4. `launch --dry-run`, read the list, then `launch`.
5. `poll`, then `analyse` when runs reach `trained`, then `poll` again.

No comments in spec YAML — bare YAML only. Reasoning goes in source comments,
commit messages, or the experiment markdown.

---

## The environment

**Never run a process on a login node.** `python`, `conda`, `mamba`, `cp` and
`rsync` are watched there; admins kill them and email Cedric. Everything on the
cluster is a `bsub` job, including environment updates. `/groups` is mounted in
the devcontainer, so staging a spec is a local copy and nothing of ours ever
runs on `$CLUSTER_SSH`.

| | |
|---|---|
| python (devcontainer) | `/workspace/.conda_envs/neural-graph-linux/bin/python` — plain `python` is /opt/conda base with no torch |
| python (cluster job) | relative: `python GNN_Main.py -o train <abs spec path>` |
| ssh | `ssh $CLUSTER_SSH` — bare `$CLUSTER_SSH` resolves to `node@` and is denied |
| log root | `/groups/saalfeld/home/allierc/GraphData/log/fly` |
| stage dir | `/groups/saalfeld/home/allierc/GraphData/config/fly` |
| queues | `gpu_rtx6000`, `gpu_a100`, `gpu_l4` |
| git push | `--no-verify`, always — git-lfs is missing in the devcontainer |

`launch` submits the **staged absolute path**, not the bare stem:
`add_pre_folder` classifies a bare name by looking for a domain keyword in it,
and an unrecognised one kills every job in 30 seconds.

On hardware, from experiment 0: **RTX 6000 bf16 is the fastest arm** at 48.6
it/s, 9.1 h for a 1.6 M-iteration run, against the A100 fp32's 31.6 it/s and
14.1 h. bf16 costs nothing on recovery on either card, and `gpu_a100` is the
deeper queue.

---

## The one rule

**Nothing in the report is typed. Every number and every label is derived from
the run that produced it.**

This is not style. It has been broken three times and each break produced a
wrong conclusion that survived until someone happened to look:

- The two `fit roll r` column headers were typed as "cond." and "curr.". They
  are relative to the MODEL, so every current-model table read backwards, and
  the deck appeared to show a conductance readout beating a current one on
  current data. The run says which is which in `template_rollout_model`.
  `alt_form_family` looks like the right field and is not — it is the family of
  the *generator*. `tests/test_exp_table_headers.py` now fails if a family name
  is typed into `COLUMNS` or `_PRETTY`.
- Experiment 0's speed table was typed. LSF's own `Run time` says the ranking
  was inverted and the rates were a factor of two to three out, and the "A100,
  fp32" recommendation it carried is withdrawn. The slide now reads
  `cluster.out`.
- A mean was taken across folds read at whatever iteration each job happened to
  have reached, and reported `R2_W 0.450 ± 0.472` for a group whose five folds
  were 0.913–0.932. `common_iter` now reads every fold at the same snapshot.

If a label can differ between runs, derive it and put a test on the derivation.

---

## Reading a table without being fooled

- **A constant across a sweep is a rail, not a result.** Experiment 2's
  conductance `fit roll r` was 0.112–0.120 on ten independent runs at two noise
  levels. That is the ±100 V divergence clamp in
  [`graph_tester.py:725`](../src/connectome_gnn/models/graph_tester.py#L725)
  holding 67–71% of the 13,741 neurons, not a weak model. The clamp share is now
  measured and printed in the fit-roll columns' parentheses — **a `fit roll r`
  beside a large percentage is a diverged rollout, not a score.**
- The parentheses mean two different things by column: percentage of outliers
  dropped on the `R²` columns, percentage of neuron-frames clamped on the two
  fit-roll columns.
- `rollout_r` is scored against the **noise-free twin** when one exists, not
  against the noisy trajectory; against the noisy one it saturates at ~0.82 and
  ranks nothing.
- `R²_Vrest` beside `R²_Vrest no C_i` is the offset question: the gap is how
  much of `V_rest` the per-edge offset `C_ij` was corrupting.
- Green is above 0.9. It is a reading aid, not a threshold anyone agreed on.
- The running block is the **train** split at a stated iteration; the landed
  block is **held-out**. They do not share a column's meaning.

---

## Traps

- `launch` **clears the run directory** before submitting. A leftover directory
  with the same name would otherwise have its numbers polled and printed as this
  experiment's — that has happened.
- `launch --arm X` submits a subset; the job-id record is **merged**, not
  replaced, or the other arm's ids are lost.
- To widen an arm rather than relaunch it, use `--where`:
  `launch 2 --arm cond_l25 --where noise=noise_005,noise_05` left the five
  already-landed `noise_free` folds alone.
- `training.time_step` is **gone**. A config carrying anything but `1` is
  rejected with a message naming its replacements: `rollout_horizon_schedule`
  for the depth, `rollout_loss_stride` for which steps are scored. It was
  triple-duty (depth, dataset decimation, target offset), so a result could not
  be attributed to any of the three.
- `training.derivative_target` has exactly two values. `observed_fd` is the
  nominal one, the finite difference of the observed voltage, `f(v) + ξ/Δt`.
  `y_list` is the generator's own stored derivative, `f(v)` — an **oracle**, and
  the bug behind Table 1. Never add a third.
- Determinism holds within one GPU model and fixed kernels. Compiled and eager
  differ; a bit-exactness check must fix both.
- A GPU is not free of the queue. `gpu_a100` had 2,506 jobs pending against
  `gpu_rtx6000`'s 344 when experiment 0 ran, and two arms' analysis had to be
  moved off it to land at all.

---

## Where things stand

Run `poll` and read `INDEX.md` — it is regenerated and this paragraph is not.
As of 2026-09-23: **114 jobs across six experiments, 54 landed, 60 running.**
Experiments 0 and 1 are complete; 2 is waiting on ten lasso-25 runs; 3, 4 and 5
are training.
