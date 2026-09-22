# The experimental plan, formalised

## What this replaces

`docs/experiment_manifest.tsv` is the informal version of this file: 114 rows of

    block <TAB> experiment label <TAB> arm label <TAB> run name <TAB> status

That format carries everything needed to *print* a table and nothing needed to
*check* one. It cannot say what question an experiment answers, which arm is the
baseline, what single thing an arm changes relative to that baseline, or that
two rows were trained at the same commit. Those are the four facts that decide
whether a comparison is fair, and all four currently live only in the person who
built the table.

The schema below adds exactly those four and derives the rest.

## Three layers

    experiments/plan.yaml          the plan: questions, the shared column set, the running order
    experiments/<id>.yaml          one experiment: a question, a baseline, its arms
    (derived)                      runs, statuses, cells -- never written down

Nothing in layer 3 is stored. A run's name is derived from its arm and fold, its
status from the files on disk, its cells from `results/metrics.txt`. Anything
stored twice can disagree, and the manifest's `status` column is the one field
that has already gone stale in practice.

## Layer 1 -- `plan.yaml`

    version: 1
    paper: <slug>
    log_root: <abs path to log/fly>
    config_roots: [<dir>, ...]        searched in order; a name in two is an error
    questions:
      <qid>:
        text: <one sentence, the question in words>
        read_from: [<column>, ...]    which columns bear on it
    columns:
      scalars:  [<metrics.txt key>, ...]     printed as a plain number
      recovery: [<stem>, ...]                printed as `clean [all] (pct dropped)`
      extra:    [<metrics.txt key>, ...]
    experiments: [<id>, ...]          ordered; this IS the section order of the PDF

`columns` is the one metric vocabulary, named once. A table that omits a column
its neighbour has cannot be read against it, so every experiment prints the same
set and a quantity a run does not carry prints `--`.

## Layer 2 -- `<id>.yaml`

    id: <id>                          must equal the filename stem
    title: <the table caption's subject>
    question: <one sentence>
    feeds: [<qid>, ...]               plan-level questions this experiment bears on

    task: train_cv | test_plot_cv | train | test_plot
    queue: gpu_l4 | gpu_a100 | gpu_h100
    wall: "HH:MM"
    commit: <sha> | null              null until the first arm is submitted

    folds: [cv00, cv01, ...]          the fold dimension, shared by every arm
    baseline: <arm id>                the arm every other arm is diffed against

    arms:
      - id: <arm id>
        label: <what the table row says>
        spec: <config stem, WITHOUT the fold suffix>
        overrides:                    dotted config keys, the ONLY declared difference
          <a.b.c>: <value>

    readouts:                         optional; one table block per readout
      - {id: <id>, label: <label>}

    preconditions:                    optional; facts the DATA must satisfy
      - {key: <dotted config key>, op: ">" | ">=" | "<" | "<=" | "==" | "!=",
         value: <literal>}

    report:
      caption: <one sentence, no prose beyond it>
      compare_to: <arm id>            usually the baseline

## Layer 3 -- the three bindings

Each binding is a total function with one rule. They are what make the plan
executable and printable rather than documentation.

### B1  arm x fold -> spec file

    spec_file(arm, fold) = first(root + "/" + arm.spec + "_" + fold + ".yaml"
                                 for root in config_roots if exists)

Ambiguity is an error, not a precedence rule: a stem present under both
`config/fly` and `GraphData/config/fly` means two different experiments are
about to print in one table.

### B2  arm -> GNN_Main invocation

    python GNN_Main.py -o <task> <arm.spec> --queue <queue> --wall <wall>

`-o train_cv` submits one bsub job per fold, each running
`python GNN_Main.py -o train <arm.spec>_cv<NN>`, and expects the per-fold yamls
to exist already -- it does not synthesise them. That is why `spec` is the stem
and `folds` is a separate dimension.

### B3  arm x fold -> table row

    run(arm, fold)      = arm.spec + "_" + fold
    log_dir(run)        = log_root + "/" + run
    cells(run)          = read(log_dir + "/results/metrics.txt", plan.columns)

Missing `metrics.txt` is not an error. It prints blank cells and a star, which
is the whole point of generating the PDF before the results land.

## Invariants

These are what `exp check` asserts. Each is a fact a person currently has to
remember.

| id | invariant | why |
|----|-----------|-----|
| I1 | every `spec_file(arm, fold)` resolves, and resolves in exactly one root | a silent second copy of a spec is a different experiment |
| I2 | `diff(cfg(arm, fold), cfg(baseline, fold))` **equals** `set(arm.overrides)` | the controlled-variable audit: "did we change only one thing?" |
| I3 | every arm uses the plan's `folds`, complete | a four-fold arm beside a five-fold arm is not a paired comparison |
| I4 | `arm x fold -> run` is injective | two arms resolving to one run silently print the same numbers twice |
| I5 | every landed run's `commit=` in `_completed_train` equals `commit`, and none is `-dirty` | the NeurIPS before/after differed by a COMMIT, not a flag; this is the check that would have caught it |
| I6 | every id in `feeds` exists in `plan.questions` | a question nothing feeds, or an experiment feeding nothing, is a gap in the plan |
| I7 | every column named in a report exists in `plan.columns` | one vocabulary |
| I8 | every `precondition` holds on the baseline's resolved config, per fold | an override can be a bit-exact NO-OP on the wrong dataset, and then the two arms are one arm |

I8 is the one that is not mechanical, and it earned its place immediately:
`training.derivative_target_clean` does nothing whatever on a dataset with
`measurement_noise_level = 0`, so the first draft of `derivative_bug` -- written
against `flyvis_noise_005_blank50`, which has none -- would have produced two
bit-identical five-folds and a table concluding that the bug does not matter.

I2 is equality in both directions. A key that changed but was not declared is an
uncontrolled variable; a key declared but unchanged is a claim the experiment
does not actually make.

## Status, derived

    landed   results/metrics.txt exists
    running  no metrics.txt, but tmp_training/Wij.log has a line
    pending  neither
    dirty    _completed_train records a -dirty commit

## Commands

    exp check  [<id>]     assert I1-I7; print the offending key for each failure
    exp status [<id>]     one row per arm x fold: run, status, iteration, commit
    exp report [<id>]     regenerate the PDF; blanks where results have not landed
    exp add    <id>       scaffold a new <id>.yaml from the template above

`exp report` is `docs/make_experiment_tables2.py` with its hardcoded `ARMS`
list replaced by the plan. Nothing else about that generator changes.
