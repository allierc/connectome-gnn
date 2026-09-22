# The experimental plan

One yaml per experiment, plus a plan-level yaml, plus two commands over them.
The yamls are the record of what was intended; the commands say which runs have
landed and arrange them into a PDF. Nothing validates anything.

## Files

    experiments/plan.yaml          questions, the shared column set, the running order
    experiments/<id>.yaml          one experiment: a question, a grid, its arms
    experiments/specs/fly/         the run specs, one per arm x grid point
    experiments/report.pdf         generated

`experiments/specs/fly` is named `fly` and not something prettier because
`load_run_config` derives the domain from the parent directory name and
`validate_pre_folder` rejects anything outside `{fly, drosophila_cx, larva,
zebrafish, zebrafish_oculomotor}`. Each spec keeps `config_file: fly/<name>`, so
its log still lands in `log/fly/<name>`.

## plan.yaml

    version: 1
    paper: <slug>
    log_root: <abs path to log/fly>
    config_roots: [<dir>, ...]
    questions:
      <qid>:
        text: <one sentence>
        read_from: [<column>, ...]
    columns:
      scalars:  [<metrics.txt key>, ...]     printed as a plain number
      recovery: [<stem>, ...]                printed as `clean [all] (pct dropped)`
      extra:    [<metrics.txt key>, ...]
    experiments: [<id>, ...]          ordered; this IS the section order of the PDF

`columns` is the one metric vocabulary, named once, so a cell means the same
thing in every table.

## <id>.yaml

    id: <id>                          must equal the filename stem
    title: <the section heading>
    question: <one sentence>

    task: train | train_cv | test_plot
    queue: gpu_l4 | gpu_a100 | gpu_h100
    wall: "HH:MM"

    axes:                             the grid; `fold` is averaged over
      <axis>: [<value>, ...]

    arms:
      - id: <arm id>
        label: <what the summary row says>
        spec: <format string over the axis names>
        overrides:                    what this arm changes, for the record
          <a.b.c>: <value>

    report:
      caption: <one sentence>

An experiment is a grid, not a list. `derivative_target` is three model-noise
levels x two arms x five folds, so `spec` is a format string
(`flyvis_{noise}_blank50_dtfd_{fold}`) and one arm covers the whole sweep.

`overrides` is documentation: it says what the arm changed, and nothing reads it
back.

## The two bindings

    run(arm, point)  = arm.spec.format(**point)
    cells(run)       = read(log_root/run/results/metrics.txt, plan.columns)

A run with no `metrics.txt` prints blank cells and a star. That is the point of
generating the PDF before the results land.

## Status, derived

    landed   results/metrics.txt exists
    running  no metrics.txt, but tmp_training/Wij.log has a line
    pending  neither

## Commands

    exp status [<id>]     one row per run: status, iteration, commit
    exp report [<id>]     regenerate the PDF; blanks where results have not landed

Submission is not here. Specs go to the cluster by whatever launcher you use;
for this experiment that is all thirty at once on `gpu_l4`.
