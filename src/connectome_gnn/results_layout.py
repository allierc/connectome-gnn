"""Which figures a run's results/ directory shows, and which it tucks away.

A results/ directory holding forty PNGs is one nobody reads. Everything this
codebase draws is still drawn and still saved -- the ones a reader does not open
first go to results/extras/ instead of being dropped, so no analysis is lost and
nothing has to be re-run to get them back.

WHAT STAYS IN results/, and why each earns it:

    neuron<N>_panels.png      one neuron opened up, term by term: the readout
                              that shows whether the synapses came back
    rollout_*                 the free-running trajectory against the generator
    Wij_comparison.png        the four scatters against truth. All four come
    Eij_comparison.png        from the same RecoveredParams that metrics.txt is
    tau_comparison.png        written from, and each states its own filtering:
    V_rest_comparison.png     R2 without outliers, the full-sample R2 beside it
    Wij_scatter_*             in parentheses, and the share that was dropped
    embedding_augmented.png   the UMAP the cell-type clustering is read off;
                              the plain a_i scatter goes to extras

Five writers share this: GNN_PlotFigure, plot, plot_twin, graph_tester and
tools/wij_scatter. They had five different ideas of where a figure belongs
before, which is how results/ came to hold both a headline panel and a debug
histogram with nothing distinguishing them.
"""

import os

# The four scatters against truth, by EXACT name. Exact rather than prefix
# because each has demoted siblings -- tau_comparison_cell_type,
# tau_comparison_fslope, weights_comparison_corrected -- that a prefix would
# sweep back in, and because only these four are drawn from the same
# RecoveredParams that results/metrics.txt is written from. The others are
# earlier estimators, kept for comparison, which is what extras/ is for.
KEEP_EXACT = (
    "Wij_comparison.png",
    "Eij_comparison.png",
    "tau_comparison.png",
    "V_rest_comparison.png",
    "parameter_error.png",
)

# The embedding kept is the AUGMENTED one -- the UMAP of (a_i, tau, V_rest, and
# the weight statistics) that the cell-type clustering is actually read off. The
# plain a_i scatter is a projection of two of those columns and goes to extras.
KEEP_EXACT_EMBEDDING = ("embedding_augmented.png",)

# First match wins; checked after EXTRAS_FIRST.
KEEP_PREFIXES = (
    "neuron",                  # narrowed by EXTRAS_FIRST to the per-neuron panels
    "rollout",
    "Wij_scatter",             # tools/wij_scatter, the log-axis population view
    "embedding_augmented",
    "hidden_inr_traces",
    "metrics",
)

# Checked BEFORE KEEP_PREFIXES, for the one name a keep-prefix would otherwise
# capture: the cell-type RMSE bar chart begins with "neuron" and is not a panel.
EXTRAS_FIRST = ("neuron_type_reconstruction",)

# Not figures, and not swept: written by one pass and read by a later one.
# metrics.txt is deliberately NOT here -- data_plot rewrites it from scratch and
# appends as it goes, so leaving the previous run's copy would silently produce a
# file holding two runs' numbers with nothing marking the boundary.
KEEP_ALWAYS = ("corrected_W.pt", "learned_ode_params.pt", "test_metrics.npz")


def is_headline(name) -> bool:
    """Whether this filename belongs in results/ rather than results/extras/."""
    base = os.path.basename(str(name))
    if base in KEEP_ALWAYS or base in KEEP_EXACT or base in KEEP_EXACT_EMBEDDING:
        return True
    if base.startswith(EXTRAS_FIRST):
        return False
    return base.startswith(KEEP_PREFIXES)


def fig_out(log_dir, name) -> str:
    """Absolute path for a figure, in results/ or results/extras/, made ready.

    `log_dir` may be a run's log directory OR its results/ directory already --
    two writers hold the latter -- so both are accepted rather than making each
    call site remember which it has.
    """
    log_dir = str(log_dir).rstrip("/")
    if os.path.basename(log_dir) == "results":
        log_dir = os.path.dirname(log_dir)
    out = os.path.join(log_dir, "results")
    if not is_headline(name):
        out = os.path.join(out, "extras")
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, os.path.basename(str(name)))


def clear_results(log_dir, prefixes=None, keep_extras=False, spare=()) -> int:
    """Remove a run's stale result files before the task that regenerates them.

    A figure left from an earlier commit looks exactly like one drawn today, and
    results/ is read as a snapshot of one run at one revision -- after the
    parameter readout changed underneath it, stale figures were the difference
    between two panels in one directory describing different estimators.

    `prefixes` limits the sweep to basenames starting with one of them, which is
    how `-o test` clears only the rollout outputs it owns; None clears every
    file except KEEP_ALWAYS. `spare` is the mirror image and is what `-o plot`
    uses: clear everything EXCEPT these prefixes, because the rollout figures
    belong to the test pass and a plot-only run is meant to redraw against the
    rollout already on disk. `keep_extras` leaves results/extras/ in place.
    Returns how many entries were removed.
    """
    import shutil
    results = os.path.join(str(log_dir), "results")
    if not os.path.isdir(results):
        return 0
    n = 0
    for name in sorted(os.listdir(results)):
        path = os.path.join(results, name)
        if os.path.isdir(path):
            if name == "extras" and not keep_extras and prefixes is None:
                shutil.rmtree(path, ignore_errors=True)
                n += 1
            continue
        if prefixes is not None and not name.startswith(tuple(prefixes)):
            continue
        if spare and name.startswith(tuple(spare)):
            continue
        if prefixes is None and name in KEEP_ALWAYS:
            continue
        try:
            os.remove(path)
            n += 1
        except OSError:
            pass
    return n
