"""Connectome-GNN — Code-change LLM exploration loop.

Sibling of GNN_LLM.py. Adds a block-scoped Research/Staging/Code-change/Train/
Verdict pipeline that lets the agent write new training mechanisms (not just
YAML mutations) under tight discipline:

  - Phase R (≤10 min): read memory + literature, optionally stage an analysis.
  - Phase S (≤10 min): stage a function + pytest under
      src/connectome_gnn/LLM_code/staging/block_NN/
    which must print `PASS:` before Phase C can proceed.
  - Phase C (≤5 min):  minimal wire-up (3–10 lines) in a pre-set allow-list of
    production files.
  - Phase T:            normal cluster training; identical to GNN_LLM.py body.
  - Phase V:            automatic multi-seed triple-check; KEEP or `git revert`
    the block's commits.

All code edits live on branch `agentic_code_change`; `main` stays untouched.
The existing HPO pipeline in `GNN_LLM.py` is NOT modified.

See docs/plan_adaptive_agent.md and src/connectome_gnn/LLM_code/instructions/
for the per-exploration briefs.

Command-line example:

  git checkout agentic_code_change
  python GNN_LLM_code.py -o generate_train_test_plot_Claude \
      flyvis_noise_005_010_code_change --cluster --resume
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before other imports
import argparse
import os
import warnings

from connectome_gnn.LLM_code.pipeline import run_exploration
from connectome_gnn.utils import add_pre_folder, config_path


warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connectome-GNN — Code-change LLM loop"
    )
    parser.add_argument(
        "-o", "--option", nargs="+",
        help="task option names, e.g. generate_train_test_plot_Claude <config>",
    )
    parser.add_argument(
        "--fresh", action="store_true", default=True,
        help="start from iteration 1 (ignore auto-resume)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="auto-resume from last completed batch",
    )
    parser.add_argument(
        "--cluster", action="store_true",
        help="submit training to LSF cluster (default: run locally)",
    )
    return parser.parse_args()


def _resolve_source_config_path(option_arg: str) -> str:
    """Turn a config-name arg (bare name or absolute path) into a YAML path."""
    if os.path.isabs(option_arg) or os.path.isfile(option_arg):
        return option_arg if option_arg.endswith(".yaml") else option_arg + ".yaml"
    config_file, _ = add_pre_folder(option_arg)
    return config_path(f"{config_file}.yaml")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    if not args.option or len(args.option) < 2:
        raise SystemExit(
            "usage: GNN_LLM_code.py -o <task> <config_name> [options...]"
        )
    source_config_path = _resolve_source_config_path(args.option[1])
    if not os.path.isfile(source_config_path):
        raise SystemExit(f"config YAML not found: {source_config_path}")

    run_exploration(args, root_dir, source_config_path)


# --- Usage notes ---
#
# The code-change loop requires being on branch `agentic_code_change`; the
# pipeline will refuse to start otherwise (prevents accidental edits to main).
# Create it once with:
#
#   git checkout main
#   git checkout -b agentic_code_change
#
# Then:
#
#   python GNN_LLM_code.py -o generate_train_test_plot_Claude flyvis_noise_005_010_code_change --cluster
#
