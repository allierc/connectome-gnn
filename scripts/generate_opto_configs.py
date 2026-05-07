#!/usr/bin/env python
"""Emit the optogenetics first-glimpse sweep configs.

Reads the baseline config (flyvis_noise_free_blank50_unified_cv00.yaml),
applies a per-condition optogenetics block, and writes 10 standalone YAML
files to <data_root>/config/fly/.

Sweep grid:
    targets   = TmY15, Mi1, T4c    (positive controls — ranked by null_dim)
                R1, T1              (negative controls — identifiable types)
    waveforms = white_noise, heaviside

Output: 10 YAMLs of the form
    flyvis_noise_free_blank50_cv00_opto_<target>_<waveform>.yaml

Each opto config produces a dataset under
    graphs_data/fly/flyvis_noise_free_blank50_cv00_opto_<target>_<waveform>/

Source dataset (must already exist on disk):
    fly/flyvis_noise_free_blank50_cv00

Usage:
    python scripts/generate_opto_configs.py
    python scripts/generate_opto_configs.py --dry-run   # print only
"""
import argparse
import copy
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import yaml  # noqa: E402

from connectome_gnn.utils import config_path, get_data_root, load_data_root_from_json  # noqa: E402


def _resolve_baseline_config_path(name: str) -> str:
    """Find the baseline YAML, trying repo-config/ first then data-root/config/."""
    candidates = [config_path("fly", f"{name}.yaml")]
    try:
        load_data_root_from_json()
        candidates.append(os.path.join(get_data_root(), "config", "fly", f"{name}.yaml"))
    except Exception:
        pass
    # Common explicit fallback for this user's environment.
    candidates.append(f"/groups/saalfeld/home/allierc/GraphData/config/fly/{name}.yaml")
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"baseline config not found: {name!r}; tried {candidates}"
    )


BASELINE_CONFIG_NAME = "flyvis_noise_free_blank50_unified_cv00"
SOURCE_DATASET = "flyvis_noise_free_blank50_cv00"

POSITIVE_TARGETS = ["TmY15", "Mi1", "T4c"]
NEGATIVE_TARGETS = ["R1", "T1"]
TARGETS = POSITIVE_TARGETS + NEGATIVE_TARGETS

WAVEFORMS = [
    {"kind": "white_noise", "amplitude": 0.0, "noise_level": 0.05},
    {"kind": "heaviside",   "amplitude": 1.0, "noise_level": 0.0},
]


def build_opto_block(cell_type: str, waveform: dict) -> dict:
    """Construct the simulation.optogenetics block for one condition."""
    return {
        "enabled": True,
        "source_dataset": SOURCE_DATASET,
        "output_suffix": f"_opto_{cell_type}_{waveform['kind']}",
        "target": {
            "mode": "cell_type",
            "cell_types": [cell_type],
            "column_distinct": True,
        },
        "waveform": {
            "kind": waveform["kind"],
            "amplitude": waveform["amplitude"],
            "noise_level": waveform["noise_level"],
            "onset_frame": 0,
            "offset_frame": -1,
            "seed": 42,
        },
    }


def emit(dry_run: bool):
    base_path = _resolve_baseline_config_path(BASELINE_CONFIG_NAME)
    print(f"baseline: {base_path}")
    with open(base_path) as f:
        baseline = yaml.safe_load(f)
    out_dir = os.path.dirname(base_path)
    written = []

    for target in TARGETS:
        for wf in WAVEFORMS:
            cond = f"{target}_{wf['kind']}"
            cfg = copy.deepcopy(baseline)
            cfg["dataset"] = f"{SOURCE_DATASET}_opto_{cond}"
            cfg["config_file"] = f"fly/{SOURCE_DATASET}_opto_{cond}"
            cfg["description"] = (
                f"Opto twin of {SOURCE_DATASET}: target={target} waveform={wf['kind']} "
                f"amplitude={wf['amplitude']} noise_level={wf['noise_level']}. "
                f"{'positive' if target in POSITIVE_TARGETS else 'negative'} control."
            )
            cfg["simulation"]["optogenetics"] = build_opto_block(target, wf)
            out_name = f"{SOURCE_DATASET}_opto_{cond}.yaml"
            out_path = os.path.join(out_dir, out_name)
            written.append(out_path)
            if dry_run:
                print(f"would write: {out_path}")
                continue
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"wrote: {out_path}")
    print(f"\n{len(written)} configs {'planned' if dry_run else 'written'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print intended outputs without writing")
    args = parser.parse_args()
    emit(args.dry_run)


if __name__ == "__main__":
    main()
