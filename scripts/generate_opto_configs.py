#!/usr/bin/env python
"""Emit the optogenetics sweep configs.

Reads the baseline config (flyvis_noise_free_blank50_unified_cv00.yaml),
applies a per-condition optogenetics block, and writes standalone YAML
files into the repo's config/fly/ directory.

Sweep grid:
    targets   = TmY15, Mi1, T4c    (top-3 positive controls by null_dim)
    waveforms = white_noise, heaviside

Output: 6 YAMLs of the form
    config/fly/flyvis_noise_free_blank50_opto_<target>_<waveform>.yaml

Each opto config produces a dataset under
    graphs_data/fly/flyvis_noise_free_blank50_opto_<target>_<waveform>/

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
# Output prefix drops the "_cv00" suffix — opto runs aren't per-fold.
OUTPUT_PREFIX = "flyvis_noise_free_blank50"

# Top-9 positive controls by null_dim, descending. From
# figures/structural_nullspace_table.json (regenerate via
# src/connectome_gnn/models/structural_nullspace_table.py).
TARGETS = [
    "TmY15",  # 43,299
    "Mi1",    # 25,834
    "Tm3",    # 20,471
    "Tm4",    # 15,971
    "Tm1",    # 15,525
    "Mi4",    # 14,439
    "T4c",    # 12,564
    "Mi9",    # 11,889
    "Tm2",    # 11,068
]

WAVEFORMS = [
    {"kind": "white_noise", "amplitude": 0.0, "noise_level": 0.05},
    {"kind": "heaviside",   "amplitude": 1.0, "noise_level": 0.0},
]


def _waveform_suffix(wf: dict) -> str:
    """Compact filename-friendly tag for a waveform.

    white_noise → noise level encoded as 'NNN' (e.g. 0.05 → '005');
    heaviside, impulse, video, constant → the kind name itself.
    Future noise levels (0.10 → '010', 0.20 → '020') get their own files
    automatically without needing to edit naming code.
    """
    kind = wf["kind"]
    if kind == "white_noise":
        # Match the repo-wide convention: 0.05 → '005', 0.10 → '010',
        # 0.50 → '050' (matches flyvis_noise_005 / _050 dataset names).
        nl = wf["noise_level"]
        return f"{int(round(nl * 100)):03d}"
    return kind


def build_opto_block(cell_type: str, waveform: dict) -> dict:
    """Construct the simulation.optogenetics block for one condition."""
    suffix = _waveform_suffix(waveform)
    return {
        "enabled": True,
        "source_dataset": SOURCE_DATASET,
        "output_suffix": f"_opto_{cell_type}_{suffix}",
        "target": {
            "mode": "cell_type",
            "cell_types": [cell_type],
            "column_distinct": True,
        },
        "waveform": {
            "kind": waveform["kind"],
            "amplitude": waveform["amplitude"],
            "noise_level": waveform["noise_level"],
            "seed": 42,
        },
    }


def emit(dry_run: bool):
    base_path = _resolve_baseline_config_path(BASELINE_CONFIG_NAME)
    print(f"baseline: {base_path}")
    with open(base_path) as f:
        baseline = yaml.safe_load(f)
    # Always emit into the repo's config/fly/ — opto configs are local-only,
    # not part of the data-root config tree.
    out_dir = config_path("fly")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for target in TARGETS:
        for wf in WAVEFORMS:
            cond = f"{target}_{_waveform_suffix(wf)}"
            cfg = copy.deepcopy(baseline)
            cfg["dataset"] = f"{OUTPUT_PREFIX}_opto_{cond}"
            cfg["config_file"] = f"fly/{OUTPUT_PREFIX}_opto_{cond}"
            cfg["description"] = (
                f"Opto run derived from {SOURCE_DATASET}: target={target} "
                f"waveform={wf['kind']} amplitude={wf['amplitude']} "
                f"noise_level={wf['noise_level']}."
            )
            cfg["simulation"]["optogenetics"] = build_opto_block(target, wf)
            out_name = f"{OUTPUT_PREFIX}_opto_{cond}.yaml"
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
