"""Emit the MLP/EED training configs for the measurement-noise ladder.

The generator writes the observation-noise realisation into
``x_list_<split>/noise.zarr`` and keeps ``voltage`` clean, so a run only sees
measurement noise when ``simulation.measurement_noise_level`` is set — that
field is what makes ``determine_load_fields`` request the ``noise`` array and
what gates ``apply_measurement_noise``. Pointing ``dataset`` at a noisy
directory without it trains on clean voltage.

Datasets are named ``flyvis_noise_<process>_<measurement>_blank50_cv<NN>``;
process noise is fixed at 0.05 across the ladder and only the measurement
level varies. The sigma=0 rung has no measurement token in its dataset name
(``flyvis_noise_005_blank50_cv00``) and is the in-ladder control.

Usage
-----
    python tools/make_meas_noise_configs.py            # cv00, all rungs
    python tools/make_meas_noise_configs.py --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config" / "fly"

# Base configs: the canonical baseline pair (see figures/scripts/gen_noisy_baselines.sh,
# which drives the published table off *_unified2_cv00). EED trains with
# rollout_train_steps=1 and MLP with 20 — matching the two table rows.
BASES = {
    "eed": "flyvis_noise_005_eed_unified2.yaml",
    "mlp": "flyvis_noise_005_mlp_unified2.yaml",
}

# (name token, base key, label, overrides). The token lands in the config name,
# so a variant that changes a training knob gets its own name -- and therefore
# its own log dir -- rather than silently reusing the stock one.
#
# mlp_rts1 drops the MLP's 20-step unrolled objective to the single-step
# supervision EED already uses, so the two baselines differ only in
# architecture and the rollout comparison is not confounded by the objective.
MODELS = [
    ("eed", "eed", "EED", {}),
    ("mlp", "mlp", "MLP", {}),
    ("mlp_rts1", "mlp", "MLP (rollout_train_steps=1)",
     {"training": {"rollout_train_steps": 1}}),
]

# (config token, dataset infix, sigma_meas). The dataset infix is empty for the
# control rung, whose directory carries no measurement token.
RUNGS = [
    ("000", "", 0.0),
    ("010", "010_", 0.1),
    ("020", "020_", 0.2),
    ("050", "050_", 0.5),
]

PROCESS_NOISE = 0.05


def build(model: str, base_key: str, label: str, overrides: dict,
          token: str, infix: str, sigma: float, fold: int) -> tuple[str, dict]:
    base = yaml.safe_load((CONFIG_DIR / BASES[base_key]).read_text())

    dataset = f"flyvis_noise_005_{infix}blank50_cv{fold:02d}"
    name = f"flyvis_noise_005_meas_{token}_{model}_blank50_cv{fold:02d}"

    base["dataset"] = dataset
    base["description"] = (
        f"{label} blank50 baseline on the measurement-noise ladder: "
        f"process sigma=0.05, measurement sigma={sigma}, fold {fold:02d}"
    )
    base["simulation"]["noise_model_level"] = PROCESS_NOISE
    base["simulation"]["measurement_noise_level"] = sigma

    for section, fields in overrides.items():
        base[section].update(fields)

    # The claude block drives the agentic hyperparameter loop, not a plain
    # training run, and its case_study_brief names a different dataset.
    base.pop("claude", None)

    return name, base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0],
                    help="CV folds to emit (default: 0)")
    ap.add_argument("--models", nargs="+", default=[m[0] for m in MODELS],
                    choices=[m[0] for m in MODELS],
                    help="model variants to emit (default: all)")
    args = ap.parse_args()

    written = []
    for fold in args.folds:
        for model, base_key, label, overrides in MODELS:
            if model not in args.models:
                continue
            for token, infix, sigma in RUNGS:
                name, cfg = build(model, base_key, label, overrides,
                                  token, infix, sigma, fold)
                path = CONFIG_DIR / f"{name}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
                written.append((name, cfg["dataset"], sigma,
                                cfg["training"]["rollout_train_steps"]))

    for name, dataset, sigma, rts in written:
        print(f"{name:56s} -> {dataset:38s} sigma_meas={sigma}  rts={rts}")
    print(f"\n{len(written)} configs written to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
