#!/usr/bin/env python
"""Single-config driver for add_optogenetics_stimulus.

Reads a YAML config (must have simulation.optogenetics.enabled=True and
simulation.optogenetics.source_dataset set), then calls the re-simulation
pass that produces a new dataset under config.dataset with the configured
optogenetic perturbation added.

Usage:
    python scripts/run_add_optogenetics.py config/fly/<name>.yaml

The output dataset is a fully self-contained V3 zarr directory:
    graphs_data/<config.dataset>/x_list_{train,test}/
        voltage.zarr stimulus.zarr noise.zarr optogenetics_stimulus.zarr
        pos.zarr group_type.zarr neuron_type.zarr
"""
import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.optogenetics import add_optogenetics_stimulus  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        # try repo-relative if not found at the literal path
        candidates = [cfg_path, os.path.join(REPO_ROOT, cfg_path)]
        cfg_path = next((p for p in candidates if os.path.isfile(p)), cfg_path)

    config = NeuralGraphConfig.from_yaml(cfg_path)
    logging.info(f"loaded config: {cfg_path}")
    logging.info(f"  dataset:        {config.dataset}")
    logging.info(f"  source_dataset: {config.simulation.optogenetics.source_dataset}")
    logging.info(f"  target.mode:    {config.simulation.optogenetics.target.mode}")
    logging.info(f"  waveform.kind:  {config.simulation.optogenetics.waveform.kind}")

    add_optogenetics_stimulus(config)


if __name__ == "__main__":
    main()
