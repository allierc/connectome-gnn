import sys
import os
import shutil

# Ensure src/ is on the path so connectome_gnn is always importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import re

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.graph_data_generator import data_generate
from connectome_gnn.models.graph_trainer import data_train, data_test, data_train_INR
# SPEND-style Noise2Noise trainer (sibling of data_train, data_train_INR).
# Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195)
from connectome_gnn.models.graph_trainer_spend import data_train_spend
from connectome_gnn.utils import (
    set_device, add_pre_folder, log_path, config_path, validate_pre_folder,
    set_data_root, git_sha, git_dirty_files, get_repo_root, graphs_data_path,
    load_config_fallback_roots, load_data_fallback_roots,
)


def _yellow(msg: str) -> str:
    return f"\033[33m{msg}\033[0m"


def _resolve_config_path(yaml_path: str) -> str:
    """If yaml_path doesn't exist at the local repo, try each fallback config
    root from data_paths.json (cluster_data_dir/config, cluster_root_dir/config).
    Returns the first existing path (with a yellow warning), or the original
    path if nothing is found.
    """
    if os.path.isfile(yaml_path):
        return yaml_path
    repo_config_root = os.path.join(get_repo_root(), 'config')
    if not yaml_path.startswith(repo_config_root + os.sep):
        return yaml_path
    rel = os.path.relpath(yaml_path, repo_config_root)
    for root in load_config_fallback_roots():
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            print(_yellow(f"  config not found at {yaml_path}"))
            print(_yellow(f"  using fallback config: {candidate}"))
            return candidate
    return yaml_path


def _maybe_fallback_data_root(config, explicit_output_root: bool, task: str) -> None:
    """If the dataset is missing at the current data root, try each fallback
    data root from data_paths.json (cluster_data_dir). Switch the data root to
    the first one that has it (and print a yellow warning). Skipped when
    --output_root / GNN_OUTPUT_ROOT was explicitly provided or when generating
    fresh data locally.
    """
    if explicit_output_root or 'generate' in task:
        return
    dataset_dir = graphs_data_path(config.dataset)
    if os.path.isdir(dataset_dir):
        return
    for root in load_data_fallback_roots():
        candidate = os.path.join(root, 'graphs_data', config.dataset)
        if os.path.isdir(candidate):
            print(_yellow(f"  data not found at {dataset_dir}"))
            print(_yellow(f"  switching data root to: {root}"))
            set_data_root(root)
            return

# Optional imports (not available in flyvis-gnn spinoff)
try:
    from connectome_gnn.models.NGP_trainer import data_train_NGP
except ImportError:
    data_train_NGP = None
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="connectome_gnn")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="CV: number of seeds (default 5, uses 42..42+N-1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="CV: comma-separated seeds, e.g. 42,43,44 (overrides --n_seeds)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory for log/ and  (default: cwd)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of data even if it already exists")
    parser.add_argument("--skip_phase2", action="store_true", default=False,
                        help="CV: skip phase 2 (zero-shot DAVIS→hold-out test). Use when no pre-trained DAVIS model exists.")
    parser.add_argument("--test_mode", type=str, default="",
                        help='Test-time variant, e.g. "test_ablation_50" (zero out 50%% of edges before rollout) or "test_modified_0.1" (add Gaussian noise σ=0.1 to W).')

    print()
    device = []
    args = parser.parse_args()

    output_root = args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
    explicit_output_root = output_root is not None

    if output_root:
        assert os.path.isdir(output_root), f"--output_root does not exist: {output_root}"
        assert os.access(output_root, os.W_OK), f"--output_root is not writable: {output_root}"
        set_data_root(output_root)

    if args.option:
        print(f"Options: {args.option}")
    CONFIG_LISTS = {
        'flyvis_baselines': [
            '/groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_baseline_00',
            '/groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_010_baseline_00',
            '/groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_stride_5_baseline_00',
            '/groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_stride_5_yt_baseline_00',
        ],
        'drosophila_cx_baselines': [
            'drosophila_cx_known_ode',
            'drosophila_cx_rnn',
            'drosophila_cx_neuralode',
        ],
        'known_ode': [
            'flyvis_noise_free_known_ode',
            'flyvis_noise_005_known_ode',
            'flyvis_noise_05_known_ode',
            'flyvis_noise_005_INR_known_ode',
        ],
        'retest_noisy_rollouts': [
            *[f'flyvis_noise_005_cv{i:02d}' for i in range(10)],
            *[f'flyvis_noise_05_cv{i:02d}' for i in range(10)],
            *[f'flyvis_noise_free_default_cv{i:02d}' for i in range(10)],
            *[f'flyvis_noise_005_default_cv{i:02d}' for i in range(10)],
            *[f'flyvis_noise_05_default_cv{i:02d}' for i in range(10)],
        ],
        'flyvis_blank_sweep': [
            'flyvis_noise_005_blank01',
            'flyvis_noise_005_blank05',
            'flyvis_noise_005_blank10',
            'flyvis_noise_005_blank25',
            'flyvis_noise_005_blank50',
        ],
        'flyvis_noise_free_blank50_unified_cv': [
            'flyvis_noise_free_blank50_unified_cv00',
            'flyvis_noise_free_blank50_unified_cv01',
            'flyvis_noise_free_blank50_unified_cv02',
            'flyvis_noise_free_blank50_unified_cv03',
            'flyvis_noise_free_blank50_unified_cv04',
        ],
        'flyvis_noise_005_blank50_unified_cv': [
            'flyvis_noise_005_blank50_unified_cv00',
            'flyvis_noise_005_blank50_unified_cv01',
            'flyvis_noise_005_blank50_unified_cv02',
            'flyvis_noise_005_blank50_unified_cv03',
            'flyvis_noise_005_blank50_unified_cv04',
        ],
        'flyvis_noise_05_blank50_unified_cv': [
            'flyvis_noise_05_blank50_unified_cv00',
            'flyvis_noise_05_blank50_unified_cv01',
            'flyvis_noise_05_blank50_unified_cv02',
            'flyvis_noise_05_blank50_unified_cv03',
            'flyvis_noise_05_blank50_unified_cv04',
        ],
        'hybrid_flywireRF_variants': [
            'flyvis_hybrid_flywireRF_noise_005',
            'flyvis_hybrid_flywireRF_e15_noise_005',
            'flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005',
            'flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_noise_005',
            'flyvis_hybrid_flywireRF_zeroedge_sl_noise_005',
        ],
        # SPEND-style Noise2Noise trainer; invoke with -o train_SPEND flyvis_spend.
        # Cite: https://github.com/buchenglab/SPEND
        'flyvis_spend': [
            'flyvis_noise_005_010_spend_replay',
            'flyvis_noise_005_010_spend_time',
            'flyvis_noise_005_010_spend_typed',
            'flyvis_noise_005_010_spend_combined',
        ],
    }

    if args.option is not None:
        task = args.option[0]
        config_name = args.option[1]
        # Support passing a full path whose basename is a CONFIG_LISTS key,
        # e.g. /groups/.../config/fly/flyvis_blank_sweep
        _list_key = config_name if config_name in CONFIG_LISTS else os.path.basename(config_name)
        if _list_key in CONFIG_LISTS:
            _entries = CONFIG_LISTS[_list_key]
            # If config_name was an absolute path, expand relative entries to that directory
            if os.path.isabs(config_name):
                _dir = os.path.dirname(config_name)
                config_list = [
                    os.path.join(_dir, e) if not os.path.isabs(e) else e
                    for e in _entries
                ]
            else:
                config_list = _entries
            best_model = None
            test_config_name = None
        else:
            config_list = [config_name]
            if len(args.option) > 2:
                best_model = args.option[2]
            else:
                best_model = None
            if len(args.option) > 3:
                test_config_name = args.option[3]
            else:
                test_config_name = None
    else:
        best_model = ''
        task = task = 'plot'
        config_list = ['flyvis_noise_free_blank50_unified_cv00']
        test_config_name = None

    if task == 'cv':
        from connectome_gnn.models.cv_runner import run_cv
        if args.seeds is not None:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
        else:
            seeds = list(range(42, 42 + args.n_seeds))
        run_cv(config_name, seeds, skip_phase2=args.skip_phase2)
        sys.exit(0)

    for config_file_ in config_list:
        print(" ")

        if os.path.isfile(config_file_) or os.path.isabs(config_file_):
            # config_file_ is a direct filesystem path — load without repo lookup.
            # Append .yaml if not already present.
            # pre_folder is derived from the parent directory name.
            yaml_file = config_file_ if config_file_.endswith('.yaml') else config_file_ + '.yaml'
            parent = os.path.basename(os.path.dirname(os.path.abspath(yaml_file)))
            pre_folder = parent + "/" if parent else ""
            validate_pre_folder(pre_folder)
            config = NeuralGraphConfig.from_yaml(yaml_file)
            if not config.dataset.startswith(pre_folder):
                config.dataset = pre_folder + config.dataset
            # If config_file is still the default "none", derive it from the YAML path
            # so logs go to log/<domain>/<config_name>/ not log/none/
            if config.config_file == "none":
                stem = os.path.splitext(os.path.basename(yaml_file))[0]
                config.config_file = pre_folder + stem
        else:
            config_file, pre_folder = add_pre_folder(config_file_)

            # load config — if YAML not found, try stripping _cvNN suffix (CV folds
            # share a base config; the cv_runner overrides dataset/config_file at runtime)
            yaml_path = _resolve_config_path(config_path(f"{config_file}.yaml"))
            cv_match = re.search(r'_cv(\d+)$', config_file_)
            if not os.path.isfile(yaml_path) and cv_match:
                base_name = config_file_[:cv_match.start()]
                base_file, _ = add_pre_folder(base_name)
                print(f"  CV fold detected: loading base config {base_name}.yaml, "
                      f"dataset/log -> {config_file_}")
                yaml_file = _resolve_config_path(config_path(f"{base_file}.yaml"))
                config = NeuralGraphConfig.from_yaml(yaml_file)
                config.dataset = pre_folder + config_file_
                config.config_file = pre_folder + config_file_
            else:
                yaml_file = yaml_path
                config = NeuralGraphConfig.from_yaml(yaml_file)
                if not config.dataset.startswith(pre_folder):
                    config.dataset = pre_folder + config.dataset
                config.config_file = pre_folder + config_file_

        _maybe_fallback_data_root(config, explicit_output_root, task)

        if device == []:
            device = set_device(config.training.device)

        run_log_dir = log_path(config.config_file)
        sha = git_sha()
        dirty = git_dirty_files()
        if dirty:
            sha = sha + '-dirty'

        # Snapshot the source config yaml into the run directory so the run is self-describing
        os.makedirs(run_log_dir, exist_ok=True)
        shutil.copy2(yaml_file, os.path.join(run_log_dir, 'config.yaml'))

        # Reset _complete at the start of each run
        _complete_path = os.path.join(run_log_dir, '_complete')
        if os.path.exists(_complete_path):
            os.remove(_complete_path)

        if "generate" in task:
            _marker = os.path.join(run_log_dir, '_completed_generate')
            if os.path.exists(_marker):
                os.remove(_marker)
            data_generate(
                config,
                device=device,
                visualize=True,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=args.force,
                save=True,
                step=100,
                compute_ranks=False,
            )
            os.makedirs(run_log_dir, exist_ok=True)
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        if 'train_NGP' in task:
            _marker = os.path.join(run_log_dir, '_completed_train')
            if os.path.exists(_marker):
                os.remove(_marker)
            # use new modular NGP trainer pipeline
            data_train_NGP(config=config, device=device)
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        elif 'train_INR' in task:
            _marker = os.path.join(run_log_dir, '_completed_train')
            if os.path.exists(_marker):
                os.remove(_marker)
            # train INR (SIREN/NGP) on a field from x_list_train
            # usage: -o train_INR [field_name] [inr_type]
            # field_name: stimulus (default), voltage, calcium, fluorescence
            # inr_type: siren_txy (default for stimulus), siren_t (for voltage)
            field_name = args.option[2] if len(args.option) > 2 else 'stimulus'
            inr_type_arg = args.option[3] if len(args.option) > 3 else None
            data_train_INR(config=config, device=device, total_steps=100000,
                           field_name=field_name, n_training_frames=0,
                           inr_type=inr_type_arg)
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        elif 'train_SPEND' in task:
            _marker = os.path.join(run_log_dir, '_completed_train')
            if os.path.exists(_marker):
                os.remove(_marker)
            # SPEND-style Noise2Noise trainer.
            # Cite: https://github.com/buchenglab/SPEND
            # usage: -o train_SPEND <config>          (single config)
            #        -o train_SPEND flyvis_spend      (CONFIG_LIST -> 4 SPEND yamls)
            data_train_spend(
                config=config,
                erase=True,
                best_model=best_model,
                device=device,
            )
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        elif "train" in task:
            _marker = os.path.join(run_log_dir, '_completed_train')
            if os.path.exists(_marker):
                os.remove(_marker)
            data_train(
                config=config,
                erase=True,
                best_model=best_model,
                style='color',
                device=device,
            )
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        if "test" in task:
            _marker = os.path.join(run_log_dir, '_completed_test')
            if os.path.exists(_marker):
                os.remove(_marker)
            # Optional: load a second config for cross-dataset test data
            test_config = None
            if test_config_name:
                # Accept paths with or without .yaml extension
                tc_yaml = test_config_name if test_config_name.endswith('.yaml') else test_config_name + '.yaml'
                if os.path.isfile(tc_yaml):
                    tc_parent = os.path.basename(os.path.dirname(os.path.abspath(tc_yaml)))
                    tc_pre = tc_parent + "/" if tc_parent else ""
                    validate_pre_folder(tc_pre)
                    test_config = NeuralGraphConfig.from_yaml(tc_yaml)
                    if not test_config.dataset.startswith(tc_pre):
                        test_config.dataset = tc_pre + test_config.dataset
                    # test_config.config_file left as-is from the YAML
                else:
                    tc_file, tc_pre = add_pre_folder(test_config_name)
                    test_config = NeuralGraphConfig.from_yaml(config_path(f"{tc_file}.yaml"))
                    if not test_config.dataset.startswith(tc_pre):
                        test_config.dataset = tc_pre + test_config.dataset
                    test_config.config_file = tc_pre + test_config_name
                print(f'cross-dataset test: model from {config.dataset}, test data from {test_config.dataset}')

            data_test(
                config=config,
                visualize=True,
                style="color name continuous_slice",
                verbose=False,
                best_model=best_model if best_model else 'best',
                run=0,
                test_mode=args.test_mode,   # e.g. "test_ablation_50"
                sample_embedding=False,
                step=10,
                n_rollout_frames=250,
                device=device,
                particle_of_interest=0,
                new_params=None,
                rollout_without_noise=True,
                test_config=test_config,
            )
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        if 'plot' in task:
            _marker = os.path.join(run_log_dir, '_completed_plot')
            if os.path.exists(_marker):
                os.remove(_marker)
            folder_name = log_path(pre_folder, 'tmp_results') + '/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True, skip_svd=True)
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        # Write commit SHA and completion marker for this run
        with open(os.path.join(run_log_dir, '_commit'), 'w') as f:
            f.write(f"{sha}\n")
            if dirty:
                f.write("dirty_files:\n")
                for line in dirty:
                    f.write(f"  {line}\n")
        with open(os.path.join(run_log_dir, '_complete'), 'w') as f:
            f.write(f"commit={sha}\nargv={sys.argv}\n")




# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_noise_005"
# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o cv /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_noise_005 --n_seeds 5"


# python GNN_Main.py -o cv flyvis_noise_005 --n_seeds 10
# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 "python GNN_Main.py -o train null_edges_cross"
# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_noise_005"

# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 \
#   -o logs/cv_cross.out -e logs/cv_cross.err \
#   "bash run_cv_null_edges_cross.sh"

# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 10:00 -Is "python GNN_Main.py -o train_test_plot null_edges_cross"
# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is  -o logs/cv_cross.out -e logs/cv_cross.err   "bash run_cv_null_edges_cross.sh"
# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o cv /groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005 --n_seeds 5 "

# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train /groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_stride_5_yt_Claude_00"

# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o generate_train_test_plot flyvis_hybrid_zeroedge_e15_u2_noise_005"
# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o generate_train_test_plot flyvis_hybrid_flywireRF_zeroedge_e15_u2_noise_005"

# export GNN_OUTPUT_ROOT=graphs_data
# unset GNN_OUTPUT_ROOT
# CUDA_VISIBLE_DEVICES=1 python GNN_Main.py -o train_test_plot flyvis_noise_005_hidden_010_ngp_anchors --output_root /groups/saalfeld/home/allierc/GraphData
# CUDA_VISIBLE_DEVICES=0 python GNN_Main.py -o train_test_plot flyvis_noise_005_ss0 --output_root /groups/saalfeld/home/allierc/GraphData 
# python GNN_Main.py -o test   /groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_blank50_unified_cv00   --output_root /groups/saalfeld/home/allierc/GraphData
# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o generate_train_test_plot hybrid_flywireRF_variants --force"

# bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_noise_005 --force"
#  bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train_test_plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_known_ode_noise_005 --force"
#  bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train_test_plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_known_ode_noise_005 --force"
#  bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o test_plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005 --force"
# bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_hybrid_flywireRF_zeroedge_sl_noise_005 --force"
#  bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train_test_plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_hybrid_flywireRF_zeroedge_sl_known_ode_noise_005 --force"python GNN_Main.py -o plot flyvis_noise_free_blank50_unified_cv00
# python GNN_Main.py -o plot flyvis_noise_free_blank50_unified_cv00