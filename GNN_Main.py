import sys
import os
import shutil

# Ensure src/ is on the path so connectome_gnn is always importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.graph_data_generator import data_generate
from connectome_gnn.models.graph_trainer import (
    data_train, data_test, data_train_INR, data_train_task,
)
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.utils import (
    set_device, add_pre_folder, log_path, config_path, validate_pre_folder,
    set_data_root, git_sha, git_dirty_files,
)


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
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest completed per-epoch checkpoint "
                             "(continues at the next epoch, iter 0); does not erase the log dir")
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
        # Yang 2019 cognitive battery — 20 single-task configs (all use the
        # same TaskRNN free-W architecture and ruleset='all' for I/O shape,
        # but differ in which rule is sampled). Use `python GNN_Main.py -o
        # generate cortex_all_tasks` to generate the 20 per-task datasets in
        # one process (sequential).
        'cortex_all_tasks': [
            'cortex_fdgo', 'cortex_reactgo', 'cortex_delaygo',
            'cortex_fdanti', 'cortex_reactanti', 'cortex_delayanti',
            'cortex_dm1', 'cortex_dm2',
            'cortex_contextdm1', 'cortex_contextdm2', 'cortex_multidm',
            'cortex_delaydm1', 'cortex_delaydm2',
            'cortex_contextdelaydm1', 'cortex_contextdelaydm2', 'cortex_multidelaydm',
            'cortex_dmsgo', 'cortex_dmsnogo', 'cortex_dmcgo', 'cortex_dmcnogo',
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
        task = task = 'train'
        config_list = ['drosophila_cx_pi_voltage_noise_005']
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

        config, yaml_file = load_run_config(config_file_, explicit_output_root, task)

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

        if 'train_task' in task:
            _marker = os.path.join(run_log_dir, '_completed_train')
            if os.path.exists(_marker):
                os.remove(_marker)
            data_train_task(
                config=config, erase=not args.resume, best_model=best_model,
                device=device, resume=args.resume,
            )
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        elif 'train_NGP' in task:
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

        elif "train" in task:
            _marker = os.path.join(run_log_dir, '_completed_train')
            if os.path.exists(_marker):
                os.remove(_marker)
            data_train(
                config=config,
                erase=not args.resume,
                best_model=best_model,
                style='color',
                device=device,
                resume=args.resume,
            )
            with open(_marker, 'w') as f:
                f.write(f"commit={sha}\nargv={sys.argv}\n")

        if "test" in task:
            _marker = os.path.join(run_log_dir, '_completed_test')
            if os.path.exists(_marker):
                os.remove(_marker)
            # Release training-phase CUDA memory (incl. CUDA Graphs private pools)
            # before allocating the test model. Without this, large e15 GNN runs
            # OOM at test-time model creation despite total need being modest.
            try:
                import gc as _gc
                import torch as _torch
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    if hasattr(_torch.cuda, 'reset_peak_memory_stats'):
                        _torch.cuda.reset_peak_memory_stats()
            except Exception as _e:
                print(f'[warn] pre-test cuda cleanup failed: {_e}')
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
            # `pre_folder` used to be set inline before load_run_config absorbed
            # the config-loading block. Derive it from config.config_file
            # (which now has the form "<pre_folder>/<config_name>").
            pre_folder = os.path.dirname(config.config_file)
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

# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o generate_train_test_plot full_eye_flywireRF_proximal_nulls_noise_005"

# export GNN_OUTPUT_ROOT=graphs_data
# unset GNN_OUTPUT_ROOT
# CUDA_VISIBLE_DEVICES=1 python GNN_Main.py -o train_test_plot flyvis_noise_005_hidden_010_ngp_anchors --output_root /groups/saalfeld/home/allierc/GraphData
# CUDA_VISIBLE_DEVICES=0 python GNN_Main.py -o train_test_plot flyvis_noise_005_ss0 --output_root /groups/saalfeld/home/allierc/GraphData 
# python GNN_Main.py -o test   /groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_blank50_unified_cv00   --output_root /groups/saalfeld/home/allierc/GraphData
# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o generate_train_test_plot hybrid_flywireRF_variants --force"

# bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o plot config/fly/full_eye_flywireRF_proximal_nulls_noise_005 --force"
#  bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train_test_plot config/fly/full_eye_flywireRF_proximal_nulls_known_ode_noise_005 --force"
#  bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o test_plot config/fly/e8_flywireRF_proximal_nulls_known_ode_noise_005 --force"
# python GNN_Main.py -o plot flyvis_noise_free_blank50_unified_cv00

# python GNN_Main.py -o train flyvis_noise_005_blank50_heaviside_var_cv00