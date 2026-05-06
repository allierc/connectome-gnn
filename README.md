# Connectome-GNN

Graph neural networks recover interpretable circuit models from neural activity.

Synapse-level connectomes describe the structure of neural circuits but not their dynamics; neural
activity recordings capture dynamics but not structure. Recovering circuit parameters from activity
alone is possible when both connectivity and activity are sufficiently rich and identifiable. This
may fail on the sparse, repeat-column architectures of biological nervous systems, where parameter
symmetries and limited state-space coverage can make additional constraints from connectivity or
perturbations necessary to disambiguate the inverse problem. We explore conditions under which
mechanistic inference from connectivity and activity is feasible, using a connectome-constrained
simulator. We train graph neural networks (GNNs) on a Drosophila visual-system simulation with
13,741 neurons, 65 cell types, and up to 434,112 synaptic connections, driven by naturalistic video.
Given the binary connectome graph and activity, the GNNs jointly recover the unknown per-neuron
update functions and per-synapse interaction functions, together with effective connectivity
weights, membrane time constants, resting potentials, and unsupervised cell-type clusters---on par
with an oracle that knows the true ODE form. A SIREN extension additionally recovers unknown visual
stimuli. Recovery remains robust when the connectome prior is relaxed by adding +400\% random edges,
and across FlyWire-derived connectome variants with heterogeneous wiring. We find that per-neuron
noise breaks degeneracies that otherwise render the system unidentifiable, suggesting that
perturbations enhancing data diversity are essential when activity alone is insufficient. ML
baselines match one-step activity prediction but their learned dynamics do not encode the underlying
connectivity neural dynamics, showing that fitting activity is a poor proxy for recovering
mechanism. The repository also includes an agentic workflow for hyper-parameter optimization in this
ill-posed inverse problem.

## Installation

### Conda environment

- Linux:

```bash
conda env create -f envs/environment.linux.yaml
conda activate connectome-gnn
```

After install make sure that CUDA enabled pytorch wheels were successfully downloaded.

- Mac:

```bash
conda env create -f envs/environment.mac.yaml
conda activate connectome-gnn
```

Note: we make use of `torch.compile` with `reduce-overhead`, which is a CUDA-only feature.
`torch.compile` is only guaranteed to work on CUDA — it currently fails on Apple Silicon / MPS. On
a Mac, run on CPU (see the smoke-test section below for the one-line `sed` to flip the configs to
`device: cpu`); if you hit other compile issues, turn compilation off entirely or switch the
`torch.compile` calls to `mode="default"`.

Run `conda activate connectome-gnn && pip install -e .` to add src/ to the PYTHONPATH and install
the package into the environment. Or set `$PYTHONPATH`.

### Data dependencies

- FlyVis model: the pretrained flyvis model (model 000, ~105 KB) is bundled in
  `assets/flyvis_model/` and used automatically.

- DAVIS-2017 dataset stimuli

Download the [DAVIS 2017](https://davischallenge.org/davis2017/code.html) dataset (480p). We split
the data for training and validation as below:

```bash
$ tree -L 3 DAVIS2017-train-val
DAVIS2017-train-val
└── JPEGImages
    └── 480p
        ├── bear
        ├── bike-trial
        ├── blackswan
        ├── bmx-bumps
        ├── bmx-trees
        ├── boat
        ├── boxing
        ├── breakdance
        ├── breakdance-flare
        ├── burnout
        ├── bus
        ├── camel
        ├── car-roundabout
        ├── car-shadow
        ├── car-turn
        ├── choreography
        ├── cows
        ├── dance-jump
        ├── dance-twirl
        ├── demolition
        ├── dive-in
        ├── dog
        ├── dog-agility
        ├── dog-control
        ├── dolphins
        ├── drift-chicane
        ├── drift-straight
        ├── drift-turn
        ├── e-bike
        ├── elephant
        ├── flamingo
        ├── goat
        ├── grass-chopper
        ├── hike
        ├── hockey
        ├── horsejump-high
        ├── horsejump-low
        ├── hurdles
        ├── inflatable
        ├── juggle
        ├── kart-turn
        ├── kids-turning
        ├── kite-surf
        ├── kite-walk
        ├── libby
        ├── lions
        ├── lucia
        ├── mallard-fly
        ├── mallard-water
        ├── mbike-santa
        ├── monkeys
        ├── motocross-bumps
        ├── motocross-jump
        ├── motorbike
        ├── ocean-birds
        ├── paragliding
        ├── paragliding-launch
        ├── parkour
        ├── pole-vault
        ├── rhino
        ├── rollerblade
        ├── running
        ├── scooter-black
        ├── scooter-gray
        ├── selfie
        ├── skydive
        ├── soapbox
        ├── soccerball
        ├── speed-skating
        ├── stroller
        ├── surf
        ├── swing
        ├── swing-boy
        ├── tackle
        ├── tennis
        ├── train
        ├── turtle
        ├── varanus-tree
        ├── vietnam
        └── wings-turn
```

The holdout test data consists of these videos, put them in a separate DIR like this:

```bash
$ tree -L 3 DAVIS2017-holdout-test
DAVIS2017-holdout-test
└── JPEGImages
    └── 480p
        ├── aerobatics
        ├── bike-packing
        ├── boxing-fisheye
        ├── carousel
        ├── car-race
        ├── cat-girl
        ├── cats-car
        ├── chamaleon
        ├── classic-car
        ├── color-run
        ├── crossing
        ├── dancing
        ├── deer
        ├── disc-jockey
        ├── dog-gooses
        ├── dogs-jump
        ├── dogs-scale
        ├── drone
        ├── giant-slalom
        ├── girl-dog
        ├── gold-fish
        ├── golf
        ├── guitar-violin
        ├── gym
        ├── helicopter
        ├── horsejump-stick
        ├── hoverboard
        ├── india
        ├── judo
        ├── kid-football
        ├── koala
        ├── lab-coat
        ├── lady-running
        ├── lindy-hop
        ├── loading
        ├── lock
        ├── longboard
        ├── man-bike
        ├── mbike-trick
        ├── miami-surf
        ├── monkeys-trees
        ├── mtb-race
        ├── night-race
        ├── orchid
        ├── people-sunset
        ├── pigs
        ├── planes-crossing
        ├── planes-water
        ├── rallye
        ├── rollercoaster
        ├── salsa
        ├── schoolgirls
        ├── scooter-board
        ├── seasnake
        ├── sheep
        ├── shooting
        ├── skate-jump
        ├── skate-park
        ├── slackline
        ├── snowboard
        ├── stunt
        ├── subway
        ├── tandem
        ├── tennis-vest
        ├── tractor
        ├── tractor-sand
        ├── tuk-tuk
        ├── upside-down
        ├── varanus-cage
        └── walking

```

Set these two env vars to the downloaded data:

```bash
export DATAVIS_ROOT=/path/to/DAVIS2017-train-val
export DATAVIS_TEST_ROOT=/path/to/DAVIS2017-holdout-test
```

## Usage

By default the code creates simulation and model outputs in the repo root. These are .gitignored by
default. This may be undesirable since the contents can be large >> 100MB. Set the env var
`$GNN_OUTPUT_ROOT` to redirect elsewhere.

For any config in `./config/fly/*.yaml`, run data generation, training & testing like this:

```bash
# Single training run
conda activate connectome-gnn
python GNN_Main.py -o generate_train_test_plot flyvis_noise_05_blank50_unified_cv00
```

WARNING: many different models share the same generated data set by the `dataset:` tag in the
config.yaml. If you run two configs in parallel with `-o generate` and they share the same
`config.dataset` tag, they will overwrite, which will lead to unpredictable results. So generate
data for 1 config for each unique `config.dataset` first with `-o generate` alone. Then run
`GNN_Main.py -o train_test_plot` all in parallel.

### Smoke test (all five model types)

The repo ships five tiny configs in `config/fly/flyvis_tiny_*_cv00.yaml` — one each for **GNN**,
**Known ODE**, **MLP**, **EED**, and **Stimulus** — that all share the dataset tag
`flyvis_tiny_cv00`. They use 10k frames, `data_augmentation_loop=1`, and `n_epochs=1`, so a full
generate + train_test_plot sweep finishes in a few minutes on a modest machine and exercises every
architecture end-to-end.

Generate the shared dataset once, then train+test+plot each architecture:

```bash
# one-time generate (any of the five configs works — they share dataset flyvis_tiny_cv00)
python GNN_Main.py -o generate flyvis_tiny_gnn_cv00

# train + test + plot, one per architecture
for cfg in flyvis_tiny_gnn_cv00 \
           flyvis_tiny_known_ode_cv00 \
           flyvis_tiny_mlp_cv00 \
           flyvis_tiny_eed_cv00 \
           flyvis_tiny_stimulus_ctx5_cv00; do
    python GNN_Main.py -o train_test_plot "$cfg"
done
```

On a Mac the default `device: auto` in each yaml prefers MPS over CPU. MPS support in PyTorch is
incomplete (no `float64`, no `torch.compile`) and may fail mid-run with errors like
`<op> not implemented for MPS` or `Cannot reuse outer quote character in f-strings`. If you hit one
of those, fall back to CPU by switching `device: auto` → `device: cpu` in each smoke-test config:

```bash
sed -i.bak 's/^  device: auto$/  device: cpu/' config/fly/flyvis_tiny_*_cv00.yaml
```

You may also need to set `torch_compile: false` in the same configs if compilation fails on your
platform.
