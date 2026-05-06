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

Note: we make use of torch.compile and use `reduce-overhead` which is a CUDA only feature. If
running on a make change the torch.compile incantations to use `default` mode. If you run into
difficulties with torch compile, just turn off compilation.

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

Trained GNN models and loss files are stored with [Git LFS](https://git-lfs.com/). After cloning,
pull the model files:

```bash
git lfs install
git lfs pull
```

Simulation data must be generated first (Notebook 00.py) before training or testing.

## Usage

```bash
# Single training run
python GNN_Main.py -o generate_train_test_plot flyvis_noise_05

```
