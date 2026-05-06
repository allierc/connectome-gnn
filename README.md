# FlyVis-GNN

Graph neural networks recover interpretable circuit models from neural activity.

Synapse-level connectomes describe the structure of circuits, but not the electrical and chemical
dynamics. Conversely, large-scale recordings of neural activity capture these dynamics, but not the
circuit structure. We asked whether combining binary connectivity and recorded neural activity can
be used to infer mechanistic models of neural circuits. We trained a graph neural network model
(GNN) to forecast the activity of Drosophila visual system simulations. Trained on activity
trajectories in response to visual inputs, the model recovers effective connectivity weights, neuron
types, and nonlinear activation functions, even when 200% random connections are added to the
adjacency matrix. Moreover, it correctly predicts causal effects of connection removal,
demonstrating the ability to infer mechanistic dependencies directly from activity data. Our simple,
flexible, and interpretable method recovers both structure and dynamics from incomplete anatomical
reconstructions and activity.

The repository also includes an agentic workflow for hyper-parameter optimization in this ill-posed
inverse problem.

**Project page:**
[https://saalfeldlab.github.io/flyvis-gnn/](https://saalfeldlab.github.io/flyvis-gnn/)

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
