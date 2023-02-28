# Muse: Robotics real/sim research code

A flexible and modular code base for machine learning, with a focus in robotics.

## Installation

The code in this repo is mostly python, and is easily pip installable.

1. Create conda environment.
2. Install dependencies (a partial list is found under `requirements.txt`)
3. `pip install -e <MUSE_ROOT>` to setup the package, which installs three packages
   1. `muse`: includes all source code
   2. `configs`: Configuration module (`configs/README.md`).
   3. `cfgs`: static configuration files for quick use.
4. Under root, download and unzip assets using: `gdown https://drive.google.com/uc?id=<FILE ON DRIVE>`
   1. latest: June 2022: `FILE_ID = 1TR0ph1uUmtWFYJr8B0yo1hASjj0cP8fc`
5. Create several additional directories in the root:
   1. `data/`: This will store all the data
   2. `experiments/`: This is where runs will log and write models to.
   3. `plots/`: (optional)
   4. `videos/`: (optional)

All set up!

## Design Overview
Here we give an overview of the design philosophy for this repository. [TODO] Submodules will have further explanations that go into greater detail.

The components are:
- **Datasets**: read / write access to data, and data statistics.
- **Models**: These hold all _parameters_, e.g, neural network weights. They also define `forward()` and `loss()` functions for training.
- **Policies**: These operate on a model and inputs to produce an "action" to be applied in the environment.
- **Environments**: Step-able, reset-able, gym-like format, where step(action) uses the output of the policy.
- **Metrics**: Things that will take inputs/outputs, and compute some tensor, e.g. representing a "loss", supporting group_by operations.
- **Rewards**: Similar to metrics, but compute a single value representing a reward potentially using the model too, and this can object be reset. 
- **Trainers**: Compose datasets, models, policies, environments, metrics, and rewards (optional) together into a training algorithm. Native tensorboard writing.

### AttrDict
We use the `attr-dicts` package, which implements nested dictionaries that are easy to access, filter, combine, and write to. 
Most classes accept AttrDict's rather than individual parameters for initialization, and often for methods as well.
See here for greater detail about working with AttrDicts: https://github.com/Stanford-ILIAD/attrdict

### muse.datasets
Datasets implement various storage and reading mechanisms. 
`muse.datasets.NpDataset` is the one used for most things, and other types of datasets are built on top of this (e.g., see `muse.datasets.TorchDataset`).
Some datasets haven't been implemented (like `muse.datasets.Hdf5Dataset`).

Some methods of note:
- `get_batch(indices, ...)`: Gets a batch of data as two AttrDicts: inputs, outputs.
- `add_episode(inputs, outputs, ...)`: Adds data as an episode of inputs and outputs (both are AttrDicts).
- `add(input, output, ...)`: Adds a single input / output to the dataset (still AttrDicts).
- `__len__`: Size of the dataset.

### muse.envs
Environments are very similar to that in OpenAI's `gym` format. They use a shared asset folder `muse/envs/assets/` which should have been downloaded in installation.
These environments implement, for example:
- `step(action: AttrDict, ...) -> obs, goal, done`: Similar to gym, but everything is an AttrDict, except done which is a 1 element bool array.
- `reset(presets: AttrDict) -> obs, goal`: Like gym, but enables presets to constrain the resetting.

Environments used for PLATO are described under the Play-Specific Environments section below.

### muse.models
Models are an extension of `torch.nn.Module`, but with native support for AttrDicts, input / output normalization, pre/postprocessing, and much more.

We adopt the practice of first "declaring" architectures before instantiating them. We do this using `LayerParams` and `SequentialParams`, which accept individual layer arguments.
See `build_mlp_param_list()` in `muse.utils.param_utils` for an example of what this looks like. Model configurations will usually adopt a similar syntax.

See `RnnModel` for a recurrent structure model, which we use for most experiments.

### muse.grouped_models
GroupedModels enable multiple sub-modules (inheriting from `Model`) to be declared, checked, and debugged more easily.

### muse.policies
Policies _use_ models, but do not contain them. Think of policies as a functional abstraction that takes in `(model, obs, goal)` and outputs `(action)`. 
See `MemoryPolicy` for an example of how to keep track of a running memory (e.g., for `RNNModels`). 
The policy config will be responsible for providing the right `policy_model_forward_fn` for these more generic policies.

### muse.trainers
Trainers compose all the above modules into a training algorithm, involving optimizers, model losses, saving checkpoints, and optionally also involving stepping some environment.
The most used classes are `muse.trainers.Trainer` and `muse.sandbox.new_trainer.goal_trainer.GoalTrainer`.

---