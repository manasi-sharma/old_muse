# Muse: Robotics real/sim research code

A flexible and modular code base for machine learning, with a focus in robotics.

## Installation

The code in this repo is mostly python, and is easily pip installable.

1. Create conda environment (python 3.10)
2. Install dependencies (a partial list is found under `requirements.txt`)
3. `pip install -e <MUSE_ROOT>` to setup the package, which installs three packages
   1. `muse`: includes all source code (`muse/README.md`)
   2. `configs`: Configuration module and programmatic config files (`configs/README.md`).
   3. `cfgs`: declarative config files for quick use.
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

The components of `muse` are:
- **Datasets**: read / write access to data, and data statistics.
- **Models**: These hold all _parameters_, e.g, neural network weights. They also define `forward()` and `loss()` functions for training.
- **Policies**: These operate on a model and inputs to produce an "action" to be applied in the environment.
- **Environments**: Step-able, reset-able, gym-like format, where step(action) uses the output of the policy.
- **Metrics**: Things that will take inputs/outputs, and compute some tensor, e.g. representing a "loss", supporting group_by operations.
- **Rewards**: Similar to metrics, but compute a single value representing a reward potentially using the model too, and this can object be reset. 
- **Trainers**: Compose datasets, models, policies, environments, metrics, and rewards (optional) together into a training algorithm. Native wandb or tensorboard writing.

Further details and an example of collecting data and training BC-RNN from vision is outlined in `muse/README.md`.

### AttrDict
We use the `attr-dicts` package, which implements nested dictionaries that are easy to access, filter, combine, and write to. 
Most classes accept AttrDict's rather than individual parameters for initialization, and often for methods as well.
See here for greater detail about working with AttrDicts: https://github.com/Stanford-ILIAD/attrdict

### muse.datasets
Datasets implement various storage and reading mechanisms. 
`muse.datasets.NpDataset` is the one used for most things. `muse.datasets.Hdf5Dataset` is implemented and tested but should rarely be used, since it is often slower (without in-memory caching).

Some methods of note:
- `get_batch(indices, ...)`: Gets a batch of data as two AttrDicts: inputs, outputs.
- `get_episode(i, ...)`: Gets a full episode of data as either two AttrDicts (inputs, outputs) or one (datadict).
- `add_episode(inputs, outputs, ...)`: Adds data as an episode of inputs and outputs (both are AttrDicts).
- `add(input, output, ...)`: Adds a single input / output to the dataset (still AttrDicts).
- `__len__`: Size of the dataset.

### muse.envs
Environments are very similar to that in OpenAI's `gym` format. They use a shared asset folder `assets/` which should have been downloaded in installation.
These environments implement, for example:
- `step(action: AttrDict, ...) -> obs, goal, done`: Similar to gym, but everything is an AttrDict, except done which is a 1 element bool array.
- `reset(presets: AttrDict) -> obs, goal`: Like gym, but enables presets to constrain the resetting.
- `get_default_env_spec_params(env_params: AttrDict) -> AttrDict`: This returns a default env spec parameters for a given environment (including the class as `cls=...`)

Some Implemented Environments:
- `muse.envs.simple.gym.GymEnv`: a gym environment wrapper for all gymnasium environments
- `muse.envs.simple.point_mass_env.PointMassEnv`: a 2D point mass example where you are trying to reach a moving (or static) target.
- `muse.envs.robosuite.robosuite_env.RobosuiteEnv`: robosuite environments. currently this uses a robomimic wrapper, so both robomimic and robosuite should be installed (TODO fix)
- `muse.envs.pymunk.<>`: 2D block manipulation environments in pymunk, including maze navigation, 2D stacking, and much more.
- `muse.envs.polymetis.polymetis_panda_env.PolymetisPandaEnv`: a polymetis wrapper for robot control
- `muse.envs.bullet_envs.block3d.<>`: 3D block / mug manipulation environments implemented in pybullet.

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

### configs
In order to set up experiments, we have a python-based configuration format, which resembles more common yaml-based configurations but allows for pythonic language in configurations, including functions, expressions, and dependent parameters.
For more information on how to set up and work with configs, see `configs/README.md`.

---

## Scripts

Scripts in `muse` follow a similar format, here are the basic ones that one might regularly use:
- `scripts/train.py`: \[legacy\] Train a model (used primarily for offline training without environment interaction)
- `scripts/goal_train.py`: Train a model with a policy and a goal policy (used primarily when online rollouts are needed to evaluate different checkpoints during training)
- `scripts/resolve.py`: Resolve the configuration module, and print out the resulting parameters
- `scripts/tests/load_batches.py`: Load a dataset module and load batches while timing things, as a way to debug dataset loading.
- `scripts/eval.py`: Evaluate a model and policy in an environment.
- `scripts/collect.py`: Evaluate a model and policy in an environment, and also save it to dataset.
- `scripts/interactive_collect.py`: Evaluate a model and policy in an environment, with a pygame window that lets you control which episode to keep, when to reset, etc

Each of these starts with a similar header:
```python
from configs.helpers import load_base_config, get_script_parser
parser = get_script_parser()
parser.add_argument('config', type=str, help="common params for all modules.")
# <other parser argument declarations for the given script>
# ...

# parse the local command line arguments
local_args, unknown = parser.parse_known_args()

# load the config tree with command line arguments that were not locally recognized
params, root = load_base_config(local_args.config, unknown)

# load the experiment name from the config root node.
exp_name = root.get_exp_name()
```

First we load local arguments for the script, then we load the config tree (see `configs/README.md`), and then we get an experiment name from the root config node.
Usually following this you can use the `params` AttrDict to instantiate any groups that are needed for the script. `params` will take the form of:
```python
from attrdict import AttrDict as d
export = d(
   global_arg1=1,
   ...,
   group1=d(
      cls=...,
      arg1=True,
      ...
   ),
   group2=d(
      cls=...,
      arg2_1=10,
      ...
   ),
   ...,
)
```
