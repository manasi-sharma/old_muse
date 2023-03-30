# Muse

This folder contains all the source code for the package `muse`. Specifically the following modules:

- **Datasets**: read / write access to data, and data statistics.
- **Models**: These hold all _parameters_, e.g, neural network weights. They also define `forward()` and `loss()` functions for training.
- **Policies**: These operate on a model and inputs to produce an "action" to be applied in the environment.
- **Environments**: Step-able, reset-able, gym-like format, where step(action) uses the output of the policy.
- **Metrics**: Things that will take inputs/outputs, and compute some tensor, e.g. representing a "loss", supporting group_by operations.
- **Rewards**: Similar to metrics, but compute a single value representing a reward potentially using the model too, and this can object be reset. 
- **Trainers**: Compose datasets, models, policies, environments, metrics, and rewards (optional) together into a training algorithm. Native wandb or tensorboard writing.

---
## muse.datasets
Datasets implement various storage and reading mechanisms. 
`muse.datasets.NpDataset` is the one used for most things. `muse.datasets.Hdf5Dataset` is implemented and tested but should rarely be used, since it is often slower (without in-memory caching).

A `Dataset` defines:
- `get_batch(indices, ...)`: Gets a batch of data as two AttrDicts: inputs, outputs.
- `get_episode(i, ...)`: Gets a full episode of data as either two AttrDicts (inputs, outputs) or one (datadict).
- `add_episode(inputs, outputs, ...)`: Adds data as an episode of inputs and outputs (both are AttrDicts).
- `add(input, output, ...)`: Adds a single input / output to the dataset (still AttrDicts).
- `__len__`: Size of the dataset.
- `__iter__`: Iterable for the dataset, since this is a sub-class of `torch.dataset.IterableDataset`
- `get_sampler(self, extra_datasets=())`: Return a default sampler to use with this dataset.
- `get_num_episodes()`: Number of episodes in the dataset

### muse.datasets.samplers
Samplers (`muse.datasets.samplers.sampler.Sampler`) are responsible for getting the indices used for `get_batch`, however each dataset defines some default sampling behavior if `indices=None`. The sampler will only be created in the trainer (see [trainers](##muse.trainers)), potentially using multiple datasets.

A sampler is instantiated with a single or list of datasets, and has the following methods:
- `get_indices()`: Gets the indices of the dataset's batch size


---
## muse.envs
Environments are very similar to that in OpenAI's `gym` format. They use a shared asset folder `assets/` which should have been downloaded in installation.
These environments implement, for example:
- `step(action: AttrDict, ...) -> obs, goal, done`: Similar to gym, but everything is an AttrDict, except done which is a 1 element bool array.
- `reset(presets: AttrDict) -> obs, goal`: Like gym, but enables presets to constrain the resetting.
- `default_params: AttrDict`: This is a set of default params that you might use to instantiate the environment (`make` uses this)
- `get_default_env_spec_params(env_params: AttrDict) -> AttrDict`: This returns a default env spec parameters for a given environment, including the spec class as `cls=...` (`make` uses this)

Some Implemented Environments:
- `muse.envs.simple.gym.GymEnv`: a gym environment wrapper for all gymnasium environments
- `muse.envs.simple.point_mass_env.PointMassEnv`: a 2D point mass example where you are trying to reach a moving (or static) target.
- `muse.envs.robosuite.robosuite_env.RobosuiteEnv`: robosuite environments. currently this uses a robomimic wrapper, so both robomimic and robosuite should be installed (TODO fix)
- `muse.envs.pymunk.<>`: 2D block manipulation environments in pymunk, including maze navigation, 2D stacking, and much more.
- `muse.envs.polymetis.polymetis_panda_env.PolymetisPandaEnv`: a polymetis wrapper for robot control
- `muse.envs.bullet_envs.block3d.<>`: 3D block / mug manipulation environments implemented in pybullet.

For all environments that implement the above functions, you can manually create them or instantiate them through `muse.envs.env.make`, for example point mass:

`env = make(PointMassEnv, AttrDict(render=True))`

You can provide additional parameters in the call to `make` to override `Env.default_params`, for example `render=True` here.
This function simply creates the parameters for the environment, uses that to instantiate an env_spec using `get_default_env_spec_params`, then instantiates the env_spec and the env.
You can access the env_spec via `env.env_spec`.

---
## muse.models
Models are an extension of `torch.nn.Module`, but with native support for AttrDicts, input / output normalization, pre/postprocessing, and much more.

We adopt the practice of first "declaring" architectures before instantiating them. We do this using `LayerParams` and `SequentialParams`, which accept individual layer arguments.
See `build_mlp_param_list()` in `muse.utils.param_utils` for an example of what this looks like. Model configurations will usually adopt a similar syntax.

See `RnnModel` for a recurrent structure model, which we use for most experiments.

`Model` defines the following:
- `pretrain(datasets_holdout=None)`: Any actions to run when the model is created, e.g. preprocessing some inputs and adding them to the dataset(s). See [trainers](##muse.trainers) for an example of when pretraining can be performed.
- `load_statistics(dd)`: Compute statistics using the linked training dataset (`self._dataset_train`), and save it internally to the torch means / stds. See [trainers](##muse.trainers) for an example of when statistics are loaded.
- `normalize_by_statistics(inputs, names, ...)`: Normalize names in AttrDict inputs using computed dataset statistics. Models should be responsible for normalization to avoid bugs later on. Supports unnormalization using `inverse=True`.
- `forward(inputs, ...)`: Default behavior just does optional normalization. Override in subclasses.
- `loss(inputs, outputs, ...)`: Compute some loss that can be used by an external optimizer. The default behavior here is to call `self._loss_fn` passed in params, or to use any internally defined metrics.
- `train_step(inputs, outputs, ...)`: A full training step, including optimization. This is useful for models that have complicated optimization processes, or interleaved optimization and forward calls. If you override this, set the static parameter `implements_train_step=True`.
- `restore_from_checkpoint(chkpt)`: Load the model from a checkpoint. 

You can create an empty model by calling using `params=AttrDict(ignore_inputs=True)`.

---
## muse.grouped_models
GroupedModels enable multiple sub-modules (inheriting from `Model`) to be declared, checked, and debugged more easily.

`GroupedModel` declares the following fields:
- `required_models` (static): predefined list of model names in this field. During instantiation, these will be checked for.
- `model_order` (params): the order of models internally, used for iterating and instantiating. Must include all the required models.
- `<model_name>` (params): For each model, this should point to its cls and parameters, e.g. using a standard config group

`GroupedModel` instances are iterable, and support item indexing (the order is determined by `model_order`). Additionally each model can be accessed via dot notation `self.<model_name>`

`GroupedModel` automatically handles the normalization, statistics loading, and pretraining for all internal models.

It also declares a `get_default_mem_policy_forward_fn` function that can be used with a policy, to keep track of things like hidden states, as needed within each Model.

NOTE: the `get_kwargs(model_name, kwargs)` function parses nested dictionary kwargs for a given model name. It is recommended you use this to parse out kwargs that are meant for each model in `self.forward` before calling `self.<model_name>.forward`

---
## muse.policies
Policies _use_ models, but do not contain them. Think of policies as a functional abstraction that takes in `(model, obs, goal)` and outputs `(action)`. 
See `MemoryPolicy` for an example of how to keep track of a running memory (e.g., for `RNNModels`). 
The policy config will be responsible for providing the right `policy_model_forward_fn` for these more generic policies.

---
## muse.trainers
Trainers compose all the above modules into a training algorithm, involving optimizers, model losses, saving checkpoints, and optionally also involving stepping some environment.
The most used classes are `muse.trainers.Trainer` (deprecated) and `muse.trainers.goal_trainer.GoalTrainer`. 

### GoalTrainer
GoalTrainer defines a hierarchical evaluation paradigm, supporting:
- `model`: the model to train.
- `datasets_train`: list of datasets for train_step
- `datasets_holdout`: list of datasets for holdout_step
- `env_train`: source environment, used for rollouts.
- `policy`: policy to determine the action for the source env
- `goal_policy`: hierarchical policy to determine the goal for the source env
- `env_holdout`: target environment (can be None)
- `policy_holdout` [optional]
- `goal_policy_holdout` [optional]
- `reward` [optional]: a reward module that will be used to compute rewards
- `writer` [optional]: a writer instance to use for logging, if None will create a tensorboard writer.
- `optimizer` [optional]: optimizer to use, see [optimizers](###muse.trainers.optimizers)
- `sampler` [optional]: sampler parameters (AttrDict), see [samplers](####muse.datasets.samplers). This should have a cls field. If provided, this cls will be used to create samplers for each dataset, using all the provided datasets. See below for how each sampler is instantiated.

**Sampling**: By default, `GoalTrainer` creates samplers and samples from datasets in the following way:
1. On init, create a sampler instance using either provided `sampler` params or using each dataset's `get_sampler()` function.
    1. If `sampler` params are provided, each dataset `ds` in `datasets_train` is created with the given `sampler.cls` using the rest of the params, with all the datasets in `datasets_train`, with `ds` moved to the front. Likewise for `datasets_holdout`.
    2. Otherwise, each dataset `ds` gets a sampler `ds.get_sampler()`, which will not see the other provided datasets.
2. On train_step, for each paired dataset and sampler, sample using the indices returned frm `sampler.get_indices()` for each dataset, calling `dataset.get_batch(indices=sampler.get_indices())`

**Model loading**: In order to load checkpoints (after pausing training for example), GoalTrainer looks for the `params.checkpoint_model_file` in the current experiment folder. If provided and found, it will load the model from this. If not, we do not error, but rather we will check for the `params.pretrained_model_file` if provided, and if `params.pretrained_model_file` is provided but does not exist (global path), an exception will be raised. You can sub-class and override `_restore_model_from_pretrained_checkpoint(checkpoint)` to implement a different pretrained model loading behavior (e.g. only loading part of the network).


### muse.trainers.optimizers
Optimizers are used within trainers to control gradient steps for networks. An optimizer can also include a scheduler. `optimizer.SingleOptimizer` is the one that should be used in most cases, but there are times when `optimizer.MultiOptimizer` makes sense, for example see `sac_optimizer.SACOptimizer`.

---
## Case Study: Training BC-RNN on Real World Data

Next we will go through an example of how to collect demonstration data, and then train BC-RNN. We will use making coffee as the task.

### Collecting Data

To collect data, we will use `scripts/collect.py`, which runs data collection using a policy, env, and dataset for storage.
[TODO]

### Training BC-RNN from vision

Let's move the given dataset for the coffee task to `data/real/make-coffee_eimgs_100ep.npz`. 
For training BC-RNN with vision, we can use the declarative config: `cfgs/exp_hvs/coffee/vis_bc_rnn.py` and `scripts/goal_train.py`.
The command is as follows

`python scripts/goal_train.py --no_env cfgs/exp_hvs/coffee/vis_bc_rnn.py --dataset make-coffee_eimgs_100ep`

We specify `--no_env` to prevent the creation of the polymetis panda environment, since we are purely training offline. 
We also override the dataset field to use the correct dataset (if you look at the config, you can see how the global dataset parameter gets used in the `dataset_train` group).

This command will train on a cuda-enabled device for 600000 steps by default, saving checkpoints every 20000 steps to `experiments/<exp_name>/models/`.

### Evaluating BC-RNN interactively

With the model trained in `experiments/<exp_name>/models/` (e.g., `model.pt`), we can evaluate on robot hardware using the interactive collection script.

`python scripts/interactive_collect.py --max_steps 10000 --model_file model.pt --save_file robot_eval.npz cfgs/exp_hvs/coffee/vis_bc_rnn.py --dataset make-coffee_eimgs_100ep`

We once again specify the same dataset to the config since that will load the correct experiment name. This will launch a pygame based interface for controlling execution.

For the polymetis panda environment we use, you can specify the correct camera id by adding `%env_train %%camera --cv_cam_id <>` at the end of the command above, or by manually changing the environment config in `cfgs/env/polymetis_panda.py`.