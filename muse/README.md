# Muse

This folder contains all the source code for the package `muse`. Specifically the following modules:

- **Datasets**: read / write access to data, and data statistics.
- **Models**: These hold all _parameters_, e.g, neural network weights. They also define `forward()` and `loss()` functions for training.
- **Policies**: These operate on a model and inputs to produce an "action" to be applied in the environment.
- **Environments**: Step-able, reset-able, gym-like format, where step(action) uses the output of the policy.
- **Metrics**: Things that will take inputs/outputs, and compute some tensor, e.g. representing a "loss", supporting group_by operations.
- **Rewards**: Similar to metrics, but compute a single value representing a reward potentially using the model too, and this can object be reset. 
- **Trainers**: Compose datasets, models, policies, environments, metrics, and rewards (optional) together into a training algorithm. Native wandb or tensorboard writing.

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

We once again specify the dataset to the config since that will load the correct experiment name. This will launch a pygame based interface for controlling execution.
