# Configurations

In `muse` we use a modular, tree-like configuration specification which is written entirely in python, and supports command line overrides.
This allows configs to contain python expressions, objects, and functions.

Config files specify experimental parameters, and each script will use different groups. For example, `scripts/train.py` requires the following groups:
- env_spec
- env_train
- dataset_train
- dataset_holdout
- model
- policy
- trainer

These groups might also have their own subgroups, and the `configs` module will resolve this structure into a tree, where each node is a `ConfigNode` object that has local parameters and children that are `ConfigNode` objects.

To create these configs, the `configs` module here supports either *declarative* or *programmatic* configuration files specified in python. 
Declarative configs declare parameters initially and somewhat statically.
Programmatic configs declare a parser to load arguments, and then later compute the parameters based on the arguments and other values in the tree.
Both types of config can be used in place of any group or subgroup, which makes this a very modular way of specifying configurations.

The `configs` module also natively supports command line overriding of parameters, of the form:

`python <script> <script_args> <config_file> <args> %group_name <args> %%subgroup_name <args> %next_group_name <args> ...`

This makes it very easy to update parameters for experiment runs without creating a new configuration file.


## Declarative Configs

Declarative configs declare all the parameters and groups up front in the form of an `AttrDict`, for example:
```python
from attrdict import AttrDict as d
export = d(
   global_arg1=1,
   ...,
   group1=d(
      cls=...,
      arg1=True,
      ...,
      subgroup1_1=d(
          cls=...,
          subarg1=...,
          ...,
      )
   ),
   group2=d(
      cls=...,
      arg2_1=10,
      ...
   ),
   ...,
)
```

The above declarative config will resolve into the following tree of ConfigNodes:
```
root:
    group1:
        subgroup1
    group2:
```

### Command Line

All `float`, `int`, `bool`, and `string` args in this config will automatically be override-able from the command line. 
For `bool` args specifically, if the value is True, the argument name `arg_name` will be automatically changed to `no-{arg_name}` for parsing.
For example, I might run from command line:

`python <some_script> <config_file_above> --global_arg1 2 %group1 --no-arg1 %group2 --arg2_1 11`

This will update the params to be:
```python
from attrdict import AttrDict as d
export = d(
   global_arg1=2,
   ...,
   group1=d(
      cls=...,
      arg1=False,
      ...,
      subgroup1_1=d(
          cls=...,
          subarg1=...,
          ...,
      )
   ),
   group2=d(
      cls=...,
      arg2_1=11,
      ...
   ),
   ...,
)
```

Because this is python, we can easily import 

### Fields

We also define a `Field` attribute which allows for conditional or dependent parameters using other tree elements. 
A `Field` is initialized with a key to link the output to, and a function which takes the final processed value for the key and arbitrarily maps it.
Currently fields cannot link to keys that also have Field values (no recursive Fields). 
Keys will first be looked up locally in the config node (use a relative path), and then in the entire existing tree structure (global path)

For example let us assume we want to declare the batch size and horizon length globally for both the dataset and the model to see.
Our config might take the following form:
```python
from attrdict import AttrDict as d
from configs.fields import Field as F
from muse.models.basic_model import BasicModel
from muse.datasets.np_dataset import NpDataset
export = d(
    half=5,
    batch_size=100,
    horizon=10,
    model=d(
        cls=BasicModel,
        horizon=F('horizon'),
        full=F('half', lambda x: 2*x),
    ),
    dataset_train=d(
        cls=NpDataset,
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        ...
    ),
)
```

### Experiment Names

The experiment name is also computed in a recursive fashion, in depth first order of the groups and subgroups. 
Each group can have an `exp_name` argument that can be either:
1. A string, with support for substitution with string-convertable local arguments in brackets and support for boolean conditional strings.
   1. For example, `exp_name="dir/b{batch_size}_h{horizon}{?augment:_aug}"` for batch size 100, horizon 10, and augment=True would parse into `exp_name="dir/b100_h10_aug"`
   2. Boolean conditionals take the form `"{?bool_arg_name:true_string\[:false_string\]}"`
2. A function, which more generally takes in the parameters at this level and returns a string.

Importantly this defines the local name of the experiment, where subgroups might add on to this in depth first order.

For example consider the following experiment from earlier but with exp_names: 
```python
from attrdict import AttrDict as d
from configs.fields import Field as F
from muse.models.basic_model import BasicModel
from muse.datasets.np_dataset import NpDataset
export = d(
    exp_name="directory/b{batch_size}_h{horizon}{?augment:_aug}",
    augment=False,
    half=5,
    batch_size=100,
    horizon=10,
    model=d(
        exp_name="basic_model_f{full}",
        cls=BasicModel,
        horizon=F('horizon'),
        full=F('half', lambda x: 2*x),
    ),
    dataset_train=d(
        cls=NpDataset,
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        ...
    ),
)
```
The final experiment name here would be `exp_name="directory/b100_h10_basic_model_f10"`. 
Most of the time, this functional string representation is all you need, but sometimes you might need more general experiment names, so we also support functions in place of a string.


Examples of declarative configs can be found in `cfgs/...`.

## Programmatic Configs

Programmatic Configs are defined as subclasses of `ConfigNode`, which should override (call super first):
1. `declare_arguments(self, parser, defaults, cls_skip_args)`: Take a parser, default values, and class arguments to skip, and declare new arguments to the parser.
2. `process_params(self, namespace, node_params, global_params=None)`: Take the parsed local arguments, default node parameters, and global parameters (tree so far), and update node parameters based on the arguments.

You can also mix programmatic and declarative configs using the following optional default values to override:
- `default_cls`: default class for the config node (same as default value for `cls` in the declarative format)
- `default_local_exp_name`: default local experiment name chunk (same as in declarative, string or function)
- `default_cls_skip_args`: default arguments to skip from cls.predefined_arguments
- `default_params`: default parameters for the class to be instantiated with (same as the other parameters in a group)

Examples of programmatic configs can be found in `configs/...`.

[TODO more on programmatic configs]