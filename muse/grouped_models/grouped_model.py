import torch
from typing import List, Union, Iterable

from muse.experiments import logger
from muse.models.model import Model
from attrdict import AttrDict as d
from attrdict.utils import get_with_default


class GroupedModel(Model, Iterable):
    """Abstract base class for an grouping of models.

    NOTE: this is basically an aggregation of models in some standard way.
    This does NOT implement training schedules or individual optimizer groups

    GroupedModel consist of...
    - models: (Model) these are all the models we run forward and/or backward on
    - TODO metrics: (Metric) these are all the metrics we might need to compute for this algorithm
        - loss_metrics: specific to computing loss
        - log_metrics: things that get logged
    - parameter group access for combinations of model parameters specific to algorithm

    Added (dynamic) params for each model
    (1) {name}_no_grad: Sets requires_grad on the model to False (will not train)
        Loss definition should also reference this to save computation time.
    (2) {name}_file: Sets a file to load when restore is called

    """

    # model names that we require
    required_models = []

    def _init_params_to_attrs(self, params: d):
        super()._init_params_to_attrs(params)
        self._parse_models(params)

        # model names that get used online
        self.forward_models = get_with_default(params, "forward_models", list(self.required_models))

    def _parse_models(self, params):
        # all models passed in as AttrDict should be specified in model_order as string names
        self._model_order = get_with_default(params, 'model_order', [])
        logger.debug("Model names: " + str(self._model_order))

        # check for missing required models.
        if len(self.required_models) > 0:
            assert set(self.required_models).issubset(self._model_order), \
                f"{self.__class__} is missing models {set(self.required_models).difference(self._model_order)}"

        # the remaining models that were passed in are "optional"
        self.optional_models = [s for s in self._model_order if s not in self.required_models]
        if len(self.optional_models) > 0:
            logger.debug("Optional model names:", self.optional_models)

        self.model_params = params > self._model_order

    def _init_setup(self):
        super()._init_setup()

        # instantiate all models in order
        for model_name in self._model_order:
            # can be either (Model instance or params(cls: subclass_of_Model))
            model = self.model_params[model_name]
            if not isinstance(model, Model):
                assert isinstance(model, d), \
                    f"All models must be instantiable, but \'{model_name}\' was not."
                cls = model.cls
                logger.debug(
                    f"Instantiating \"{model_name}\" with class {cls}")
                model = cls(model, self.env_spec, self._dataset_train)
                assert isinstance(model, Model), f"{model_name} is {type(model)} but expected a subclass of Model"

            # assign them locally for parameter linking
            setattr(self, model_name, model.to(device=self.device))

    def __getitem__(self, model_name: str):
        # nested lookup of grouped model
        if '/' in model_name:
            splits = model_name.split('/')
            local, model_name = splits[0], '/'.join(splits[1:])
            local = getattr(self, local)
            assert isinstance(local, GroupedModel)
            return local[model_name]
        else:
            return getattr(self, model_name)

    def __iter__(self):
        for m in self._model_order:
            yield self[m]

    def load_statistics(self, dd=None):
        # nested lookup of statistics
        dd = super(GroupedModel, self).load_statistics(dd)
        for m in self:
            dd = m.load_statistics(dd)
        return dd

    def normalize_by_statistics(self, inputs: d, names, shared_dtype=None, check_finite=True, inverse=False,
                                shift_mean=True, normalize_sigma=None, required=True):
        # normalized names at this level
        inputs, all_norm_names = super(GroupedModel, self).normalize_by_statistics(inputs, names,
                                                                                   shared_dtype=shared_dtype,
                                                                                   check_finite=check_finite,
                                                                                   inverse=inverse,
                                                                                   shift_mean=shift_mean,
                                                                                   normalize_sigma=normalize_sigma,
                                                                                   required=False)

        # remove the ones that were normalized
        names = [n for n in names if n not in all_norm_names]

        # lookup in children
        for model in self:
            if len(names) == 0:
                break
            # find names that
            inputs, new_norm_names = model.normalize_by_statistics(inputs, names,
                                                                   shared_dtype=shared_dtype,
                                                                   check_finite=check_finite, inverse=inverse,
                                                                   shift_mean=shift_mean,
                                                                   normalize_sigma=normalize_sigma,
                                                                   required=False)
            names = [n for n in names if n not in new_norm_names]
            all_norm_names += new_norm_names

        if required and len(names) > 0:
            raise ValueError(f"Missing required names to normalize in tree!: {names}")

        if required:
            return inputs
        else:
            return inputs, all_norm_names

    def restore_from_checkpoints(self, model_names, checkpoints, strict=False):
        """
        Normal restoring behavior loads all parameters for the model. we might want to split loading between files

        Parameters
        ----------
        model_names: which models to load
        checkpoints: which checkpoint to use for each model
        strict: require all exact key match between each checkpoint and sub model

        Returns
        -------

        """

        assert len(model_names) == len(checkpoints), "Same number of files must be passed in as models"
        models = [self[n] for n in model_names]
        for m, chkpt in zip(models, checkpoints):
            m.restore_from_checkpoint(chkpt, strict=strict)

    def restore_from_files(self, model_names: List[str], file_names: Union[str, List[str]],
                           prefixes: Union[str, List[str]] = None):
        """
        Restore nested
        Parameters
        ----------
        model_names: Which models to load
        file_names: Which file to load for each model, or one file
        prefixes: Which prefix to use within each file, or one prefix (default no prefix)

        Returns
        -------

        """
        if isinstance(file_names, str):
            file_names = [file_names] * len(model_names)

        if prefixes is None or isinstance(prefixes, str):
            prefixes = [prefixes] * len(model_names)

        assert len(file_names) == len(prefixes) == len(model_names), "Specify same number of inputs for each names!"

        # load each file
        chkpts = [torch.load(f, map_location=self.device) for f in file_names]
        # prefix within checkpoint
        chkpts = [(c if p is None else c[p]) for p, c in zip(prefixes, chkpts)]
        self.restore_from_checkpoints(model_names, chkpts)

    def pretrain(self, datasets_holdout=None):
        """ Pretrain actions """
        for m in self:
            m.pretrain(datasets_holdout=datasets_holdout)

    @staticmethod
    def get_kwargs(name, kwargs) -> dict:
        if f'{name}_kwargs' in kwargs:
            return kwargs[f'{name}_kwargs']
        else:
            return {}

    def get_default_mem_policy_forward_fn(self, *args, separate_fns=False, **kwargs):
        """ Function that policy will use to run model forward (see MemoryPolicy)

        Default behavior is to call all sub models, with their pre actions, then locally model forward, then post.

        A different sub-dict will be used in memory for each model that specifies a forward_fn.

        Parameters
        ----------
        args
        separate_fns: bool
        kwargs

        Returns
        -------

        """
        names, pre_fns, post_fns = [], [], []
        for n in self.forward_models:
            m = self[n]
            if hasattr(m, "get_default_mem_policy_forward_fn"):
                pre_fn, post_fn, _ = m.get_default_mem_policy_forward_fn(*args, separate_fns=True, **kwargs)
                pre_fns.append(pre_fn)
                post_fns.append(post_fn)
                names.append(n)

        def agg_pre_fn(model, obs, goal, memory, **inner_kwargs):
            for name, pre_forward_fn in zip(names, pre_fns):
                if name not in memory:
                    memory[name] = d()
                if f"{name}_kwargs" not in inner_kwargs:
                    inner_kwargs[f"{name}_kwargs"] = {}
                obs, goal, memory[name], inner_kwargs[f"{name}_kwargs"] = \
                    pre_forward_fn(model[name], obs, goal, memory[name], **inner_kwargs[f"{name}_kwargs"])
            return obs, goal, memory, inner_kwargs

        def agg_post_fn(model, out, obs, goal, memory):
            for name, post_forward_fn in zip(names, post_fns):
                out[name] = post_forward_fn(model[name], out[name], obs, goal, memory[name])
                # move everything to top level as well.
                out.combine(out[name])
            return out

        def forward_fn(model, obs, goal, memory, **inner_kwargs):
            obs, goal, memory, inner_kwargs = agg_pre_fn(model, obs, goal, memory, **inner_kwargs)

            # local count tracker
            if 'count' not in memory:
                memory.count = 0
                memory.increment_count_locally = True

            out = model.forward(obs, goal, **inner_kwargs)

            if 'increment_count_locally' in memory and memory['increment_count_locally']:
                memory.count += 1

            return agg_post_fn(model, out, obs, goal, memory)

        if separate_fns:
            return agg_pre_fn, agg_post_fn, forward_fn
        else:
            return forward_fn
