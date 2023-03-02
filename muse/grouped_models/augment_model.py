import numpy as np
import torch

from muse.experiments import logger
from muse.grouped_models.grouped_model import GroupedModel
from attrdict import AttrDict as d
from attrdict.utils import get_with_default
from muse.utils.general_utils import timeit, maybe_context, is_next_cycle
from muse.utils.torch_utils import torch_disable_grad


class AugmentModel(GroupedModel):
    """
    This model augments data and then passes it into another model.

    Augment: model that augments data
        - trains from ["block_augment_train_steps", "stop_augment_train_steps")
        - starts inference (updates model inputs) from ["block_augment_inf_steps", inf), modifying the inputs
            - runs once every "augment_every_n_train_steps"
            - for each run step, modifies "augment_frac" of batch size of the inputs.
            - If running inference, will copy over sub-key "augment_prefix" of augment's output as input to model.

    Model: main model
        - trains from ["block_model_train_steps", "stop_model_train_steps")
        - always runs inference

    Note: if train_step is called, at least one of model or augment must be in a training mode.

    """

    required_models = [
        "augment",
        "model",
    ]

    implements_train_step = True

    def _init_params_to_attrs(self, params: d):
        super(AugmentModel, self)._init_params_to_attrs(params)

        # how many steps to wait before beginning inference.
        self._block_augment_inference_steps = get_with_default(params, "block_augment_inf_steps", 0)

        self._augment_every_n_train_steps = get_with_default(params, "augment_every_n_train_steps", 1)
        self._augment_frac = get_with_default(params, "augment_frac", 1.)
        assert 0 < self._augment_frac <= 1., f"{self._augment_frac} must be between (0, 1]"

        # training range to allow model updates
        self._block_augment_train_steps = get_with_default(params, "block_augment_train_steps", 0)
        self._stop_augment_train_steps = get_with_default(params, "stop_augment_train_steps", np.inf)

        self._block_model_train_steps = get_with_default(params, "block_model_train_steps", 0)
        self._stop_model_train_steps = get_with_default(params, "stop_model_train_steps", np.inf)

        # arguments to pass to each model's .forward() at training and/or inference time.
        self._aug_train_kwargs = get_with_default(params, "augment_train_kwargs", {})  # train
        self._aug_train_inf_kwargs = get_with_default(params, "augment_train_inf_kwargs", {})  # train + inf
        self._aug_inf_kwargs = get_with_default(params, "augment_inf_kwargs", {})  # inf
        self._model_train_inf_kwargs = get_with_default(params, "model_train_inf_kwargs", {})  # train + inf
        self._model_inf_kwargs = get_with_default(params, "model_inf_kwargs", {})  # inf

        self._aug_prefix = params << "augment_prefix"

        logger.info(f"Augment: train {self._block_augment_train_steps} -> {self._stop_augment_train_steps}, "
                    f"block infer = {self._block_augment_inference_steps}, "
                    f"then every {self._augment_every_n_train_steps} steps, "
                    f"with {int(self._augment_frac * 100)}% of samples")

        logger.info(f"Model: train {self._block_model_train_steps} -> {self._stop_model_train_steps}")

    def forward(self, inputs, training=False, preproc=True, postproc=True,
                do_augment=False, do_augment_model=False, augment_nograd=False,
                do_model=True, model_disable_grad=False, augment_kwargs=None, augment_batch_size=None,
                **kwargs):
        """

        Will augment before model input

        :param inputs:
        :param training:
        :param preproc:
        :param postproc:
        :param do_augment: run augment model
        :param do_augment_model: copies over augmentation outputs to model inputs.
        :param augment_nograd: no gradient tracking for augmentation

        :param do_model: run model
        :param model_disable_grad: disable gradients for model
        :param augment_kwargs: for augment (in dictionary format)
        :param augment_batch_size: How many examples of input to augment
        :param kwargs: for model
        :return:
        """
        assert do_augment or do_model

        inputs = inputs.leaf_copy()
        model_outs = d()

        if do_augment:
            with timeit("forward/augment"):
                with maybe_context(augment_nograd, torch.no_grad):
                    augment_kwargs = augment_kwargs if isinstance(augment_kwargs, dict) else {}
                    aug_inputs = inputs

                    # sub selection of inputs will be augmented
                    if augment_batch_size is not None:
                        B = aug_inputs.get_one().shape[0]
                        assert augment_batch_size < B, \
                            f"Augment batch size {augment_batch_size} must be less than batch size {B}"
                        # this really doesn't need to be random
                        indices = torch.randperm(B, dtype=torch.long, device=self.device)[:augment_batch_size]
                        aug_inputs = aug_inputs.leaf_apply(lambda arr: arr[indices])

                    aug_out = self.augment(aug_inputs, training=training, **augment_kwargs)
                    model_outs['augment'] = aug_out
                    if do_augment_model:  # copy over to inputs after detaching
                        if self._aug_prefix is None:
                            new_inputs = aug_out.leaf_apply(lambda arr: arr.detach())
                        else:
                            new_inputs = (aug_out[self._aug_prefix]).leaf_apply(lambda arr: arr.detach())

                        if augment_batch_size is not None:
                            # replace values at indices with new ones
                            shared_keys = inputs.leaf_key_intersection(new_inputs)
                            assert len(shared_keys) > 0, [new_inputs.list_leaf_keys(), inputs.list_leaf_keys()]
                            # skip if not present (forgiving).
                            mapped_new_inputs = self.env_spec.map_to_types(new_inputs > shared_keys, skip_keys=True)
                            for key in shared_keys:
                                inputs[key][indices] = mapped_new_inputs[key]
                        else:
                            inputs = inputs & new_inputs  # override keys
        if do_model:
            with timeit("forward/model"):
                with maybe_context(model_disable_grad, torch_disable_grad, self._model):
                    out = self._model(inputs, training=training, preproc=preproc, postproc=postproc, **kwargs)
                    model_outs['model'] = out.leaf_copy()
                    if not training:
                        model_outs.combine(out)  # copy to top level as well

        return model_outs

    def train_step(self, inputs, outputs, i=0, writer=None, writer_prefix="", ret_dict=False, optimizer=None, meta=None,
                   dataset_idx=0, dataset=None, ti=0, **kwargs):
        """
        Augment always runs, since we are always training something.
        Model runs only if train_model = True
        """
        # will be passed as kwargs to model.forward in loss()
        # during training, do_model=do_augment=True, so this tells us what gets gradients (if any)
        train_model = self._block_model_train_steps <= ti < self._stop_model_train_steps
        train_augment = self._block_augment_train_steps <= ti < self._stop_augment_train_steps
        assert train_model or train_augment

        with timeit("train/loss"):
            aug_inf = is_next_cycle(ti, self._augment_every_n_train_steps) and ti >= self._block_augment_inference_steps

            # either inference only for a given model, or both train and inference
            model_kwargs = self._model_train_inf_kwargs
            aug_kwargs = self._aug_train_inf_kwargs if aug_inf else self._aug_train_kwargs
            # allows different mode when not training a given model and only running inference.
            if not train_model:
                model_kwargs = self._model_inf_kwargs
            if not train_augment:
                aug_kwargs = self._aug_inf_kwargs

            aug_batch_size = int(np.clip(np.round(self._augment_frac * dataset.batch_size), 1, dataset.batch_size)) \
                if self._augment_frac < 1. else None

            loss = self.loss(inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix, training=True,
                             ret_dict=ret_dict, meta=meta,
                             do_model=train_model, model_disable_grad=False,
                             do_augment=aug_inf or train_augment, do_augment_model=aug_inf,
                             augment_disable_grad=not train_augment, augment_kwargs=aug_kwargs,
                             augment_batch_size=aug_batch_size,
                             **model_kwargs)

            if writer is not None and aug_batch_size is not None:
                writer.add_scalar(writer_prefix + "augment/batch_size", aug_batch_size, i)

        with timeit('train/backprop'):
            optimizer.step(loss, inputs, outputs, dataset_idx, meta=meta, i=i, ti=ti, writer=writer,
                           writer_prefix=writer_prefix)

        return loss

    @property
    def augment(self):
        return self._augment

    @property
    def model(self):
        return self._model
