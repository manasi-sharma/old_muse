"""

"""
import abc
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from muse.datasets.np_dataset import NpDataset
from muse.models.basic_model import BasicModel
from muse.models.bc.gcbc import BaseGCBC, RNN_GCBC, TransformerGCBC
from muse.models.model import Model
from muse.models.rnn_model import RnnModel
from muse.utils.abstract import Argument, resolve_arguments
from muse.utils.param_utils import get_dist_cap, SequentialParams, build_mlp_param_list, LayerParams, \
    get_dist_out_size
from muse.utils.general_utils import timeit, is_next_cycle

from attrdict import AttrDict as d
from attrdict.utils import get_with_default, get_or_instantiate_cls

from muse.experiments import logger
from muse.utils.torch_utils import combine_after_dim


class DynamicActionBaseGCBC(BaseGCBC):
    """
    This will be the new base class. Adds in a mode prediction level

    Implements mode prediction, by reading from the inner model output by default
    """
    N_MODES = 2

    sparse_model_cls = None

    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument("sparse_normalize_states", action='store_true'),
        Argument("sparse_normalize_actions", action='store_true'),

        # loss things
        Argument('gamma', type=float, default=0.5, help="Weighting of action losses by mode per step"),
        Argument('mode_beta', type=float, default=0.01, help="Weighting of mode losses"),
        Argument('balance_mode_loss', action='store_true',
                 help="Balance action loss across modes, by batch freq"),
        Argument('balance_cross_entropy', action='store_true',
                 help="Balance mode XE loss by class freq in dataset"),
        Argument('label_smoothing', type=float, default=0., help="Apply label smoothing to mode XE"),
        Argument('use_smooth_mode', action='store_true'),

        # sparse action cap
        Argument("sparse_use_policy_dist", action="store_true"),
        Argument("sparse_policy_num_mix", type=int, default=1),
        Argument("sparse_use_policy_dist_mean", action="store_true"),
        Argument("sparse_policy_sig_min", type=float, default=1e-5),
        Argument("sparse_policy_sig_max", type=float, default=1e3),
        Argument("sparse_use_tanh_out", action="store_true"),

        Argument("use_mode_predictor", action="store_true"),
        # if not separate mode prediction, here are the args that define it.
        Argument("split_head_layers", type=int, default=2),
        Argument("inner_hidden_size", type=int, default=128),
        Argument("mode_head_size", type=int, default=128),
        Argument("action_head_size", type=int, default=128),
    ]

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        # set the loss, unless params specified something different.
        self._loss_fn = get_with_default(params, "loss_fn", self.default_loss_fn)

        self.sparse_in_names = list(self.novision_policy_in_names)
        self.sparse_action_names = params["sparse_action_names"]

        self.sparse_preproc_fn = get_with_default(params, "sparse_preproc_fn", self.inner_preproc_fn)
        self.sparse_postproc_fn = get_with_default(params, "sparse_postproc_fn", self.inner_postproc_fn)

        # other sparse model specific params.
        self._sparse_model_params_to_attrs(params)

        # optionally update sparse model inputs from vision
        if not self.vision_encoder_params.is_empty():
            self._update_sparse_policy_in_from_vision()

        # get the actual model, sub-classes will likely need to override this.
        self.sparse_model_params = self._get_sparse_model_shared_params() & self._get_sparse_model()

        self.mode_key = get_with_default(params, "mode_key", "mode")
        self.mode_logit_out_name = get_with_default(params, "mode_logit_out_name", "mode_logits")
        self.mode_prob_out_name = get_with_default(params, "mode_prob_out_name", "mode_probs")
        if self.use_mode_predictor:
            self._mode_predictor_model_params_to_attrs(params)
            self.mp_params = self._get_mode_predictor_model()

        # defining some default losses
        self._mode0_loss_fn: Callable = params["mode0_loss_fn"]
        self._mode1_loss_fn: Callable = params["mode1_loss_fn"]
        self.ac_loss_fns = [self._mode0_loss_fn, self._mode1_loss_fn]

        # loss weighting for sparse vs. dense
        self.loss_weights = [get_with_default(params, "mode0_weight", 1.), 1.]

        if self.sparse_normalize_states:
            logger.debug(f"[DynGCBC] Normalize input names: {self.state_names}")
        if self.sparse_normalize_actions:
            logger.debug(f"[DynGCBC] Normalize action names: {self.sparse_action_names}")

    def _update_sparse_policy_in_from_vision(self):
        # updates the sizes and names of sparse policy input with knowledge of using vision
        # self.state_names += [self.encoder_out_name]
        self.sparse_in_names += [self.encoder_out_name]
        self.sparse_in_size += self.encoder_out_size

    def _inner_model_params_to_attrs(self, params):
        if not self.use_mode_predictor:
            # change the policy raw out size to match the hidden size
            self.true_policy_raw_out_size = self.policy_raw_out_size
            self.policy_raw_out_size = self.inner_hidden_size

    def _sparse_model_params_to_attrs(self, params):
        # by default, sparse model takes in the same inputs as inner model.
        # NOTE this will include the vision inputs as well!
        self.sparse_raw_out_name = get_with_default(params, "sparse_raw_out_name", "sparse_policy_raw")

        self.sparse_sample_cat = get_with_default(params, "sparse_sample_cat", False)
        if self.sparse_sample_cat:
            logger.debug(
                "Sparse -- Will sample from categorical distribution (make sure you are using compatible action space)")

        self.sparse_normalize_inputs = []
        self.sparse_save_normalize_inputs = []

        # states will be normalized & stats saved
        if self.sparse_normalize_states:
            self.sparse_normalize_inputs += self.state_names
            self.sparse_save_normalize_inputs += self.state_names

        # action stats will be saved
        if self.sparse_normalize_actions:
            self.sparse_save_normalize_inputs += self.sparse_action_names

        self.sparse_in_size = self.env_spec.dim(self.sparse_in_names)
        self.sparse_policy_out_size = self.env_spec.dim(self.sparse_action_names)

        self.sparse_policy_raw_out_size = get_dist_out_size(self.sparse_policy_out_size,
                                                            prob=self.sparse_use_policy_dist,
                                                            num_mix=self.sparse_policy_num_mix)

    def _mode_predictor_model_params_to_attrs(self, params):
        raise NotImplementedError(f'Mode predictor not implemented in {type(self)}')

    def _get_sparse_model_shared_params(self):
        # some common parameters to use as a starting point
        return d(
            cls=self.sparse_model_cls,
            device=self.device,
            model_inputs=self.sparse_in_names,
            model_output=self.sparse_raw_out_name,
            preproc_fn=self.sparse_preproc_fn,
            postproc_fn=self.sparse_postproc_fn,
            normalize_inputs=len(self.sparse_normalize_inputs) > 0,
            normalization_inputs=self.sparse_normalize_inputs,
            save_normalization_inputs=self.sparse_save_normalize_inputs,
            default_normalize_sigma=self.default_normalize_sigma,
        )

    @abc.abstractmethod
    def _get_sparse_model(self) -> d:
        raise NotImplementedError

    def _get_mode_predictor_model(self) -> d:
        raise NotImplementedError(f'Mode predictor not implemented in {type(self)}')

    def _init_models(self):
        model_cstr = lambda cls, prms: cls(prms, self.env_spec, self._dataset_train)

        super()._init_models()

        if self.use_mode_predictor:
            # instantiate separate mode prediction network
            self.mode_predictor = get_or_instantiate_cls(self.mp_params, None, Model, constructor=model_cstr)

        else:
            # initializes the head(s) for the inner model
            self._init_inner_model_heads()

        # instantiate sparse model
        self.sparse_model = get_or_instantiate_cls(self.sparse_model_params, None, Model, constructor=model_cstr)

    def _init_inner_model_heads(self):
        self.mode_head = SequentialParams(build_mlp_param_list(self.policy_raw_out_size,
                                                               [self.mode_head_size] * self.split_head_layers +
                                                               [self.N_MODES])
                                          ).to_module_list(as_sequential=True).to(device=self.device)
        self.action_head = SequentialParams(build_mlp_param_list(self.policy_raw_out_size,
                                                                 [self.action_head_size] * self.split_head_layers +
                                                                 [self.true_policy_raw_out_size])
                                            ).to_module_list(as_sequential=True).to(device=self.device)

    def _init_caps(self):
        super()._init_caps()
        # cap for the sparse model
        self.sparse_model_cap = get_dist_cap(self.sparse_use_policy_dist, self.sparse_use_tanh_out,
                                             num_mix=self.sparse_policy_num_mix, sig_min=self.sparse_policy_sig_min,
                                             sig_max=self.sparse_policy_sig_max)
        self.sparse_model_cap = self.sparse_model_cap.to_module_list(as_sequential=True).to(self.device)

        class_weights = None
        if self.balance_cross_entropy and len(self._dataset_train) > 0:
            # TODO fix this.
            logger.debug(f"Loading mode class weights from dataset for training: {self._dataset_train}")
            assert isinstance(self._dataset_train, NpDataset)
            modes = self._dataset_train.get_datadict()[self._mode_key].reshape(-1)
            m_unique, m_count = np.unique(modes, return_counts=True)  # sorted
            assert len(m_unique) == 2, m_unique
            # weight according to likelihood
            class_weights = torch.tensor(1. / m_count, dtype=torch.float32, device=self.device)
            class_weights = class_weights / torch.min(class_weights)  # divide by the min
            logger.debug(f"Found modes: {m_unique}, with counts: {m_count}. using XE weights: {class_weights}")

        self.mode_loss_obj = CrossEntropyLoss(weight=class_weights, label_smoothing=self.label_smoothing)

    def _set_models(self):
        super()._set_models()
        self._models.append(self.sparse_model)
        if self.use_mode_predictor:
            self._models.append(self.mode_predictor)

    def pretrain(self, datasets_holdout=None):
        super().pretrain(datasets_holdout=None)
        self.sparse_model.pretrain(datasets_holdout)
        if self.use_mode_predictor:
            self.mode_predictor.pretrain(datasets_holdout)

    def forward(self, inputs, preproc=True, postproc=True, training=True, sparse_kwargs=None, inner_kwargs=None,
                mp_kwargs=None, horizon_idx=None, **kwargs):
        inputs, out = self.preamble_forward(inputs, preproc=preproc)

        inner_kwargs = inner_kwargs or {}
        sparse_kwargs = sparse_kwargs or {}
        mp_kwargs = mp_kwargs or {}

        # run inner model
        with timeit('inner_model'):
            inner_outs = self.inner_model(inputs, timeit_prefix="inner_model/", **inner_kwargs)

        if self.use_mode_predictor:
            # run mode prediction model
            with timeit('mode_predictor'):
                mp_inputs = inputs
                if horizon_idx is not None:
                    # only run mode predictor on whatever horizon_idx is
                    mp_inputs = mp_inputs.leaf_apply(lambda arr: arr[:, horizon_idx, None])
                mp_outs = self.mode_predictor(mp_inputs, timeit_prefix="mode_pred_model/", **mp_kwargs)
                mp_logit = mp_outs[self.mode_logit_out_name]
                out.mode_predictor = mp_outs
        else:
            # compute mode from inner model output partial embedding
            embedding = inner_outs[self.policy_raw_out_name]

            with timeit('inner_model_heads'):
                mp_logit = self.mode_head(embedding)
                inner_outs[self.policy_raw_out_name] = self.action_head(embedding)

        # cap the end of the policy (inner_model)
        out[self.policy_raw_out_name] = self.inner_model_cap(inner_outs[self.policy_raw_out_name])
        out[self.mode_prob_out_name] = F.softmax(mp_logit, dim=-1)
        out[self.mode_logit_out_name] = mp_logit
        out[self.mode_key] = torch.argmax(mp_logit, dim=-1, keepdim=True)

        # run sparse model
        with timeit('sparse_model'):
            sp_inputs = inputs
            if horizon_idx is not None:
                # only run mode predictor on whatever horizon_idx is
                sp_inputs = sp_inputs.leaf_apply(lambda arr: arr[:, horizon_idx, None])
            sparse_outs = self.sparse_model(sp_inputs, timeit_prefix="sparse_model/", **sparse_kwargs)

        # cap the end of the sparse policy (sparse_model)
        sparse_outs[self.sparse_raw_out_name] = self.sparse_model_cap(sparse_outs[self.sparse_raw_out_name])
        out.combine(sparse_outs)

        # parse the action names from either vector or distribution.
        out = self.parse_model_output_fn(self.env_spec, inputs, out,
                                         self.policy_raw_out_name, self.action_names,
                                         use_mean=self.use_policy_dist_mean, sample_cat=self.sample_cat)

        # parse the sparse names from either vector or distribution.
        out = self.parse_model_output_fn(self.env_spec, inputs, out,
                                         self.sparse_raw_out_name, self.sparse_action_names,
                                         use_mean=self.sparse_use_policy_dist_mean, sample_cat=self.sparse_sample_cat)

        # RAW
        out.inner_model = inner_outs
        out.sparse_model = sparse_outs

        return self._postproc_fn(inputs, out) if postproc else out

    def action_loss_fn(self, model, policy_out, ins, outs, mode=0, **kwargs):
        """ default: return the losses passed in params. """
        return self.ac_loss_fns[mode](model, policy_out, ins, outs, **kwargs)

    @staticmethod
    def default_loss_fn(self: BaseGCBC, policy_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="", **kwargs):
        true_mode = (ins[self.mode_key]).to(dtype=torch.long)  # index
        B, H = true_mode.shape[:2]

        mode_action_losses = []
        coeffs = []

        for m in range(self.N_MODES):
            with timeit(f'loss/policy_mode_{m}_loss'):
                mode_action_losses.append(
                    self.action_loss_fn(self, policy_out, ins, outs, i=i, writer=writer,
                                        writer_prefix=writer_prefix + f"action_mode_{m}/",
                                        **kwargs, mode=m))

            # loss where in the relevant mode
            importance_weight = 1.
            if self.balance_mode_loss:
                importance_weight = true_mode.numel() / ((true_mode == m).to(
                    dtype=torch.float32).sum() + 1e-4)  # each mode gets it own equal loss, basically.
                importance_weight = importance_weight.item()
                # weighting on mode's samples.
                importance_weight = torch.where(true_mode == m, importance_weight, 1.).view(B, H)
            coeffs.append(importance_weight * torch.where(true_mode == m, 1 - self.gamma, self.gamma).view(B, H))

        # mode probability loss
        mode_prob = policy_out[self.mode_prob_out_name]

        with timeit('loss/mode_loss'):
            # predict the right mode
            if self.use_smooth_mode:
                smooth_mode = (ins["smooth_mode"]).view(B * H, 1)
                all_prob = torch.cat([1. - smooth_mode, smooth_mode], dim=-1)
                mode_loss = self.mode_loss_obj(mode_prob.view(B * H, -1), all_prob).mean()
            else:
                mode_loss = self.mode_loss_obj(mode_prob.view(B * H, -1), true_mode.view(B * H)).mean()

        if writer is not None:
            with timeit('writer'):
                writer.add_scalar(writer_prefix + f"mode_loss", mode_loss.item(), i)
                for m in range(self.N_MODES):
                    writer.add_scalar(writer_prefix + f"action_mode{m}_loss", mode_action_losses[m].mean().item(),
                                      i)
                    writer.add_scalar(writer_prefix + f"weighted_action_mode{m}_loss",
                                      (coeffs[m] * mode_action_losses[m]).mean().item(), i)

        # with timeit('loss/combine_all'):
        #     loss = 0
        #     if i >= self._train_policy_after_step:
        #         loss = sum((w * c * l).mean() for w, c, l in zip(self.loss_weights, coeffs, mode_action_losses))
        #
        #     if i < self._train_mode_until_step:
        #         loss = self.mode_beta * mode_loss + loss

        loss = sum((w * c * l).mean() for w, c, l in zip(self.loss_weights, coeffs, mode_action_losses))
        loss = self.mode_beta * mode_loss + loss

        return loss

    @staticmethod
    def get_default_mem_policy_forward_fn(*args, flush_horizon=0, add_goals_in_hor=False, **kwargs):
        assert isinstance(flush_horizon, int) and flush_horizon >= 0, flush_horizon
        if flush_horizon == 0:
            logger.warn(
                "Note: DynGCBC will never flush the hidden state online (flush_horizon=0)! This can cause issues "
                "online.")

        # online execution using MemoryPolicy or subclass
        def mem_policy_model_forward_fn(model: DynamicActionBaseGCBC, obs: d, goal: d, memory: d, **inner_kwargs):
            obs = obs.leaf_copy()
            if memory.is_empty():
                memory.count = 0  # total steps
                memory.flush_count = 0  # just for flushing the rnn state

                # init hidden_state
                if model.use_mode_predictor and isinstance(model.mode_predictor, RnnModel):
                    memory.mp_hn = model.mode_predictor.hidden_name
                    memory.mp_state = {'rnn_hidden_init': None}
                if isinstance(model.sparse_model, RnnModel):
                    memory.m0_hn = model.sparse_model.hidden_name
                    memory.m0_state = {'rnn_hidden_init': None}
                if isinstance(model.inner_model, RnnModel):
                    memory.m1_hn = model.inner_model.hidden_name
                    memory.m1_state = {'rnn_hidden_init': None}

            if is_next_cycle(memory.flush_count, flush_horizon):
                if memory.has_leaf_key('mp_state'):
                    memory['mp_state']['rnn_hidden_init'] = None
                if memory.has_leaf_key('m0_state'):
                    memory['m0_state']['rnn_hidden_init'] = None
                if memory.has_leaf_key('m1_state'):
                    memory['m1_state']['rnn_hidden_init'] = None

            if not add_goals_in_hor and not goal.is_empty():
                obs.goal_states = goal

            memory.count += 1
            memory.flush_count += 1

            out = model.forward(obs, mp_kwargs=memory << "mp_state", sparse_kwargs=memory << "m0_state",
                                inner_kwargs=memory << "m1_state", **inner_kwargs)

            # update hidden_state
            if memory.has_leaf_key('mp_state'):
                memory['mp_state']['rnn_hidden_init'] = out[f"mode_predictor/{memory.mp_hn}"]
            if memory.has_leaf_key('m0_state'):
                memory['m0_state']['rnn_hidden_init'] = out[f"sparse_model/{memory.m0_hn}"]
            if memory.has_leaf_key('m1_state'):
                memory['m1_state']['rnn_hidden_init'] = out[f"inner_model/{memory.m1_hn}"]

            # default online postproc defined in Model
            return model.online_postproc_fn(model, out, obs, goal, memory, **inner_kwargs)

        return mem_policy_model_forward_fn


#####################################################
#   versions of each with different architectures   #
# lookups will be in left to right order, then base #
#####################################################


class DAS_SparseMLP_GCBC(DynamicActionBaseGCBC):
    """
    MLP implementation for sparse network
    """
    sparse_model_cls = BasicModel

    sparse_dropout_p = Argument("sparse_dropout_p", type=float, default=0)
    sparse_mlp_size = Argument("sparse_mlp_size", type=int, default=128)
    sparse_mlp_depth = Argument("sparse_mlp_depth", type=int, default=3)

    predefined_arguments = DynamicActionBaseGCBC.predefined_arguments + [
        sparse_dropout_p, sparse_mlp_size, sparse_mlp_depth
    ]

    def _get_sparse_model(self) -> d:
        self.sparse_mlp_network = SequentialParams(
            build_mlp_param_list(self.sparse_in_size,
                                 [self.sparse_mlp_size] * self.sparse_mlp_depth + [self.sparse_policy_raw_out_size],
                                 dropout_p=self.sparse_dropout_p))

        return d(
            cls=BasicModel,
            network=self.sparse_mlp_network
        )


class RNN_DAS_GCBC(RNN_GCBC, DAS_SparseMLP_GCBC):
    predefined_arguments = resolve_arguments(RNN_GCBC, DAS_SparseMLP_GCBC)

    # switch the inheritance path to DAS by default for the mem policy fn
    @classmethod
    def get_default_mem_policy_forward_fn(cls, *args, **kwargs):
        return DAS_SparseMLP_GCBC.get_default_mem_policy_forward_fn(*args, **kwargs)


class Transformer_DAS_GCBC(TransformerGCBC, DAS_SparseMLP_GCBC):
    predefined_arguments = resolve_arguments(TransformerGCBC, DAS_SparseMLP_GCBC) + [
        Argument('use_dynamic_mask', action='store_true'),
    ]

    def compute_transformer_mask(self, inputs, is_training=False):
        # transformer specific, masking by mode (dense modes can't see previous segments.
        with timeit('compute_transformer_mask'):
            # B x H
            mode = combine_after_dim(inputs[self.mode_key].to(dtype=torch.bool), 1)
            B, H = mode.shape
            #
            prev_mode = torch.constant_pad_nd(mode, [1, 0])[:, :-1]
            transition = mode & ~prev_mode

            mask = torch.ones([B, H, H], dtype=torch.float32, device=mode.device)
            # z x 2
            transition_idxs = transition.nonzero()
            # TODO can we parallelize this?
            for b, h in transition_idxs:
                # one step delay on when the mask gets updated during training
                # (since modes only get predicted as an output during eval)
                mask[b, h + int(is_training):, :h] = 0.

        # (B x 1 x T x T), second dim is for num_heads
        return mask[:, None]

    def forward(self, inputs, inner_kwargs=None, training=True, **kwargs):
        if self.use_dynamic_mask:
            # update the inner mask
            inner_kwargs = inner_kwargs or {}
            inner_kwargs['mask'] = self.compute_transformer_mask(inputs, is_training=training)
        return super().forward(inputs, inner_kwargs=inner_kwargs, training=training, **kwargs)

    # forward function needs to handle transformer inner model.
    @staticmethod
    def get_default_mem_policy_forward_fn(*args, flush_horizon=0, add_goals_in_hor=False, output_horizon_idx=None,
                                          horizon=None, **kwargs):
        assert isinstance(flush_horizon, int) and flush_horizon >= 0, flush_horizon
        if flush_horizon == 0:
            logger.warn(
                "Note: DynGCBC will never flush the hidden state online (flush_horizon=0)! This can cause issues "
                "online.")

        # online execution using MemoryPolicy or subclass
        def mem_policy_model_forward_fn(model: DynamicActionBaseGCBC, obs: d, goal: d, memory: d, **inner_kwargs):
            nonlocal output_horizon_idx, horizon

            # run vision for one step, treat it as an additional input
            if model.vision_model is not None:
                with timeit('vision_model_online'):
                    embed = model.vision_model(obs, timeit_prefix="vision_model/")
                    obs &= embed

            inputs = obs.leaf_copy()
            if not add_goals_in_hor and not goal.is_empty():
                inputs.goal_states = goal
            else:
                inputs = inputs & goal

            if memory.is_empty():
                memory.count = 0  # total steps
                memory.flush_count = 0  # just for flushing any of the rnn states

                if output_horizon_idx is None:
                    output_horizon_idx = model.inner_model._default_output_horizon_idx
                assert output_horizon_idx == -1, \
                    "This function might not work for output_horizon_idx != last, due to indexing of different model outputs"
                if horizon is None:
                    # prefer model's horizon to inner model's horizon (e.g. if we are stacking tokens)
                    if hasattr(model, "horizon"):
                        horizon = model.horizon
                    else:
                        assert hasattr(model.inner_model, "_horizon")
                        horizon = model.inner_model._horizon

                # list of inputs, shape (B x 1 x ..), will be concatenated later
                memory.input_history = [model.inner_model.get_online_inputs(inputs) for _ in range(horizon)]

                # model might need the mode, set all first ones to 0 by default.
                # will be shifted left one compared to training
                memory.output_mode_history = [torch.zeros((1, 1, 1), device=model.device) for _ in range(horizon)]
                memory.out_mode = torch.zeros((1, horizon, 1), device=model.device)

                # avoid allocating memory again
                memory.alloc_inputs = d.leaf_combine_and_apply(memory.input_history,
                                                               lambda vs: torch.cat(vs, dim=1))

                # init hidden_state
                if model.use_mode_predictor and isinstance(model.mode_predictor, RnnModel):
                    memory.mp_hn = model.mode_predictor.hidden_name
                    memory.mp_state = {'rnn_hidden_init': None}
                if isinstance(model.sparse_model, RnnModel):
                    memory.m0_hn = model.sparse_model.hidden_name
                    memory.m0_state = {'rnn_hidden_init': None}

            if is_next_cycle(memory.flush_count, flush_horizon):
                if memory.has_leaf_key('mp_state'):
                    memory['mp_state']['rnn_hidden_init'] = None
                if memory.has_leaf_key('m0_state'):
                    memory['m0_state']['rnn_hidden_init'] = None

            memory.count += 1
            memory.flush_count += 1

            # add new inputs, maintaining sequence length
            memory.input_history = memory.input_history[1:] + [model.inner_model.get_online_inputs(inputs)]

            def set_vs(k, vs):
                # set allocated array, return None
                torch.cat(vs, dim=1, out=memory.alloc_inputs[k])

            d.leaf_combine_and_apply(memory.input_history, set_vs, pass_in_key_to_func=True)

            torch.cat(memory.output_mode_history, dim=1, out=memory.out_mode)

            # TODO horizon idx should tell the model to only run sparse / mp on the last state.
            # NOTE sparse is an MLP, which is why horizon_idx=-1. for transformer, sparse would need to be different.
            out = model.forward(memory.alloc_inputs & d.from_dict({model.mode_key: memory.out_mode}),
                                mp_kwargs=memory << "mp_state", sparse_kwargs=memory << "m0_state",
                                horizon_idx=-1, skip_vision=True, training=False, **inner_kwargs)

            # update hidden_state
            if memory.has_leaf_key('mp_state'):
                memory['mp_state']['rnn_hidden_init'] = out[f"mode_predictor/{memory.mp_hn}"]
            if memory.has_leaf_key('m0_state'):
                memory['m0_state']['rnn_hidden_init'] = out[f"sparse_model/{memory.m0_hn}"]

            # grab a horizon index from the base model's output arrays (default is last idx)
            # then reshape to (B x 1 x ...)
            out.combine(
                out.leaf_arrays().leaf_apply(lambda arr: arr[:, output_horizon_idx, None])
            )

            # add the mode in
            memory.output_mode_history = memory.output_mode_history[1:] + [out[model.mode_key]]

            # default online postproc defined in Model
            return model.online_postproc_fn(model, out, obs, goal, memory, **inner_kwargs)

        return mem_policy_model_forward_fn
