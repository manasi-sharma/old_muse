"""
Different types of Goal Conditioned BC in class format
"""
import abc
from typing import List

import numpy as np
from muse.utils.torch_utils import unsqueeze_then_gather
import torch.distributions

from muse.experiments import logger
from muse.models.basic_model import BasicModel
from muse.models.gpt.bet_layers import TransformerConfig
from muse.models.bc.lmp import play_helpers
from muse.models.model import Model
from muse.models.rnn_model import RnnModel
from muse.models.seq_model import SequenceModel
from muse.models.vision import vision_encoders as ve
from muse.utils.abstract import Argument
from muse.utils.param_utils import LayerParams, build_mlp_param_list, SequentialParams, get_policy_dist_out_size, \
    get_policy_dist_cap
from muse.utils.general_utils import timeit

from attrdict import AttrDict as d
from attrdict.utils import get_with_default, get_or_instantiate_cls

class BaseGCBC(Model):
    """
    GCBC base class, compatible with vision based or state based inputs.

    Supports deterministic and probabilistic (gaussian or gmm) action spaces (implemented at sub-class level)

    Sub-classes will declare the inner_model.

    """

    # its useful to know the exact class (e.g. for declaring the policy_forward_fn)
    inner_model_cls = None

    # add them to the arguments
    predefined_arguments = Model.predefined_arguments + [
        # simple arguments defined here, will be overwritten with value during init()
        Argument("use_goal", action="store_true"),
        Argument("use_final_goal", action="store_true"),
        Argument("normalize_states", action='store_true'),
        Argument("normalize_actions", action='store_true'),

        # default encoder arguments
        Argument("use_vision_encoder", action='store_true', help="enable vision encoder"),
        Argument("encoder_call_jointly", action='store_true', help="encode all image names together"),
        Argument("default_img_embed_size", type=int, default=64, help="embedding size per image"),
        Argument("default_use_spatial_softmax", action='store_true'),
        Argument("default_use_crop_randomizer", action='store_true'),
        Argument("default_use_color_randomizer", action='store_true'),
        Argument("default_use_erasing_randomizer", action='store_true'),
        Argument("default_downsample_frac", type=float, default=1.),
        Argument("crop_frac", type=float, default=0.9),
        Argument("encoder_use_shared_params", action='store_true',
                 help="we will share net params for the each image key."),
        # arguments that base classes should utilize for model specific things
        Argument("use_policy_dist", action="store_true"),
        Argument("policy_num_mix", type=int, default=1),
        Argument("use_policy_dist_mean", action="store_true"),
        Argument("policy_sig_min", type=float, default=1e-5),
        Argument("policy_sig_max", type=float, default=1e3),
        Argument("use_tanh_out", action="store_true"),
    ]

    def _init_params_to_attrs(self, params):
        self.read_predefined_params(params)

        # states
        self.state_names = params["state_names"]
        # goals (not prefixed by goal), likely some subset of state names
        self.goal_names = params["goal_names"]
        # actions
        self.action_names = params["action_names"]

        # compute the input size of the policy
        self.policy_in_names = list(self.state_names)
        self.policy_raw_out_name = get_with_default(params, "raw_out_name", "policy_raw")

        if self.use_goal:
            self.policy_in_names += list(self.goal_names)

        # save the policy names pre adding vision
        self.novision_policy_in_names = list(self.policy_in_names)
        self.policy_in_size = self.env_spec.dim(self.policy_in_names)
        self.policy_out_size = self.env_spec.dim(self.action_names)  # e.g., deterministic output
        
        self.sample_cat = get_with_default(params, "sample_cat", False)
        if self.sample_cat: 
            logger.debug("Will sample from categorical distribution (make sure you are using compatible action space)")

        # size of the output of network
        self.policy_raw_out_size = get_policy_dist_out_size(self.policy_out_size,
                                                            prob=self.use_policy_dist, num_mix=self.policy_num_mix)

        # --- figure out vision model params --- #

        # (optional) encoder params
        self._init_vision_params_to_attrs(params)

        # get encoders
        self.vision_encoder_params = self._get_vision_encoder_params(params)

        # add vision embeddings to policy input
        if not self.vision_encoder_params.is_empty():
            logger.debug("[GCBC] Using vision encoder!")
            assert self.encoder_out_name not in self.state_names, "Do not add the encoder out name to state_names! " \
                                                                  "This is handled internally"
            self._update_policy_in_from_vision()

        # --- figure out inner model params --- #

        self._inner_model_params_to_attrs(params)

        assert not self.normalize_inputs, "GCBC should not normalize anything! handled by inner_model..."

        # for inner_model to use
        self.inner_normalize_inputs = []
        self.inner_save_normalize_inputs = []

        # states will be normalized & stats saved
        if self.normalize_states:
            self.inner_normalize_inputs += self.state_names
            self.inner_save_normalize_inputs += self.state_names

        # action stats will be saved
        if self.normalize_actions:
            self.inner_save_normalize_inputs += self.action_names

        # goals will be normalized & stats saved
        if self.use_final_goal:
            self.inner_normalize_inputs += [f"goal/{g}" for g in self.goal_names]
            self.inner_save_normalize_inputs += [f"goal/{g}" for g in self.goal_names]

        # preproc fn
        self.inner_preproc_fn = get_with_default(params, "inner_preproc_fn", self.get_default_preproc_fn())
        self.inner_postproc_fn = get_with_default(params, "inner_postproc_fn", None)  # self.get_default_postproc_fn())

        # the model we instantiate
        self.inner_model_params = self._get_inner_model_shared_params() & self._get_inner_model()

        logger.debug(f"[GCBC] use_goal={self.use_goal}, use_final_goal={self.use_final_goal}")
        logger.debug(f"[GCBC] Policy in names: {self.policy_in_names}")
        logger.debug(f"[GCBC] Action names: {self.action_names}")
        if self.normalize_states:
            logger.debug(f"[GCBC] Normalize input names: {self.state_names}")
        if self.normalize_actions:
            logger.debug(f"[GCBC] Normalize action names: {self.action_names}")

    def _init_setup(self):
        self._init_models()
        self._init_caps()

        self._set_models()

    def _init_models(self):
        model_cstr = lambda cls, prms: cls(prms, self.env_spec, self._dataset_train)

        self.vision_model = None
        if not self.vision_encoder_params.is_empty():
            # instantiate the vision encoder
            self.vision_model = get_or_instantiate_cls(self.vision_encoder_params, None, Model, constructor=model_cstr)

        # instantiate inner model
        self.inner_model = get_or_instantiate_cls(self.inner_model_params, None, Model, constructor=model_cstr)

    def _init_caps(self):
        # cap for the inner model
        self.inner_model_cap = get_policy_dist_cap(self.use_policy_dist, self.use_tanh_out, num_mix=self.policy_num_mix,
                                                   policy_sig_min=self.policy_sig_min,
                                                   policy_sig_max=self.policy_sig_max)
        self.inner_model_cap = self.inner_model_cap.to_module_list(as_sequential=True).to(self.device)

    def _set_models(self):
        self._models = [self.inner_model]
        if self.vision_model:
            self._models.append(self.vision_model)

    def warm_start(self, model, observation, goal):
        pass

    def pretrain(self, datasets_holdout=None):
        """ Pretrain actions """
        if self.vision_model is not None:
            self.vision_model.pretrain(datasets_holdout=datasets_holdout)

        self.inner_model.pretrain(datasets_holdout=datasets_holdout)

    def _update_policy_in_from_vision(self):
        # updates the sizes and names of policy input with knowledge of using vision
        # self.state_names += [self.encoder_out_name]
        self.policy_in_names += [self.encoder_out_name]
        self.policy_in_size += self.encoder_out_size

    def _init_vision_params_to_attrs(self, params):
        self.encoder_out_name = get_with_default(params, "encoder_out_name", "img_embed_out")

        # DEFAULT PARAMETERS
        self.image_keys = get_with_default(params, "image_keys", ["image"])

        # encode sizes
        # self.default_img_embed_size = get_with_default(params, "default_img_embed_size", 64)
        self.img_embed_sizes = get_with_default(params, "img_embed_sizes",
                                                [self.default_img_embed_size] * len(self.image_keys))
        assert len(self.img_embed_sizes) == len(self.image_keys), [self.img_embed_sizes, self.image_keys]

        self.encoder_out_size = sum(self.img_embed_sizes)

    def _get_vision_encoder_params(self, params) -> d:
        if not self.use_vision_encoder:
            return d()

        if "vision_encoder_params" in params.keys():
            assert params['vision_encoder_params'].has_node_leaf_keys(['cls', 'params']), \
                "[GCBC] Vision params specified but not instantiable!"
            return params['vision_encoder_params']

        if (np.array(self.img_embed_sizes) != self.img_embed_sizes[0]).any():
            # embed to different sizes if we are calling things separately.
            assert self.call_separate and not self.use_shared_params, \
                "Cannot embed images to different sizes unless calling net separately!"
            logger.debug("[GCBC] using multiple encoder nets (different projection")
            # we need explicitly different nets for each input
            encoder_net = get_with_default(params, "vision_encoder_net",
                                           [ve.get_resnet18_encoder_layer(
                                               self.env_spec.names_to_shapes[self.image_keys[i]],
                                               crop_random_frac=self.crop_frac if self.default_use_crop_randomizer else 0,
                                               downsample_frac=self.default_downsample_frac,
                                               use_color_randomizer=self.default_use_color_randomizer,
                                               use_erasing_randomizer=self.default_use_erasing_randomizer,
                                               use_spatial_softmax=self.default_use_spatial_softmax,
                                               img_embed_size=self.img_embed_sizes[i])
                                               for i in range(len(self.image_keys))])
            assert len(encoder_net) == len(self.image_keys), [len(encoder_net), len(self.image_keys)]
        else:
            # pass in the same encoder net layer
            encoder_net = get_with_default(params, "vision_encoder_net", ve.get_resnet18_encoder_layer(
                self.env_spec.names_to_shapes[self.image_keys[0]],
                crop_random_frac=self.crop_frac if self.default_use_crop_randomizer else 0,
                downsample_frac=self.default_downsample_frac,
                use_color_randomizer=self.default_use_color_randomizer,
                use_erasing_randomizer=self.default_use_erasing_randomizer,
                use_spatial_softmax=self.default_use_spatial_softmax,
                img_embed_size=self.img_embed_sizes[0]))

        return d(
            cls=BasicModel,
            device=self.device,
            model_inputs=self.image_keys,
            model_output=self.encoder_out_name,
            call_separate=not self.encoder_call_jointly,
            use_shared_params=self.encoder_use_shared_params,
            network=encoder_net,
        )

    def _inner_model_params_to_attrs(self, params):
        pass

    def _get_inner_model_shared_params(self):
        # some common parameters to use as a starting point
        return d(
            cls=self.inner_model_cls,
            device=self.device,
            model_inputs=self.policy_in_names,
            model_output=self.policy_raw_out_name,
            preproc_fn=self.inner_preproc_fn,
            postproc_fn=self.inner_postproc_fn,
            normalize_inputs=len(self.inner_normalize_inputs) > 0,
            normalization_inputs=self.inner_normalize_inputs,
            save_normalization_inputs=self.inner_save_normalize_inputs,
            default_normalize_sigma=self.default_normalize_sigma,
        )

    @abc.abstractmethod
    def _get_inner_model(self) -> d:
        raise NotImplementedError

    def get_default_preproc_fn(self):
        # reusing some stuff to get the goal and preprocess this.
        return play_helpers.get_gcbc_preproc_fn(not self.use_goal, self.use_final_goal, device=self.device,
                                                POLICY_NAMES=self.state_names + (
                                                    [] if self.vision_encoder_params.is_empty() else [
                                                        self.encoder_out_name]),
                                                POLICY_GOAL_STATE_NAMES=self.goal_names)

    @staticmethod
    def parse_model_output_fn(env_spec, inputs: d, model_outputs: d, raw_out_name, action_names, use_mean=False, sample_cat=False):
        """
        Parses model output (raw_out_name) into keys (action_names)
        """
        new_outs = model_outputs.leaf_copy()
        raw = model_outputs[raw_out_name]  # B x H x

        if isinstance(raw, torch.distributions.Distribution):
            # raw is a distribution
            # sample = raw.rsample()
            if use_mean:
                sample = raw.mean  # mean
            elif sample_cat:
                assert use_mean, "Cannot sample cat but not use mean!"
                assert isinstance(raw, torch.distributions.MixtureSameFamily), "raw must be mixture, but was: {raw}"
                mean = raw.component_distribution.mean  # .. x k x D
                _, max_idxs = raw.mixture_distribution.sample()  # ..
                sample = unsqueeze_then_gather(mean, max_idxs, dim=len(max_idxs.shape))  # .. x D
            else:
                sample = raw.rsample()  # sample
        else:
            sample = raw

        new_outs.combine(env_spec.parse_view_from_concatenated_flat(sample, action_names))
        return new_outs

    def load_statistics(self, dd=None):
        dd = super(BaseGCBC, self).load_statistics(dd)
        for m in self._models:
            dd = m.load_statistics(dd)
        return dd

    def normalize_by_statistics(self, inputs: d, names, shared_dtype=None, check_finite=True, inverse=False,
                                shift_mean=True):
        """ Nested normalization """
        this_level_names = list(set(names).intersection(self.save_normalization_inputs))
        names = list(set(names).difference(self.save_normalization_inputs))  # missing ones
        if len(this_level_names) > 0:
            inputs = super().normalize_by_statistics(inputs, this_level_names,
                                                     shared_dtype=shared_dtype,
                                                     check_finite=check_finite, inverse=inverse,
                                                     shift_mean=shift_mean)

        for model in self._models:
            if len(names) == 0:
                break
            next_level_names = list(set(names).intersection(model.save_normalization_inputs))
            names = list(set(names).difference(model.save_normalization_inputs))  # missing ones
            if len(next_level_names) > 0:
                inputs = model.normalize_by_statistics(inputs, next_level_names,
                                                       shared_dtype=shared_dtype,
                                                       check_finite=check_finite, inverse=inverse,
                                                       shift_mean=shift_mean)

        if len(names) > 0:
            raise ValueError(f"Missing names to normalize: {names}")

        return inputs

    def preamble_forward(self, inputs, preproc=True, skip_vision=False):
        inputs = inputs.leaf_copy()

        if preproc:
            inputs = self._preproc_fn(inputs)

        out = d()

        if self.vision_model is not None:
            if self.encoder_out_name not in inputs.leaf_keys():
                assert not skip_vision, f"Vision is required since {self.encoder_out_name} was missing from input, but skip_vision=True!"
                    
                # run vision to get embeddings
                with timeit('vision_model'):
                    embed = self.vision_model(inputs, timeit_prefix="vision_model/")
                    out.vision_out = embed

                # combine
                inputs = inputs & embed

        return inputs, out

    def forward(self, inputs, preproc=True, postproc=True, training=True, skip_vision=False, **kwargs):
        inputs, out = self.preamble_forward(inputs, preproc=preproc, skip_vision=skip_vision)

        # run inner model
        with timeit('inner_model'):
            inner_outs = self.inner_model(inputs, timeit_prefix="inner_model/", **kwargs)
            # cap it off
            inner_outs[self.policy_raw_out_name] = self.inner_model_cap(inner_outs[self.policy_raw_out_name])

        # parse the action names from either vector or distribution.
        out = self.parse_model_output_fn(self.env_spec, inputs, inner_outs,
                                         self.policy_raw_out_name, self.action_names, use_mean=self.use_policy_dist,
                                         sample_cat=self.sample_cat)
        out.combine(inner_outs)

        return self._postproc_fn(inputs, out) if postproc else out

    def print_parameters(self, prefix="", print_fn=logger.debug):
        print_fn(prefix + "[GCBC]")
        for n, p in self.named_parameters():
            print_fn(prefix + "[GCBC] %s <%s> (requires_grad = %s)" % (n, list(p.shape), p.requires_grad))

    @classmethod
    def get_default_mem_policy_forward_fn(cls, *args, **kwargs):
        return cls.inner_model_cls.get_default_mem_policy_forward_fn(*args, **kwargs)


class MLP_GCBC(BaseGCBC):
    """
    MLP implementation for policy
    """
    inner_model_cls = BasicModel

    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument("dropout_p", type=float, default=0),
        Argument("mlp_size", type=int, default=128),
        Argument("mlp_depth", type=int, default=3),
    ]

    def _get_inner_model(self) -> d:
        self.mlp_network = SequentialParams(
            build_mlp_param_list(self.policy_in_size, [self.mlp_size] * self.mlp_depth + [self.policy_raw_out_size],
                                 dropout_p=self.dropout_p))

        return d(
            cls=BasicModel,
            network=self.mlp_network
        )


class RNN_GCBC(BaseGCBC):
    """
    RNN inner model, with support for LSTM or GRU
    """
    inner_model_cls = RnnModel

    arg_hidden_size = Argument("hidden_size", type=int, default=128)
    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument("dropout_p", type=float, default=0),
        arg_hidden_size,
        Argument("policy_size", type=int, default=arg_hidden_size),
        Argument("rnn_depth", type=int, default=2),
        Argument("rnn_type", type=str, default="gru"),
    ]

    def _inner_model_params_to_attrs(self, params):
        super()._inner_model_params_to_attrs(params)
        assert self.rnn_type in ['lstm', 'gru'], f"RNN type unimplemented: {self.rnn_type}"

    def _get_inner_model(self) -> d:
        # mlp after the rnn (will be removed if policy_size=0, special case)
        mlp_after_rnn_dims = [self.policy_size, self.policy_size, self.policy_raw_out_size]
        if self.policy_size == 0:
            mlp_after_rnn_dims = [self.policy_raw_out_size]  # no mlp.

        self.mlp_after_rnn_network = SequentialParams(build_mlp_param_list(self.hidden_size, mlp_after_rnn_dims,
                                                                           dropout_p=self.dropout_p))

        return d(
            cls=RnnModel,
            rnn_output_name="rnn_output_policy",
            hidden_name="hidden_policy",
            rnn_before_net=True,
            tuple_hidden=self.rnn_type == "lstm",
            recurrent_network=LayerParams(self.rnn_type, input_size=self.policy_in_size,
                                          hidden_size=self.hidden_size, num_layers=self.rnn_depth,
                                          bidirectional=False, batch_first=True,
                                          dropout=self.dropout_p),
            # outputs (B x Seq x Hidden)
            network=self.mlp_after_rnn_network,
        )

    @classmethod
    def get_default_mem_policy_forward_fn(cls, *args, **kwargs):
        forward_fn = cls.inner_model_cls.get_default_mem_policy_forward_fn(*args, **kwargs)
        # RNN specific thing, forward on the inner model
        return lambda model, *inner_args, root_model=None, **inner_kwargs: forward_fn(model.inner_model, *inner_args,
                                                                                      root_model=model if root_model is None else root_model,
                                                                                      **inner_kwargs)


class TransformerGCBC(BaseGCBC):
    """
    Transformer inner model, with support for GPT-like arch
    """
    inner_model_cls = SequenceModel

    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument("transformer_type", type=str, default="gpt"),
        Argument("dropout_p", type=float, default=0),
        Argument("transformer_dropout_p", type=float, default=0.1),
        Argument("n_embed", type=int, default=128),
        Argument("n_head", type=int, default=16),
        Argument("n_layer", type=int, default=8),
        Argument("no_causal", action='store_true'),
    ]

    def _inner_model_params_to_attrs(self, params):
        super()._inner_model_params_to_attrs(params)

        self.horizon = params["horizon"]
        # self._dropout_p = get_with_default(params, "dropout_p", 0)

        # accepts either a TransformerConfig, or individual params
        if params.has_leaf_key('transformer_cfg'):
            self.cfg = params.transformer_cfg
            assert isinstance(self.cfg, TransformerConfig)
            assert self.cfg.block_size >= self.horizon, "Horizon length is not covered by transformer!"
            self.transformer_out_size = self.cfg.vocab_size
        else:
            logger.debug("[Transformer-GCBC] Transformer config not specified, will create one using params")
            self.transformer_out_size = get_with_default(params, "transformer_out_size", self.policy_out_size)
            self.cfg = TransformerConfig(vocab_size=self.transformer_out_size, block_size=self.horizon)
            self.cfg.embd_pdrop = self.cfg.attn_pdrop = self.cfg.resid_pdrop = self.transformer_dropout_p
            self.cfg.input_size = self.policy_in_size
            self.cfg.n_embd = self.n_embed
            self.cfg.n_layer = self.n_layer
            self.cfg.n_head = self.n_head
            self.cfg.causal = not self.no_causal

        # todo add some more
        assert self.transformer_type in ['gpt'], f"Transformer type unimplemented: {self.transformer_type}"

    @property
    def online_input_names(self) -> List[str]:
        return [] if self.vision_encoder_params.is_empty() else [self.encoder_out_name]

    def _get_inner_model(self) -> d:
        # transformer
        self.transformer_layer = LayerParams(self.transformer_type, config=self.cfg)

        # mlp after transformer, to project to policy_out_size
        self.mlp_after_rnn_layers = build_mlp_param_list(self.transformer_out_size, [self.policy_raw_out_size],
                                                         dropout_p=self.dropout_p)

        # full net
        self.transformer_net = SequentialParams([self.transformer_layer] + self.mlp_after_rnn_layers)

        return d(
            cls=SequenceModel,
            horizon=self.horizon,
            # we will pass in the encoder out name online as an additional input to aggregate.
            online_inputs=self.state_names + self.goal_names + self.online_input_names,
            network=self.transformer_net
        )

    @classmethod
    def get_default_mem_policy_forward_fn(cls, *args, **kwargs):
        forward_fn = cls.inner_model_cls.get_default_mem_policy_forward_fn(*args, **kwargs)
        # SequenceModel specific thing, forward on the inner model, with optional vision
        def mem_forward_fn(model, obs, goal, *inner_args, root_model=None, **inner_kwargs):
            if model.vision_model is not None:
                # run vision for one step, treat it as an additional input
                with timeit('vision_model_online'):
                    embed = model.vision_model(obs, timeit_prefix="vision_model/")
                    obs &= embed
            return forward_fn(model.inner_model, obs, goal, *inner_args, 
                              root_model=model if root_model is None else root_model, 
                              skip_vision=True,
                              **inner_kwargs)
        
        return mem_forward_fn