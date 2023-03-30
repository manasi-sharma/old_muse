import random

import torch
from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.experiments import logger
from muse.models.basic_model import BasicModel
from muse.models.bc.action_decoders import ActionDecoder, RNNActionDecoder, MLPActionDecoder
from muse.models.rnn_model import RnnModel
from muse.utils.abstract import Argument, resolve_arguments
from muse.utils.general_utils import timeit, value_if_none
import torch.nn.functional as F

from muse.utils.param_utils import get_dist_cap, SequentialParams, build_mlp_param_list, get_dist_out_size, LayerParams


class HydraActionDecoder(ActionDecoder):
    """
    Decodes input names into actions.
    """

    required_models = ['decoder', 'sparse_decoder']

    predefined_arguments = ActionDecoder.predefined_arguments + [
        # sparse action cap
        Argument("sparse_use_policy_dist", action="store_true"),
        Argument("sparse_policy_num_mix", type=int, default=1),
        Argument("sparse_use_policy_dist_mean", action="store_true"),
        Argument("sparse_policy_sig_min", type=float, default=1e-5),
        Argument("sparse_policy_sig_max", type=float, default=1e3),
        Argument("sparse_use_tanh_out", action="store_true"),
        Argument("sparse_policy_sample_cat", action="store_true"),

        # use a separate network for mode predictor.
        Argument("use_mode_predictor", action="store_true"),

        # noise for ablations
        Argument('mode_classifier_noise', type=float, default=0.,
                 help='Noise to add to probabilities before arg-maxing'),
        Argument('mode_classifier_flip_prob', type=float, default=0.,
                 help='likelihood of flipping mode classifier likelihood'),

        # if not separate mode prediction, here are the args that define it.
        Argument("decoder_inter_size", type=int, default=128),
        Argument("split_head_layers", type=int, default=2),
        Argument("mode_head_size", type=int, default=128),
        Argument("action_head_size", type=int, default=128),
    ]

    def _preload_params(self, params):
        if not self.use_mode_predictor:
            # output of decoder is the action size, keep it unchanged
            # output of decoder will be this now (need to do this before loading decoder params)
            self.decoder_out_size = self.decoder_inter_size

        # things before creating default model params
        super()._preload_params(params)

        self.sparse_policy_out_size = self.env_spec.dim(self.sparse_action_names)
        self.sparse_policy_raw_out_size = get_dist_out_size(self.sparse_policy_out_size,
                                                            prob=self.sparse_use_policy_dist,
                                                            num_mix=self.sparse_policy_num_mix)

        # get default params for grouped model before parsing but after reading params, allowing for override
        params.sparse_decoder = self.get_default_sparse_decoder_params() & \
                                get_with_default(params, "sparse_decoder", d())

        # same for the mode prediction head if we are using a separate one.
        if self.use_mode_predictor:
            params.mode_predictor = self.get_default_mode_predictor_params() & \
                                    get_with_default(params, "mode_predictor", d())
            if 'mode_predictor' not in params.model_order:
                logger.debug('Adding mode predictor to model order!')
                params.model_order.append('mode_predictor')

    def _init_params_to_attrs(self, params):
        self.sparse_action_names = params['sparse_action_names']
        self.mode_key = get_with_default(params, "mode_key", "mode")
        self.mode_logit_out_name = get_with_default(params, "mode_logit_out_name", "mode_logits")
        self.mode_prob_out_name = get_with_default(params, "mode_prob_out_name", "mode_probs")

        self.sparse_raw_out_name = get_with_default(params, "sparse_raw_out_name", "sparse_policy_raw")

        super()._init_params_to_attrs(params)

        # these are for various ablations.
        if self.mode_classifier_flip_prob > 0:
            logger.warn(f'Mode classifier using flip probability {self.mode_classifier_flip_prob}!')
        if self.mode_classifier_noise > 0:
            logger.warn(f'Mode classifier using uniform noise {self.mode_classifier_noise}!')

    def _init_action_caps(self):
        super()._init_action_caps()

        # cap for the sparse model
        self.sparse_action_cap = get_dist_cap(self.sparse_use_policy_dist, self.sparse_use_tanh_out,
                                              num_mix=self.sparse_policy_num_mix, sig_min=self.sparse_policy_sig_min,
                                              sig_max=self.sparse_policy_sig_max)
        self.sparse_action_cap = self.sparse_action_cap.to_module_list(as_sequential=True).to(self.device)

    def _init_setup(self):
        super()._init_setup()

        if not self.use_mode_predictor:
            # initialize 2 heads of hydra (third head has its own network)
            self.mode_head = SequentialParams(build_mlp_param_list(self.decoder_out_size,
                                                                   [self.mode_head_size] * self.split_head_layers +
                                                                   [2])
                                              ).to_module_list(as_sequential=True).to(device=self.device)
            self.action_head = SequentialParams(build_mlp_param_list(self.decoder_out_size,
                                                                     [self.action_head_size] * self.split_head_layers +
                                                                     [self.policy_raw_out_size])
                                                ).to_module_list(as_sequential=True).to(device=self.device)

    def get_default_sparse_decoder_params(self) -> d:
        """ Override this in sub-classes to specify default sparse decoders (e.g., RNN or MLP)

        Optionally use parameters in self (e.g. Argument values)

        Returns
        -------

        """

        return d()

    def get_default_mode_predictor_params(self) -> d:
        """ Override this in sub-classes to specify default mode prediction head (e.g., RNN or MLP)

        Returns
        -------

        """

        return d()

    def forward(self, inputs, preproc=True, postproc=True, training=True, horizon_idx=None, **kwargs):
        """
        Unlike base ActionDecoder, runs the sparse model as well, and also parses the mode and action.

        TODO support for separate mode predictor

        Parameters
        ----------
        inputs: AttrDict
        preproc
        postproc
        training
        horizon_idx
        kwargs

        Returns
        -------

        """
        inputs = inputs.leaf_copy()

        if preproc:
            inputs = self._preproc_fn(inputs)

        # run inner model
        with timeit('decoder'):
            decoder_outs = self.decoder(inputs, timeit_prefix="decoder/", **self.get_kwargs('decoder', kwargs))

        out = d()
        out['decoder'] = decoder_outs

        if self.use_mode_predictor:
            assert self.policy_raw_out_name in decoder_outs, \
                f"When using MP, policy_raw_out_name ({self.policy_raw_out_name}) " \
                f"should already be in the output of the decoder!"
            mode_outs = self.mode_predictor(inputs, timeit_prefix="mode_predictor/",
                                            **self.get_kwargs('mode_predictor', kwargs))
            assert mode_outs.has_leaf_keys([self.mode_key, self.mode_prob_out_name, self.mode_logit_out_name]), \
                "Missing mode, prob, and logit!"
            out['mode_predictor'] = mode_outs
            # move mode stuff to top level
            out = out & (mode_outs > [self.mode_key, self.mode_prob_out_name, self.mode_logit_out_name])
            assert self.mode_classifier_noise == 0., "Mode classifier noise not implemented for separate network yet!"
            assert self.mode_classifier_flip_prob == 0., \
                "Mode classifier flip prob not implemented for separate network yet!"
        else:
            with timeit('mode_action_heads'):
                # compute mode from inner model output partial embedding
                embedding = out.decoder[self.policy_raw_out_name]

                mp_logit = self.mode_head(embedding)
                probs = out[self.mode_prob_out_name] = F.softmax(mp_logit, dim=-1)
                out[self.mode_logit_out_name] = mp_logit

                if self.mode_classifier_noise > 0:
                    # U[-noise, noise] added to probs
                    uniform = (2 * torch.rand_like(probs[..., 0]) - 1) * self.mode_classifier_noise
                    probs = probs.clone()

                    # clip between 0 and 1.
                    probs[..., 0] = 1 - torch.relu(1 - torch.relu(probs[..., 0] - uniform))
                    probs[..., 1] = 1 - probs[..., 0]

                if self.mode_classifier_flip_prob > 0:
                    # flipping probabilities
                    if random.random() < self.mode_classifier_flip_prob:
                        probs = 1 - probs

                out[self.mode_key] = torch.argmax(probs, dim=-1, keepdim=True)

                # run the decoder output through the action head
                out.decoder[self.policy_raw_out_name] = self.action_head(embedding)

        # cap the end of the policy (action cap)
        out.decoder[self.policy_raw_out_name] = self.action_cap(out.decoder[self.policy_raw_out_name])

        # run sparse model
        with timeit('sparse_decoder'):
            sp_inputs = inputs
            if horizon_idx is not None:
                # only run mode predictor on whatever horizon_idx is
                sp_inputs = sp_inputs.leaf_apply(lambda arr: arr[:, horizon_idx, None])
            sparse_outs = self.sparse_decoder(sp_inputs, timeit_prefix="sparse_decoder/",
                                              **self.get_kwargs('sparse_decoder', kwargs))

        # cap the end of the sparse policy (sparse_decoder)
        sparse_outs[self.sparse_raw_out_name] = self.sparse_action_cap(sparse_outs[self.sparse_raw_out_name])
        out.sparse_decoder = sparse_outs

        # parse the action names from either vector or distribution.
        ac_dc = self.parse_raw_action(self.env_spec, out.decoder[self.policy_raw_out_name], self.action_names,
                                      use_mean=self.use_policy_dist_mean, sample_cat=self.policy_sample_cat)

        # parse the sparse names from either vector or distribution.
        spac_dc = self.parse_raw_action(self.env_spec, out.sparse_decoder[self.sparse_raw_out_name],
                                        self.sparse_action_names, use_mean=self.sparse_use_policy_dist_mean,
                                        sample_cat=self.sparse_policy_sample_cat)

        # combine the top level with the parsed actions
        out.combine(ac_dc)
        out.combine(spac_dc)

        return self._postproc_fn(inputs, out) if postproc else out

    @property
    def all_action_names(self):
        return self.action_names + self.sparse_action_names


""" Useful classes for sub models """


class ModePredictionRnnModel(RnnModel):
    predefined_arguments = BasicModel.predefined_arguments + [
        Argument("in_size", type=int, default=None),

        Argument("rnn_type", type=str, default="gru"),
        Argument("hidden_size", type=int, default=128),
        Argument("mlp_size", type=int, default=0),
        Argument("rnn_depth", type=int, default=2),
        Argument("dropout_p", type=float, default=0),
    ]

    def _init_params_to_attrs(self, params):
        self.read_predefined_params(params)

        if self.in_size is None:
            self.in_size = self.env_spec.dim(params.model_inputs)

        # mode logits
        self.out_size = 2

        # mlp after the rnn (will be removed if policy_size=0, special case)
        if self.mlp_size == 0:
            mlp_after_rnn_dims = [self.out_size]  # no mlp.
        else:
            mlp_after_rnn_dims = [self.mlp_size, self.mlp_size, self.out_size]

        params.rnn_output_name = get_with_default(params, 'rnn_output_name', 'rnn_output_policy')
        params.hidden_name = get_with_default(params, 'hidden_name', 'hidden_policy')
        params.rnn_before_net = True
        params.tuple_hidden = self.rnn_type == "lstm"

        params.recurrent_network = get_with_default(params, "recurrent_network",
                                                    LayerParams(self.rnn_type, input_size=self.in_size,
                                                                hidden_size=self.hidden_size,
                                                                num_layers=self.rnn_depth,
                                                                bidirectional=False, batch_first=True,
                                                                dropout=self.dropout_p))

        params.network = get_with_default(params, "network", SequentialParams(
            build_mlp_param_list(self.hidden_size, mlp_after_rnn_dims,
                                 dropout_p=self.dropout_p)))

        super()._init_params_to_attrs(params)

        self.mode_key = get_with_default(params, "mode_key", "mode")
        self.mode_prob_out_name = get_with_default(params, "mode_prob_out_name", "mode_probs")

    def forward(self, inputs, **kwargs):
        # add in extra output names
        out_dc = super().forward(inputs, **kwargs)
        mp_logit = out_dc[self.output]
        out_dc[self.mode_prob_out_name] = F.softmax(mp_logit, dim=-1)
        out_dc[self.mode_key] = torch.argmax(mp_logit, dim=-1, keepdim=True)
        return out_dc


class HydraSparseMLPActionDecoder(HydraActionDecoder):
    """
    Sparse decoder implemented as MLP. Mode predictor implemented as RNN.

    If use_mode_predictor, will use mode_head_size as the hidden size for rnn MP.
    """

    predefined_arguments = HydraActionDecoder.predefined_arguments + [
        Argument("sparse_dropout_p", type=float, default=0),
        Argument("sparse_mlp_size", type=int, default=128),
        Argument("sparse_mlp_depth", type=int, default=3),
        Argument("mp_rnn_type", type=str, default="gru"),
    ]

    def get_default_sparse_decoder_params(self) -> d:
        base_prms = super().get_default_sparse_decoder_params()

        mlp_network = SequentialParams(
            build_mlp_param_list(self.policy_in_size,
                                 [self.sparse_mlp_size] * self.sparse_mlp_depth + [self.sparse_policy_raw_out_size],
                                 dropout_p=self.sparse_dropout_p))

        return base_prms & d(
            cls=BasicModel,
            model_inputs=self.input_names,
            model_output=self.sparse_raw_out_name,
            network=mlp_network,
        )

    def get_default_mode_predictor_params(self) -> d:
        base_prms = super().get_default_mode_predictor_params()

        return base_prms & d(
            cls=ModePredictionRnnModel,
            model_inputs=self.input_names,
            model_output=self.mode_logit_out_name,
            mode_key=self.mode_key,
            mode_prob_out_name=self.mode_prob_out_name,

            rnn_type=self.mp_rnn_type,
            hidden_size=self.mode_head_size,
            mlp_size=0,
            rnn_depth=2,
            dropout_p=0,
        )


class HydraRNNActionDecoder(HydraSparseMLPActionDecoder, RNNActionDecoder):
    predefined_arguments = resolve_arguments(HydraSparseMLPActionDecoder, RNNActionDecoder)

    # redefine rnn check since this won't happen (RNNActionDecoder is the second import)
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)
        assert self.rnn_type in ['lstm', 'gru'], f"RNN type unimplemented: {self.rnn_type}"


class HydraMLPActionDecoder(HydraSparseMLPActionDecoder, MLPActionDecoder):
    predefined_arguments = resolve_arguments(HydraSparseMLPActionDecoder, MLPActionDecoder)
