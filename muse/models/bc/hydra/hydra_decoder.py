import torch
from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.models.basic_model import BasicModel
from muse.models.bc.action_decoders import ActionDecoder, RNNActionDecoder, MLPActionDecoder
from muse.utils.abstract import Argument, resolve_arguments
from muse.utils.general_utils import timeit
import torch.nn.functional as F

from muse.utils.param_utils import get_dist_cap, SequentialParams, build_mlp_param_list, get_dist_out_size


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

        # TODO Argument("use_mode_predictor", action="store_true"),

        # if not separate mode prediction, here are the args that define it.
        Argument("decoder_inter_size", type=int, default=128),
        Argument("split_head_layers", type=int, default=2),
        Argument("mode_head_size", type=int, default=128),
        Argument("action_head_size", type=int, default=128),
    ]

    def _preload_params(self, params):
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

    def _init_params_to_attrs(self, params):
        self.sparse_action_names = params['sparse_action_names']
        self.mode_key = get_with_default(params, "mode_key", "mode")
        self.mode_logit_out_name = get_with_default(params, "mode_logit_out_name", "mode_logits")
        self.mode_prob_out_name = get_with_default(params, "mode_prob_out_name", "mode_probs")

        self.sparse_raw_out_name = get_with_default(params, "sparse_raw_out_name", "sparse_policy_raw")

        super()._init_params_to_attrs(params)

    def _init_action_caps(self):
        super()._init_action_caps()
        # cap for the sparse model
        self.sparse_action_cap = get_dist_cap(self.sparse_use_policy_dist, self.sparse_use_tanh_out,
                                              num_mix=self.sparse_policy_num_mix, sig_min=self.sparse_policy_sig_min,
                                              sig_max=self.sparse_policy_sig_max)
        self.sparse_action_cap = self.sparse_action_cap.to_module_list(as_sequential=True).to(self.device)

    def _init_setup(self):
        super()._init_setup()

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

    def forward(self, inputs, preproc=True, postproc=True, training=True, sparse_kwargs=None,
                mp_kwargs=None, horizon_idx=None, **decoder_kwargs):
        """
        Unlike base ActionDecoder, runs the sparse model as well, and also parses the mode and action.

        TODO support for separate mode predictor

        Parameters
        ----------
        inputs: AttrDict
        preproc
        postproc
        training
        sparse_kwargs: kwargs for sparse decoder
        mp_kwargs: TODO kwargs for mode predictor
        horizon_idx
        decoder_kwargs

        Returns
        -------

        """
        inputs = inputs.leaf_copy()

        if preproc:
            inputs = self._preproc_fn(inputs)

        sparse_kwargs = sparse_kwargs or {}

        # run inner model
        with timeit('decoder'):
            decoder_outs = self.decoder(inputs, timeit_prefix="decoder/", **decoder_kwargs)

        with timeit('mode_action_heads'):
            # compute mode from inner model output partial embedding
            embedding = decoder_outs[self.policy_raw_out_name]

            mp_logit = self.mode_head(embedding)
            decoder_outs[self.policy_raw_out_name] = self.action_head(embedding)

        out = decoder_outs.leaf_copy()

        # cap the end of the policy (action cap)
        out[self.policy_raw_out_name] = self.action_cap(decoder_outs[self.policy_raw_out_name])
        out[self.mode_prob_out_name] = F.softmax(mp_logit, dim=-1)
        out[self.mode_logit_out_name] = mp_logit
        out[self.mode_key] = torch.argmax(mp_logit, dim=-1, keepdim=True)

        # run sparse model
        with timeit('sparse_decoder'):
            sp_inputs = inputs
            if horizon_idx is not None:
                # only run mode predictor on whatever horizon_idx is
                sp_inputs = sp_inputs.leaf_apply(lambda arr: arr[:, horizon_idx, None])
            sparse_outs = self.sparse_decoder(sp_inputs, timeit_prefix="sparse_decoder/", **sparse_kwargs)

        # cap the end of the sparse policy (sparse_decoder)
        sparse_outs[self.sparse_raw_out_name] = self.sparse_action_cap(sparse_outs[self.sparse_raw_out_name])
        out.combine(sparse_outs)

        # parse the action names from either vector or distribution.
        ac_dc = self.parse_raw_action(self.env_spec, out[self.policy_raw_out_name], self.action_names,
                                      use_mean=self.use_policy_dist_mean, sample_cat=self.policy_sample_cat)

        # parse the sparse names from either vector or distribution.
        spac_dc = self.parse_raw_action(self.env_spec, out[self.sparse_raw_out_name],
                                        self.sparse_action_names, use_mean=self.sparse_use_policy_dist_mean,
                                        sample_cat=self.sparse_policy_sample_cat)

        out.combine(ac_dc)
        out.combine(spac_dc)

        return self._postproc_fn(inputs, out) if postproc else out

    @property
    def all_action_names(self):
        return self.action_names + self.sparse_action_names


class HydraSparseMLPActionDecoder(HydraActionDecoder):
    """
    Sparse decoder implemented as MLP.
    """

    predefined_arguments = HydraActionDecoder.predefined_arguments + [
        Argument("sparse_dropout_p", type=float, default=0),
        Argument("sparse_mlp_size", type=int, default=128),
        Argument("sparse_mlp_depth", type=int, default=3),
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


class HydraRNNActionDecoder(HydraSparseMLPActionDecoder, RNNActionDecoder):
    predefined_arguments = resolve_arguments(HydraSparseMLPActionDecoder, RNNActionDecoder)

    # redefine rnn check since this won't happen (RNNActionDecoder is the second import)
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)
        assert self.rnn_type in ['lstm', 'gru'], f"RNN type unimplemented: {self.rnn_type}"


class HydraMLPActionDecoder(HydraSparseMLPActionDecoder, MLPActionDecoder):
    predefined_arguments = resolve_arguments(HydraSparseMLPActionDecoder, MLPActionDecoder)
