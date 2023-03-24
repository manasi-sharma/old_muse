from typing import List

import torch
from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.experiments import logger
from muse.grouped_models.grouped_model import GroupedModel
from muse.models.basic_model import BasicModel
from muse.models.gpt.bet_layers import TransformerConfig
from muse.models.rnn_model import RnnModel
from muse.models.seq_model import SequenceModel
from muse.utils.abstract import Argument
from muse.utils.general_utils import timeit
from muse.utils.param_utils import get_dist_cap, LayerParams, SequentialParams, build_mlp_param_list, \
    get_dist_out_size
from muse.utils.torch_utils import unsqueeze_then_gather


class ActionDecoder(GroupedModel):
    """
    Decodes input names into actions.
    """

    required_models = ['decoder']

    predefined_arguments = GroupedModel.predefined_arguments + [
        # action space arguments
        Argument("use_policy_dist", action="store_true"),
        Argument("policy_num_mix", type=int, default=1),
        Argument("use_policy_dist_mean", action="store_true"),
        Argument("policy_sig_min", type=float, default=1e-5),
        Argument("policy_sig_max", type=float, default=1e3),
        Argument("use_tanh_out", action="store_true"),
        Argument("policy_sample_cat", action="store_true"),
    ]

    def _parse_models(self, params):
        self.input_names = params['input_names']
        self.action_names = params['action_names']

        # input size
        self.policy_in_size = params << 'policy_in_size'
        if self.policy_in_size is None:
            self.policy_in_size = self.env_spec.dim(self.input_names)

        # size of all actions stacked together
        self.policy_out_size = self.env_spec.dim(self.action_names)  # e.g., deterministic output
        # compute the raw size of the output of network
        self.policy_raw_out_size = get_dist_out_size(self.policy_out_size,
                                                     prob=self.use_policy_dist, num_mix=self.policy_num_mix)

        self.decoder_out_size = self.policy_raw_out_size

        # intermediate name for raw policy output
        self.policy_raw_out_name = get_with_default(params, "raw_out_name", "policy_raw")

        assert not self.normalize_inputs, "Input normalization should not be enabled for the decoder!"

        self._preload_params(params)

        super()._parse_models(params)

    def _preload_params(self, params):
        # get default params for grouped model before parsing but after reading params, allowing for override
        params.decoder = self.get_default_decoder_params() & get_with_default(params, "decoder", d())
        params.model_order = get_with_default(params, 'model_order', self.required_models)

    def _init_setup(self):
        super()._init_setup()

        # instantiate distribution caps for action space
        self._init_action_caps()

    def _init_action_caps(self):
        # cap for the inner model
        self.action_cap = get_dist_cap(self.use_policy_dist, self.use_tanh_out, num_mix=self.policy_num_mix,
                                       sig_min=self.policy_sig_min, sig_max=self.policy_sig_max)
        self.action_cap = self.action_cap.to_module_list(as_sequential=True).to(self.device)

    def get_default_decoder_params(self) -> d:
        """ Override this in sub-classes to specify default decoders (e.g., RNN or MLP)

        Optionally use parameters in self (e.g. Argument values)

        Returns
        -------

        """

        return d()

    def forward(self, inputs, preproc=True, postproc=True, **kwargs):
        if preproc:
            inputs = self._preproc_fn(inputs)

        # run inner model
        with timeit('decoder'):
            decoder_outs = self.decoder(inputs, timeit_prefix="decoder/", **self.get_kwargs('decoder', kwargs))
            # cap it off
            decoder_outs[self.policy_raw_out_name] = self.action_cap(decoder_outs[self.policy_raw_out_name])

        # parse the action names from either vector or distribution.
        action_dc = self.parse_raw_action(self.env_spec, decoder_outs[self.policy_raw_out_name],
                                          self.action_names, use_mean=self.use_policy_dist_mean,
                                          sample_cat=self.policy_sample_cat)
        out = decoder_outs & action_dc

        out.decoder = decoder_outs

        return self._postproc_fn(inputs, out) if postproc else out

    @property
    def all_action_names(self):
        return self.action_names

    @staticmethod
    def parse_raw_action(env_spec, raw_action, action_names, use_mean=False, sample_cat=False):
        """
        Parses raw_action (tensor) into keys (action_names)

        Parameters
        ----------
        env_spec
        raw_action: torch.Tensor or torch.distribution.Distribution
        action_names: List[str]
        use_mean: bool
            if a distribution, whether to use mean or sample
        sample_cat: bool
            for MixtureSameFamily specifically (GMM), sample from categorical only
            (use mean must be True)

        Returns
        -------

        """
        if isinstance(raw_action, torch.distributions.Distribution):
            # raw is a distribution
            # sample = raw.rsample()
            if use_mean:
                sample = raw_action.mean  # mean
            elif sample_cat:
                assert use_mean, "Cannot sample cat but not use mean!"
                assert isinstance(raw_action,
                                  torch.distributions.MixtureSameFamily), "raw must be mixture, but was: {raw}"
                mean = raw_action.component_distribution.mean  # .. x k x D
                _, max_idxs = raw_action.mixture_distribution.sample()  # ..
                sample = unsqueeze_then_gather(mean, max_idxs, dim=len(max_idxs.shape))  # .. x D
            else:
                sample = raw_action.rsample()  # sample
        else:
            sample = raw_action

        return env_spec.parse_view_from_concatenated_flat(sample, action_names)


class MLPActionDecoder(ActionDecoder):
    """
    MLP implementation for action decoder
    """

    predefined_arguments = ActionDecoder.predefined_arguments + [
        Argument("dropout_p", type=float, default=0),
        Argument("mlp_size", type=int, default=128),
        Argument("mlp_depth", type=int, default=3),
    ]

    def get_default_decoder_params(self) -> d:
        base_prms = super().get_default_decoder_params()

        mlp_network = SequentialParams(
            build_mlp_param_list(self.policy_in_size, [self.mlp_size] * self.mlp_depth + [self.decoder_out_size],
                                 dropout_p=self.dropout_p))

        return base_prms & d(
            cls=BasicModel,
            model_inputs=self.input_names,
            model_output=self.policy_raw_out_name,
            network=mlp_network,
        )


class RNNActionDecoder(ActionDecoder):
    """
    RNN action decoder, with support for LSTM or GRU
    """
    arg_hidden_size = Argument("hidden_size", type=int, default=128)

    predefined_arguments = ActionDecoder.predefined_arguments + [
        Argument("dropout_p", type=float, default=0),
        arg_hidden_size,
        Argument("policy_size", type=int, default=arg_hidden_size),
        Argument("rnn_depth", type=int, default=2),
        Argument("rnn_type", type=str, default="gru"),
    ]

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)
        assert self.rnn_type in ['lstm', 'gru'], f"RNN type unimplemented: {self.rnn_type}"

    def get_default_decoder_params(self) -> d:
        base_prms = super().get_default_decoder_params()

        # mlp after the rnn (will be removed if policy_size=0, special case)
        mlp_after_rnn_dims = [self.policy_size, self.policy_size, self.decoder_out_size]
        if self.policy_size == 0:
            mlp_after_rnn_dims = [self.decoder_out_size]  # no mlp.

        mlp_after_rnn_network = SequentialParams(build_mlp_param_list(self.hidden_size, mlp_after_rnn_dims,
                                                                      dropout_p=self.dropout_p))

        return base_prms & d(
            cls=RnnModel,
            model_inputs=self.input_names,
            model_output=self.policy_raw_out_name,
            rnn_output_name="rnn_output_policy",
            hidden_name="hidden_policy",
            rnn_before_net=True,
            tuple_hidden=self.rnn_type == "lstm",
            recurrent_network=LayerParams(self.rnn_type, input_size=self.policy_in_size,
                                          hidden_size=self.hidden_size, num_layers=self.rnn_depth,
                                          bidirectional=False, batch_first=True,
                                          dropout=self.dropout_p),
            # outputs (B x Seq x Hidden)
            network=mlp_after_rnn_network,
        )


class TransformerGCBC(ActionDecoder):
    """
    Transformer ActionDecoder, with support for GPT-like arch
    """

    predefined_arguments = ActionDecoder.predefined_arguments + [
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

    def get_default_decoder_params(self) -> d:
        # transformer
        transformer_layer = LayerParams(self.transformer_type, config=self.cfg)

        # mlp after transformer, to project to policy_out_size
        mlp_after_rnn_layers = build_mlp_param_list(self.transformer_out_size, [self.decoder_out_size],
                                                    dropout_p=self.dropout_p)

        # full net
        transformer_net = SequentialParams([transformer_layer] + mlp_after_rnn_layers)

        return d(
            cls=SequenceModel,
            model_inputs=self.input_names,
            model_output=self.policy_raw_out_name,
            horizon=self.horizon,
            # we will pass in the encoder out name online as an additional input to aggregate.
            online_inputs=self.state_names + self.goal_names + self.online_input_names,
            network=transformer_net
        )
