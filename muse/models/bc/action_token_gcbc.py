import torch

from muse.models.bc.gcbc import TransformerGCBC
from muse.models.bc.lmp import play_helpers
from muse.utils.abstract import Argument
from muse.utils.general_utils import timeit

from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.utils.torch_utils import concatenate, combine_dims, unsqueeze_n, split_dim
from muse.models.layers.common import TemporalPositionalEncoding


class ActionTokenTransformerGCBC(TransformerGCBC):
    """
    Utilizes a single action embedding token per step to predict the action.

    (1) Encodes H length sequence into tokens (B x H*C x n_embed)
    (2) Inserts in H state embeddings (B x H x (C+1) x n_embed)
    (3) adds broadcased action token to start of each time axis (B x H x (1 + C+1) x n_embed)
    (4) runs the transformer and extracts the first token element per step
    (5) loss
    """

    predefined_arguments = TransformerGCBC.predefined_arguments + [
        Argument('full_pos_enc', action='store_true')
    ]

    def _inner_model_params_to_attrs(self, params):
        super()._inner_model_params_to_attrs(params)

        self.token_sequence_name = get_with_default(params, "token_sequence_name", "token_sequence")

        # how many tokens come out of vision model, not including state token or action token
        self.tokens_per_step = get_with_default(params, "tokens_per_step", 0)
        self.true_tokens_per_step = self.tokens_per_step + 2

        if not self.vision_encoder_params.is_empty():
            assert self.true_tokens_per_step > 2, "Cannot have zero output tokens from visual encoder!"

        # now the input is the same as the embedding dim
        self.cfg.input_size = self.n_embed
        # the horizon length is longer since images, states, and the action are encoded as tokens
        self.cfg.block_size = self.horizon * self.true_tokens_per_step
        # transformer will use special casual mask (causal on blocks, rather than each token)
        self.cfg.causal_block_size = self.true_tokens_per_step
        # if a full temporal embedding is needed, we let the transformer implement the encoding
        self.cfg.use_temporal_embd = self.full_pos_enc

    def _update_policy_in_from_vision(self):
        # do NOT change the policy inputs to account for vision, we will handle all this in forward()
        pass

    def _init_setup(self):
        super()._init_setup()

        # TODO allow a final projection layer before output
        self.inner_projection = None
        
        # project the state features to a token
        self.state_projection = torch.nn.Linear(self.policy_in_size, self.n_embed)

        # random action token, learnable
        self.action_token = torch.nn.Parameter(torch.randn(self.n_embed))

        if not self.full_pos_enc:
            # generate a self.horizon length temporal embedding function
            self.pos_enc = TemporalPositionalEncoding(input_shape=(self.true_tokens_per_step, self.n_embed))

    def _get_inner_model(self) -> d:
        inner_model = super()._get_inner_model()
        # modify inner model with the correct inputs,
        inner_model.params.combine(d(
            model_inputs=[self.token_sequence_name],
            horizon=self.horizon * self.true_tokens_per_step,
        ))
        return inner_model

    def forward(self, inputs, preproc=True, postproc=True, training=True, skip_vision=False, **kwargs):
        inputs = inputs.leaf_copy()

        if preproc:
            inputs = self._preproc_fn(inputs)

        out = d()

        # combine (state + embed) into (B, H, D)
        state_vector = concatenate((inputs > self.novision_policy_in_names).leaf_apply(lambda arr: arr.to(dtype=torch.float32)),
                                   self.novision_policy_in_names, dim=-1)
        
        # now project into (B, H, 1, n_embed)
        projected_state_vector = self.state_projection(state_vector).unsqueeze(-2)

        # add in the action token (1, 1, 1, n_embed),
        expanded_action = unsqueeze_n(self.action_token, 3, 0).expand(*projected_state_vector.shape[:2], -1, -1)

        tokens = [expanded_action, projected_state_vector]

        # if vision model, add this in to the tokens
        if self.vision_model is not None:
            if self.encoder_out_name not in inputs.leaf_keys():
                assert not skip_vision, f"Vision is required since {self.encoder_out_name} was missing from input, but skip_vision=True!"
                
                # run vision to get embeddings
                with timeit('vision_model'):
                    embed = self.vision_model(inputs, timeit_prefix="vision_model/")
                    out.vision_out = embed
                inputs = inputs & embed
        
            # shape will be (B, H, C, n_embed)
            embed_vector = inputs[self.encoder_out_name]

            expected_shape = [self.horizon, self.tokens_per_step, self.n_embed]
            assert list(embed_vector.shape[1:]) == expected_shape, f"Non-batched shape must be {expected_shape} but was {embed_vector.shape[1:]}"

            # concatenate to (B, H, C+1, n_embed), adding on state vector
            tokens.append(embed_vector)

        # (B, H, (C+2), n_embed)
        tokenized = torch.cat(tokens, dim=2)

        # generate pos embedding (this will do across H elements, not across H*(C+2)), enable full_... for the latter
        if not self.full_pos_enc:
            # add in (1, H, 1, n_embed)
            tokenized = tokenized + self.pos_enc(tokenized).unsqueeze(0).unsqueeze(-2)

        # (B, H*(C+2), n_embed)
        tokenized_with_action = combine_dims(tokenized, 1, 2)

        # inner model only sees this long context sequence.
        inner_model_inputs = d.from_dict({self.token_sequence_name: tokenized_with_action})

        # run inner model, which will output (B, H*(C+2), out_dim)
        with timeit('inner_model'):
            inner_outs = self.inner_model(inner_model_inputs, timeit_prefix="inner_model/", **kwargs)
            
            out.inner_outs = inner_outs
            
            # keep just the last channel dim, since this corresponds to the action tokens
            inner_outs = inner_outs.leaf_apply(lambda arr: split_dim(arr, 1, [self.horizon, self.true_tokens_per_step])[:, :, 0])
            
            # then project it to the GMM size, if not already there.
            if self.inner_projection is not None:
                inner_outs[self.policy_raw_out_name] = self.inner_projection(inner_outs[self.policy_raw_out_name])
            
            # cap it off
            inner_outs[self.policy_raw_out_name] = self.inner_model_cap(inner_outs[self.policy_raw_out_name])

        out.combine(inner_outs)
        # parse the sparse names from either vector or distribution.
        out = self.parse_model_output_fn(self.env_spec, inputs, out,
                                         self.policy_raw_out_name, self.action_names,
                                         use_mean=self.use_policy_dist, sample_cat=self.sample_cat)

        return self._postproc_fn(inputs, out) if postproc else out


    def get_default_preproc_fn(self):
        # reusing some stuff to get the goal and preprocess this.
        return play_helpers.get_gcbc_preproc_fn(not self.use_goal, self.use_final_goal, device=self.device,
                                                POLICY_NAMES=[self.token_sequence_name],
                                                POLICY_GOAL_STATE_NAMES=[])
