from typing import List

import torch
from torch.nn import functional as F
from attrdict import AttrDict as d
from einops import rearrange, reduce

from muse.models.bc.action_decoders import ActionDecoder
from muse.models.bc.gcbc import BaseGCBC
from muse.models.diffusion.dp import DiffusionPolicyModel
from muse.models.diffusion.dp_cond_unet import ConditionalUnet1D
from muse.utils.abstract import Argument
from muse.utils.loss_utils import write_avg_per_last_dim
from muse.utils.param_utils import SequentialParams, LayerParams
from muse.utils.torch_utils import combine_then_concatenate


class DiffusionConvActionDecoder(ActionDecoder):
    """
    Diffusion Action Decoder, with Convolutional backend
     - uses DiffusionPolicyModel under the hood.

    The online model portion is very similar to the TransformerActionDecoder
    """

    predefined_arguments = ActionDecoder.predefined_arguments + [
        Argument('n_action_steps', type=int, required=True,
                 help='number of action steps in the future to predict (action horizon)'),
        Argument('n_obs_steps', type=int, required=True,
                 help='how many obs steps to condition on'),

    ]

    def _init_params_to_attrs(self, params: d):
        super()._init_params_to_attrs(params)

    def get_default_decoder_params(self) -> d:
        assert not self.use_policy_dist, "Policy distribution not implemented for diffusion models! " \
                                         "Must be deterministic."
        base_prms = super().get_default_decoder_params()

        # default parameters for generator (these would be overriden if any decoder params are specified)
        generator = d(
            cls=ConditionalUnet1D,
            # action size
            input_dim=self.policy_raw_out_size,
            # obs_dim * num_obs_steps
            global_cond_dim=self.policy_in_size * self.n_obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        # default parameters for scheduler
        import diffusers.schedulers.scheduling_ddpm as diff_sched
        noise_scheduler = d(
            cls=diff_sched.DDPMScheduler,
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',  # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
            clip_sample=True,  # required when predict_epsilon=False
            prediction_type='epsilon',  # or sample
        )

        return base_prms & d(
            cls=DiffusionPolicyModel,
            obs_inputs=self.input_names,
            action_dim=self.policy_raw_out_size,
            raw_out_name=self.policy_raw_out_name,
            generator=generator,
            noise_scheduler=noise_scheduler,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            obs_as_global_cond=True,
        )

    @property
    def online_input_names(self) -> List[str]:
        return [] if self.vision_encoder_params.is_empty() else [self.encoder_out_name]

    def init_memory(self, inputs: d, memory: d):
        super().init_memory(inputs, memory)
        # list of inputs, shape (B x 1 x ..), will be concatenated later
        memory.input_history = [(inputs > (self.state_names + self.goal_names + self.online_input_names))
                                for _ in range(self.n_obs_steps)]

        # avoid allocating memory again
        memory.alloc_inputs = d.leaf_combine_and_apply(memory.input_history,
                                                       lambda vs: torch.cat(vs, dim=1))

    def pre_update_memory(self, inputs: d, memory: d, kwargs: dict):
        inputs, kwargs = super().pre_update_memory(inputs, memory, kwargs)

        # add new inputs (the online ones), maintaining sequence length
        memory.input_history = memory.input_history[1:] + [inputs > (self.state_names + self.goal_names +
                                                                     self.online_input_names)]

        def set_vs(k, vs):
            # set allocated array, return None
            torch.cat(vs, dim=1, out=memory.alloc_inputs[k])

        # assign to alloc_inputs
        d.leaf_combine_and_apply(memory.input_history, set_vs, pass_in_key_to_func=True)

        # replace inputs with the history.
        return memory.alloc_inputs, kwargs


class DiffusionGCBC(BaseGCBC):
    """
    Diffusion-based GCBC, just implements the diffusion style loss.


    """

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True,
             ret_dict=False, meta=d(), **kwargs):

        # provide a timestep in kwargs for the decoder.
        bsz = inputs.get_one().shape[0]
        if 'action_decoder_kwargs' not in kwargs:
            kwargs['action_decoder_kwargs'] = {}
        if 'decoder_kwargs' not in kwargs['action_decoder_kwargs']:
            kwargs['action_decoder_kwargs']['decoder_kwargs'] = {}
        kwargs['action_decoder_kwargs']['decoder_kwargs']['timestep'] = torch.randint(
            0, self.decoder.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device
        ).long()
        # also provide the raw actions.
        kwargs['action_decoder_kwargs']['decoder_kwargs']['raw_action'] = \
            combine_then_concatenate(inputs, self.action_decoder.action_names, dim=2).to(dtype=torch.float32)

        # model forward
        model_outputs = self(inputs, **kwargs)

        """ 
        Decoder output should contain...
            - noise
            - noisy_trajectory
            - recon_trajectory
            - trajectory
            - condition_mask
        """
        dout = model_outputs['action_decoder/decoder']

        # compute loss mask
        loss_mask = ~dout['condition_mask']

        pred_type = self.decoder.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = dout['noise']
        elif pred_type == 'sample':
            target = dout['trajectory']
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(dout.recon_trajectory, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)

        # only write if there is a non-horizon last dimension here.
        if writer is not None:
            write_avg_per_last_dim(loss, i=i, writer=writer, writer_prefix=writer_prefix + "policy_loss/mse_dim_")

        loss = loss.mean()

        if ret_dict:
            return d(
                loss=loss
            )
        else:
            return loss


