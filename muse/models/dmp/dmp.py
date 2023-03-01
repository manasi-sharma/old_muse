import numpy as np
import scipy.interpolate as interp
import torch
from sklearn.linear_model import Ridge
from typing import List

from muse.experiments import logger
from muse.models.model import Model
from muse.models.rnn_model import RnnModel
from attrdict import AttrDict
from attrdict.utils import get_with_default, get_or_instantiate_cls
from muse.utils.torch_utils import pad_dims, to_numpy, to_torch, broadcast_dims


class DMPModel(Model):
    """
    produces a forcing term to be added on a spring damper DMP system
    """

    def _init_params_to_attrs(self, params):
        # This model contains weights for a gmm,
        self.inputs = params["dmp_inputs"]
        self.haptic_inputs = get_with_default(params, "dmp_haptic_inputs", [])
        self.num_mix = get_with_default(params, "num_mix", 10)
        self.num_dmps = int(self.env_spec.dim(self.inputs))
        self.haptic_dim = int(self.env_spec.dim(self.haptic_inputs))
        logger.debug(f"Initializing DMP Model with dim = {self.num_dmps}, hdim = {self.haptic_dim}.")

        # online inputs
        # self.online_inputs = get_with_default(params, "online_inputs", [])
        # self.online_output = get_with_default(params, "online_output", None)
        self.online_model = get_with_default(params, "online_model", AttrDict())
        # self.concat_dim = int(params.get("concat_dim", -1))
        # self.concat_dtype = params.get("concat_dtype", torch.float32)
        self.online_model_rnn_hidden_name = None
        if not self.online_model.is_empty():
            logger.debug("DMPModel: Instantiating online model..")
            # this will be a module, to call forward
            self.online_model = get_or_instantiate_cls(self.online_model, None, Model, constructor=lambda cls, prm: cls(prm, self.env_spec, self._dataset_train))
            if isinstance(self.online_model, RnnModel):
                self.online_model_rnn_hidden_name = self.online_model.hidden_name

        # converts from haptic (B x H) -> dmps (B x N) for feedback
        self.sensor_transform = get_with_default(params, "sensor_transform", np.eye(self.haptic_dim, self.num_dmps))
        assert list(self.sensor_transform.shape) == [self.haptic_dim, self.num_dmps]
        self.sensor_transform = to_torch(self.sensor_transform, device=self.device, check=True)

        # self.haptic_dim = get_with_default(params, "haptic_dim", 0)
        self.tau = params["tau"]  # forcing term output
        # steps
        self.dt = params["dt"]  # forcing term output steps
        self.num_steps = int(self.tau / self.dt)
        self._difference_fn = get_with_default(params, "difference_fn", lambda a, b: a - b)

        # this gets added on before imitation estimation, that extra oomph
        self.imitation_goal_delta = get_with_default(params, "imitation_goal_delta", np.zeros(self.num_dmps), map_fn=np.asarray)

        # TODO
        self.forcing_term_trainable = get_with_default(params, "forcing_term_trainable", True)
        self.goal_trainable = get_with_default(params, "goal_trainable", self.forcing_term_trainable)
        self.haptic_trainable = get_with_default(params, "haptic_trainable", False)

        self.stretch_haptic_before_imitate = get_with_default(params, "stretch_haptic_before_imitate", True)

        self.linear_forcing_decay = get_with_default(params, "linear_forcing_decay", False)

        # D = 25.
        self.alpha = torch.tensor(np.broadcast_to(get_with_default(params, "alpha", 25.0), (self.num_dmps,)),
                                  device=self.device, requires_grad=False)
        # = K / D = 100.0 / 25.0 = 4.0
        self.beta = get_with_default(params, "beta", self.alpha / 4.,
                                     map_fn=lambda arr: torch.tensor(np.broadcast_to(arr, (self.num_dmps,)),
                                                                     device=self.device, requires_grad=False))
        # = H
        self.alpha_h = torch.tensor(np.broadcast_to(get_with_default(params, "alpha_h", 0.), (self.num_dmps,)),
                                    device=self.device, requires_grad=False)
        # Canonical system
        self.alpha_t = torch.tensor(get_with_default(params, "alpha_t", 8.0), requires_grad=False)

        self.use_haptic = len(self.haptic_inputs) > 0

        self.r = Ridge(alpha=get_with_default(params, "ridge_penalty", 0.), fit_intercept=False)

        ## PARAMETERS
        self.weights = torch.nn.Parameter(torch.randn((self.num_mix, self.num_dmps), device=self.device),
                                          requires_grad=self.forcing_term_trainable)
        self.goal = torch.nn.Parameter(torch.randn((self.num_dmps,), device=self.device),
                                       requires_grad=self.goal_trainable)
        # this wil be interpolated to the correct length
        self.haptic_targets = torch.nn.Parameter(torch.randn((self.num_steps, self.haptic_dim), device=self.device),
                                                 requires_grad=self.haptic_trainable)

        self.mean = torch.nn.Parameter(torch.zeros((self.num_dmps,), device=self.device),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones((self.num_dmps,), device=self.device),
                                      requires_grad=False)

        # state
        self._cached_forward = None

    def phase(self, n_steps, t=None):
        """The phase variable replaces explicit timing.
        It starts with 1 at the beginning of the movement and converges
        exponentially to 0.
        """
        phases = torch.exp(-self.alpha_t * torch.linspace(0, 1, n_steps).to(self.device))
        if t is None:
            return phases
        else:
            return phases[t]

    def _features(self, num_mixtures, s):
        if num_mixtures == 0:
            raise NotImplementedError
        elif num_mixtures == 1:
            return torch.tensor([1.0], device=self.device)
        # 1D list of the feature centers
        # c = torch.linspace(1 - 0.5/num_mixtures, 0.5/num_mixtures, num_mixtures).to(self.device)
        c = self.phase(num_mixtures)
        # 1D widths of each kernel are the standard dev.
        h = c[1:] - c[:-1]
        # kernel radius (0.5 / sigma^2)
        h = 0.5 / torch.cat([h, h[-1:]], dim=-1) ** 2
        # each s (1D) compared with each c

        # NM x T, each mixture has an activation at each step
        phi = torch.exp(-h[:, None] * (s[None] - c[:, None]) ** 2)
        if self.linear_forcing_decay:
            # stable log
            log_s = torch.where(s < 1e-11, 0*s, s.log())
            phi_s = phi * (log_s[None] / self.alpha_t)
        else:
            phi_s = phi * s[None]
        # still NM x T
        feature_activations = phi_s / phi.sum(0, keepdim=True)  # normalize

        # return T x NM
        return torch.transpose(feature_activations, 0, 1)

    def forcing_term(self, tau, num_mixtures, s):
        # T x num_mix
        f_all = self._features(num_mixtures, s)

        # now, T x num_dmps
        activations = f_all.matmul(self.weights)

        return activations

    def online_forward(self, inputs, training=False, preproc=True, postproc=True, run_offline=False, **kwargs):
        """
        Equivalent to underlying model forward, and returns current best forcing term as well

        :param inputs: (B x H x ..)
        :param training:
        :param preproc:
        :param postproc:
        :param kwargs:
        :return:
        """
        assert isinstance(self.online_model, Model), "Online model must be specified"
        # caching
        if run_offline or self._cached_forward is None or self._cached_forward.is_empty():
            # if model forward has not been called yet, call it to compute all the targets
            B = inputs.get_one().shape[0]
            self._cached_forward = self.offline_forward(inputs).leaf_apply(lambda arr: broadcast_dims(arr, [0], [B]))

        # add in latest cached forward results
        inputs = inputs & self._cached_forward

        # for example might return actions to execute online
        online_out = self.online_model.forward(inputs, training=training, preproc=preproc, postproc=postproc, **kwargs)

        online_out = inputs & online_out

        return online_out

    def offline_forward(self, inputs, training=False, preproc=True, postproc=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x H x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """

        # open loop forcing term
        ft = self.forcing_term(self.tau, self.num_mix, self.phase(self.num_steps))
        # normalize
        ft = ft * self.std + self.mean
        # ft[:] = 0
        # goal is relative here (1 x H x ...)
        out = AttrDict(forcing_term=ft[None], goal=self.goal[None])
        if self.haptic_targets.numel() > 0:
            out['haptic_target'] = self.haptic_targets[None]

        return out

    def forward(self, inputs, run_online=True, run_offline=True, **kwargs):
        if run_online:
            return self.online_forward(inputs, run_offline=run_offline, **kwargs)
        else:
            assert run_offline, "One run option must be specified at least"
            return self.offline_forward(inputs, **kwargs)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def loss(self, inputs, outputs, training=True, ret_dict=False, **kwargs):
        raise NotImplementedError

    def compute_haptic_targets(self, hap_inps: AttrDict, as_torch=True):
        """
        Takes episodes of haptic info, upsamples, averages, and then downsamples back
        :param hap_inps:
        :param as_torch:
        :return:
        """
        hap_outs = AttrDict()
        for key, data in hap_inps.leaf_items():
            assert isinstance(data, List)
            # getting the average f/t to use as the initial
            max_ep_len = max(len(ep) for ep in data)
            interp_fns = [interp.interp1d(np.arange(len(ep)) / (len(ep) - 1), to_numpy(ep, check=True), axis=0)
                          for ep in data]
            upsampled = [fn(np.arange(max_ep_len) / (max_ep_len - 1)) for fn in interp_fns]
            avg_data = np.median(np.stack(upsampled), axis=0)  # mean trace over episodes
            interp_fn = interp.interp1d(np.arange(max_ep_len) / (max_ep_len - 1), avg_data, axis=0)

            hap_outs[key] = []
            for ep in data:
                arr = interp_fn(np.arange(len(ep)) / (len(ep) - 1))
                if as_torch:
                    hap_outs[key].append(to_torch(arr, device=self.device))
                else:
                    hap_outs[key].append(arr)

        return hap_outs

    def imitate(self, inputs, haptic_inputs, haptic_outputs):
        """
        inputs are all lists of tensors
        :param inputs: for each dmp key, list of tensors of shape (T, dim)
        :param outputs:
        :return: None
        """

        # (N x n_dmp), (N,)
        dc = self.imitate_forces(inputs, haptic_inputs, haptic_outputs)
        F, s, g, x0, idxs = dc.get_keys_required(['forcing_term', 'cs', 'goal', 'x0', 'start_idxs'])
        ht = dc << "haptic_target"
        # (N x num_mix)
        design = self._features(self.num_mix, s)
        # N x num_mix*num_dmps)
        # design = torch.repeat_interleave(design, self.num_dmps, dim=1)

        # lr = Ridge(alpha=1.0, fit_intercept=False)
        Fnp = to_numpy(F)
        design_np = to_numpy(design)
        g_np = to_numpy(g)
        x0_np = to_numpy(x0)
        # Fmean = Fnp.mean(axis=0, keepdims=True)
        # Fstd = Fnp.std(axis=0, keepdims=True)
        # Fnorm = (Fnp - Fmean) / Fstd
        # self.r.fit(design_np, Fnorm)
        # self.r.fit(design_np, Fnp)
        # w = self.r.coef_.T

        # decoupled in each dimension
        all_weights = []
        for d in range(self.num_dmps):
            self.r.fit(design_np, Fnp[:, d])
            all_weights.append(self.r.coef_)
        w = np.stack(all_weights, axis=-1)

        self.weights.data.copy_(to_torch(w, device=self.device))
        # self.mean.data.copy_(to_torch(Fmean.reshape(-1), device=self.device))
        # self.std.data.copy_(to_torch(Fstd.reshape(-1), device=self.device))

        # goal is just the "average" goal
        assert len(idxs) < len(g) == len(x0)
        assert len(set(idxs)) == len(idxs)
        g_np = g_np[idxs]
        x0_np = x0_np[idxs]

        avg_goal_delta = self._difference_fn(g_np, x0_np).mean(axis=0)
        self.goal.data.copy_(to_torch(avg_goal_delta, device=self.device, check=True))

        if self.use_haptic:
            ht_np = to_numpy(ht)
            # haptic target, first split into each ep
            if len(idxs) == 1:
                eps = [ht_np]
            else:
                # split along the start idxs
                eps = np.split(ht_np, idxs[1:], axis=0)
            v = np.arange(self.num_steps) / (self.num_steps - 1)
            if self.stretch_haptic_before_imitate:
                # stretch all individually
                interp_fns = [interp.interp1d(np.arange(len(ep)) / (len(ep) - 1), ep, axis=0) for ep in eps]
                full_ht = [fn(v) for fn in interp_fns]
            else:
                # truncate, then stretch all
                min_ep_len = min(len(ep) for ep in eps)
                trunc = [ep[:min_ep_len] for ep in eps]
                v0 = np.arange(min_ep_len) / (min_ep_len - 1)
                full_ht = [interp.interp1d(v0, ep, axis=0)(v) for ep in trunc]

            # all elements will be (dmp_steps, haptic_dim)
            avg_ht = np.stack(full_ht, axis=0).mean(axis=0)
            self.haptic_targets.copy_(to_torch(avg_ht, device=self.device))

        # prediction residual in normalized space
        return F - design.matmul(self.weights)

    def compute_feedback(self, X: torch.Tensor, haptic_features: torch.Tensor, last_haptic_features: torch.Tensor, haptic_targets: torch.Tensor):
        """
        :param X: (B, dim)
        :param haptic_features: (B, dim)
        :param last_haptic_features: (B, dim)
        :param haptic_targets: (B, dim)
        :return:
        """
        return (haptic_targets - haptic_features).matmul(self.sensor_transform) * self.alpha_h[None]

    def imitate_forces(self, inputs, haptic_inputs, haptic_outputs):
        """
        inputs are all lists of tensors
        :param inputs: for each dmp key, list of tensors of shape (T, dim)
        :param outputs:
        :return:
        """
        assert isinstance(inputs, AttrDict)

        # returns the forces required to follow that trajectory

        dmp_dict = inputs.leaf_filter_keys(self.inputs)
        haptic_dict = haptic_inputs.leaf_filter_keys(self.haptic_inputs)
        haptic_out_dict = haptic_outputs.leaf_filter_keys(self.haptic_inputs)

        # each entry
        def join_elementwise(dc, order):
            ls_of_ls = list(dc.get_keys_required(order))
            assert isinstance(ls_of_ls, List), [ls_of_ls]
            assert all(isinstance(ls, List) for ls in ls_of_ls), [ls_of_ls]
            assert len(set(len(ls) for ls in ls_of_ls)) == 1, [ls_of_ls]

            out = []
            for tup in zip(*ls_of_ls):
                out.append(torch.cat(list(tup), dim=-1))

            return out

        # List of tensors of shape (Ti, dim)
        X = join_elementwise(dmp_dict, self.inputs)

        if self.use_haptic:
            haptic_X = join_elementwise(haptic_dict, self.haptic_inputs)
            haptic_T = join_elementwise(haptic_out_dict, self.haptic_inputs)

        horizons = [x.shape[0] for x in X]
        ep_starts = [0]
        for h in horizons[:-1]:
            ep_starts.append(ep_starts[-1] + h)
        # goals at every time step, for varying time length sequences
        x0 = [x[None, 0].expand(x.shape[0], -1) for x in X]
        g = [x[None, -1].expand(x.shape[0], -1) for x in X]
        tau = [torch.tensor([self.dt * h] * h, device=self.device) for h in horizons]
        s = [self.phase(h) for h in horizons]
        # positional targets (x) use the user defined diff fn
        Xd = [pad_dims(self._difference_fn(x[1:], x[:-1]), [0], [1], after=False, delta=True) / self.dt for i, x in enumerate(X)]
        # velocity differences don't use the difference fn
        Xdd = [pad_dims(x[1:] - x[:-1], [0], [1], after=False, delta=True) / self.dt for i, x in enumerate(Xd)]
        # list -> tensor
        X = torch.cat(X, dim=0)
        Xd = torch.cat(Xd, dim=0)
        Xdd = torch.cat(Xdd, dim=0)
        g = torch.cat(g, dim=0) + to_torch(self.imitation_goal_delta[None], device=self.device)
        x0 = torch.cat(x0, dim=0)
        tau = torch.cat(tau, dim=0)
        s = torch.cat(s, dim=0)

        if self.use_haptic:
            last_haptic_X = [pad_dims(hx[..., :-1], [-1], [1], after=False, delta=True) for hx in haptic_X]
            haptic_X = torch.cat(haptic_X, dim=0)
            haptic_T = torch.cat(haptic_T, dim=0)
            last_haptic_X = torch.cat(last_haptic_X, dim=0)
            assert list(haptic_X.shape) == [X.shape[0], self.haptic_dim]
            assert list(haptic_T.shape) == [X.shape[0], self.haptic_dim]

        assert self.num_dmps == X.shape[-1]

        # spring damper
        F = (tau**2)[:, None] * Xdd - self.alpha * (self.beta * self._difference_fn(g, X) - tau[:, None] * Xd)

        if self.use_haptic:
            # haptic
            zeta = self.compute_feedback(X, haptic_X, last_haptic_X, haptic_T)
            F = F - zeta

        out = AttrDict(
            forcing_term=F,
            cs=s,
            goal=g,
            x0=x0,
            start_idxs=ep_starts,
        )
        if self.use_haptic:
            out["haptic_target"] = haptic_T
        return out

    def load_statistics(self, dd=None):
        if isinstance(self.online_model, Model):
            self.online_model.load_statistics(dd)