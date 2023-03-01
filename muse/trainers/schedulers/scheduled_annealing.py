import numpy as np


class AnnealScheduler:
    def __init__(self, start_val, finish_val, T):
        self.start_val = start_val
        self.finish_val = finish_val
        self.T = T

    def get_val(self, t):
        raise NotImplementedError


class LinearAnnealScheduler(AnnealScheduler):
    def __init__(self, start_val, finish_val, T):
        super(LinearAnnealScheduler, self).__init__(start_val, finish_val, T)
        self._delta = (self.start_val - self.finish_val) / T

    def get_val(self, t):
        return max(self.finish_val, self.start_val - self._delta * t)


class ExponentialAnnealScheduler(AnnealScheduler):
    def __init__(self, start_val, finish_val, time_len):
        super(ExponentialAnnealScheduler, self).__init__(start_val, finish_val, time_len)
        assert finish_val > 0, "final value must be greater than 0 for exponential decay!"
        self._exponential_scale = (-1 * time_len) / np.log(self.finish_val)

    def get_val(self, t):
        return min(self.start_val, max(np.exp((-t) / self._exponential_scale), self.finish_val))
