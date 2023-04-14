"""
Interfaces for models to use in addition to base Model class

"""
import abc
from typing import Tuple, Dict

from attrdict import AttrDict as d

from muse.models.model import Model


class OnlineModel(Model, abc.ABC):
    """
    Compatible with online policy.

    """

    def init_memory(self, inputs: d, memory: d):
        memory.count = 0

    def pre_update_memory(self,  inputs: d, memory: d, kwargs: dict) -> Tuple[d, Dict]:
        return inputs, kwargs

    def post_update_memory(self, inputs: d, outputs: d, memory: d):
        memory.count += 1

    def online_forward(self, inputs: d, memory: d = None, **kwargs):
        if memory.is_empty():
            self.init_memory(inputs, memory)

        inputs, kwargs = self.pre_update_memory(inputs, memory, kwargs)

        out = self(inputs, **kwargs)

        self.post_update_memory(inputs, out, memory)

        return out
