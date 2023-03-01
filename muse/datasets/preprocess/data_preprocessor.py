import numpy as np

from muse.envs.env_spec import EnvSpec
from attrdict.utils import get_with_default
from attrdict import AttrDict


# TODO
class DataPreprocessor:
    def __init__(self, params: AttrDict, env_spec: EnvSpec, dataset=None):
        self._init_params_to_attrs(params)
        self._dataset = dataset  # optional
        self._env_spec = env_spec

    def _init_params_to_attrs(self, params: AttrDict):
        self._name: str = get_with_default(params, "name", "generic_preprocessor")

        # (datadict, onetime, ep_idx, self) -> (datadict, onetime, changed_keys)
        self._episode_preproc_fn = get_with_default(params, "episode_preproc_fn", None)

    def forward(self, dataset, datadict, onetime_datadict, split_indices, **kwargs):
        """
        Takes a datadict, a onetime_datadict, split_indices (numpy), and does some work on it.

        :return: all the above, modified.
        """
        if len(split_indices) == 0:
            return datadict, onetime_datadict, split_indices

        if self._episode_preproc_fn is not None:
            start = 0
            i = 0
            all_ch_ep = []
            all_ch_ep_onetime = []
            new_splits = []
            all_keys = datadict.list_leaf_keys() + onetime_datadict.list_leaf_keys()
            while i < len(split_indices):
                end = split_indices[i]

                new_ep, new_ep_onetime, changed_keys = self._episode_preproc_fn(datadict.leaf_apply(lambda arr: arr[start:end]),
                                                                   onetime_datadict.leaf_apply(
                                                                       lambda arr: arr[i:i + 1]), i)
                new_ep_onetime.leaf_assert(lambda arr: len(arr) == 1)
                new_splits.append(len(new_ep['done']))
                if changed_keys is None:
                    changed_keys = list(all_keys)
                else:
                    assert set(all_keys).issubset(changed_keys) or new_splits[-1] == (end - start), \
                        "Changed keys must include all keys if episode length changed!"
                all_ch_ep.append(new_ep < changed_keys)
                all_ch_ep_onetime.append(new_ep_onetime < changed_keys)

                start = end
                i += 1

            new_datadict = datadict & AttrDict.leaf_combine_and_apply(all_ch_ep, np.concatenate)
            new_onetime_datadict = onetime_datadict & AttrDict.leaf_combine_and_apply(all_ch_ep_onetime, np.concatenate)
            return new_datadict, new_onetime_datadict, np.cumsum(new_splits)

        else:
            raise NotImplementedError

    @property
    def name(self):
        return self._name
#
#
#
# class TransformAugmentation(DataAugmentation):
#     def _init_params_to_attrs(self, params: AttrDict):
#         super(TransformAugmentation, self)._init_params_to_attrs(params)
#         self.transforms: AttrDict = get_with_default(params, "transforms")
