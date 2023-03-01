from attrdict import AttrDict as d

from muse.datasets.hdf5_dataset import Hdf5Dataset
from muse.datasets.samplers.sampler import SequentialSampler

export = d(
    cls=Hdf5Dataset,
    file='/tmp/null.hdf5',
    cache=False,
    load_episode_range=[0.0, 0.9],
    load_from_base=False,
    pad_end_sequence=True,
    sampler=d(
        cls=SequentialSampler,
        params=d(
            shuffle=True,
        ),
    ),
)
