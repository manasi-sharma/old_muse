from attrdict import AttrDict as d

from muse.datasets.samplers.sampler import SequentialSampler
from muse.datasets.np_sequence_dataset import NpSequenceDataset

export = d(
    cls=NpSequenceDataset,
    file='/tmp/null.npz',
    output_file='/tmp/null.npz',
    save_every_n_steps=0,
    index_all_keys=True,
    initial_load_episodes=0,
    load_episode_range=[0.0, 0.9],
    load_from_base=False,
    capacity=1000000,
    mmap_mode=None,
    asynchronous_get_batch=False,
    frozen=True,
    allow_padding=False,
    pad_end_sequence=True,
    load_ignore_prefixes=[],
    sampler=d(
        cls=SequentialSampler,
        params=d(
            shuffle=True,
        ),
    ),
    data_preprocessors=[],
)
