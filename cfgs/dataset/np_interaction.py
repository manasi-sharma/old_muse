from attrdict import AttrDict as d

from muse.datasets.np_interaction_dataset import NpInteractionDataset

# make sure to set `parse_interaction_bounds_from_episode_fn` to use this.
export = d(
    cls=NpInteractionDataset,

    # interaction loading params
    pad_to_horizon=True,
    sample_pre_window=True,
    sample_goals=True,
    sample_goal_as_contact_end=True,

    # basic params
    file='/tmp/null.npz',
    output_file='/tmp/null.npz',
    batch_size=100,
    horizon=10,
    save_every_n_steps=0,
    index_all_keys=True,
    initial_load_episodes=0,
    load_from_base=False,
    capacity=1000000,
    mmap_mode=None,
    asynchronous_get_batch=False,
    frozen=True,
    allow_padding=False,
    pad_end_sequence=True,
    load_ignore_prefixes=[],
    data_preprocessors=[],
)
