from attrdict import AttrDict as d
from muse.policies.replay_policy import ReplayPolicy


export = d(
    cls=ReplayPolicy,
    demo_file="",  # fill this in
    action_names=['action'],
)
