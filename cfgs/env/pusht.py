from muse.envs.simple.pusht_env import PushTEnv
from attrdict import AttrDict as d

export = d(
    cls=PushTEnv,
    legacy=False, 
    block_cog=None, damping=None,
    render_action=True,
    render_size=96,
    reset_to_state=None
)
