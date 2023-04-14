from attrdict import AttrDict as d

from cfgs.exp_hvs.square import bc_rnn
from cfgs.model import dp_conv_1d
from configs.fields import Field as F
from muse.utils.general_utils import dc_skip_keys

export = dc_skip_keys(bc_rnn.export, 'model') & d(
    horizon=16,
    model=dp_conv_1d.export & d(
        device=F('device'),
        action_decoder=d(
            horizon=F('horizon'),
        )
    ),
)
