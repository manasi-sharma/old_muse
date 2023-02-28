from attrdict import AttrDict as d

from muse.metrics.metric import ExtractMetric
from muse.metrics.tracker import BufferedTracker


export = d(
    env_train=d(
        returns=BufferedTracker(d(
            buffer_len=50,
            buffer_freq=0,
            time_agg_fn=lambda k, a, b: max(a, b),
            metric=ExtractMetric(name='returns', key='reward', source=1, reduce_fn='none'),
            tracked_names=['returns'],
        )),
    ),
    env_holdout=d(
        returns=BufferedTracker(d(
            buffer_len=50,
            buffer_freq=0,
            time_agg_fn=lambda k, a, b: max(a, b),  # sum rewards,
            metric=ExtractMetric(name='returns', key='reward', source=1, reduce_fn='none'),
            tracked_names=['returns'],
        )),
    ),
)
