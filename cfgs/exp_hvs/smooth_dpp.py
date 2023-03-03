from muse.datasets.preprocess.data_preprocessor import DataPreprocessor
from muse.models.bc.hydra.helpers import get_mode_smooth_preproc_fn
from muse.utils.python_utils import AttrDict as d


params = d(
    cls=DataPreprocessor,
    params=d(
        name="mode_smooth_preprocessor",
        episode_preproc_fn=get_mode_smooth_preproc_fn("mode", "smooth_mode"),
    )
)