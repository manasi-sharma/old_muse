from muse.datasets.preprocess.data_preprocessor import DataPreprocessor
from muse.models.bc.hydra.helpers import get_mode_to_mask_preproc_fn
from muse.utils.python_utils import AttrDict as d


params = d(
    cls=DataPreprocessor,
    params=d(
        name="mask_preprocessor",
        episode_preproc_fn=get_mode_to_mask_preproc_fn("mode", "mask", meq=0, skip_last_n=10),
    )
)
