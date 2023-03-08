from muse.datasets.preprocess.data_preprocessor import DataPreprocessor
from muse.models.bc.hydra.helpers import get_mode_to_mask_preproc_fn
from attrdict import AttrDict as d


export = d(
    cls=DataPreprocessor,
    name="mask_preprocessor",
    episode_preproc_fn=get_mode_to_mask_preproc_fn("mode", "mask", meq=0, skip_last_n=10),
)
