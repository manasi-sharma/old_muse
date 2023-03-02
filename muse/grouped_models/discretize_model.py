from muse.grouped_models.grouped_model import GroupedModel
from muse.models.discretize import Discretize
from muse.utils.general_utils import timeit
from attrdict import AttrDict as d
from attrdict.utils import get_with_default


class DiscretizeModel(GroupedModel):

    required_models = [
        "discretize",
        "model",
    ]

    def _init_params_to_attrs(self, params: d):
        super(DiscretizeModel, self)._init_params_to_attrs(params)

        # if True, will call discretize model before
        self._disc_in = get_with_default(params, "discretize_inputs", False)
        self._in_inverse = get_with_default(params, "inputs_to_continuous", False)

        # if True, will call discretize model after
        self._disc_out = get_with_default(params, "discretize_outputs", True)
        self._out_inverse = get_with_default(params, "outputs_to_continuous", False)

        if hasattr(self._model, "online_postproc_fn"):
            self.online_postproc_fn = self._model.online_postproc_fn

    def _init_setup(self):
        assert isinstance(self._discretize, Discretize)

    def forward(self, inputs, training=False, preproc=True, postproc=True, **kwargs):
        """

        Will discretize/continu-ize optionally before or after model.

        :param inputs:
        :param training:
        :param preproc:
        :param postproc:
        :param kwargs:
        :return:
        """

        if self._disc_in:
            with timeit("discretize_in"):
                inputs = self._discretize(inputs, training=training, inverse=self._in_inverse)

        with timeit("discretize_model"):
            out = self._model(inputs, training=training, preproc=preproc, postproc=postproc, **kwargs)

        if self._disc_out:
            with timeit("discretize_out"):
                out = self._discretize(out, training=training, inverse=self._out_inverse)

        return out

    @property
    def discretize(self):
        return self._discretize