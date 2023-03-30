import os
from typing import List, Union
from muse.utils.general_utils import strlist


class Field:
    """
    Specify a value in a config that depends on other value(s), with some output type

    A field will be only changeable through changing another value (rigidly tied together)

    TODO: If dtype is specified, and that type is in [float, str, int, bool], it will also support command line override.
    """

    def __init__(self, anchors: Union[str, List[str]], mod_fn=lambda *x: x[0]):
        # process anchor into a list
        if isinstance(anchors, str):
            anchors = [anchors]

        self.anchors = strlist(anchors)
        assert len(self.anchors) >= 1, "Must have at least 1 anchor passed in!"

        # default mod_fn just selects the first anchor
        self.mod_fn = mod_fn

    def process(self, name, params, global_params, path):
        """

        Parameters
        ----------
        name:
            name of field
        params:
            Local params
        global_params: AttrDict
            Root params
        path: str
            Where "params" will go in global_params

        Returns
        -------
        value: The processed value after looking up self.anchor and applying self.mod_fn

        """
        # temporary lookup dict
        gparams = params & global_params.leaf_copy()

        # lookup each anchor
        anchor_values = []
        for anchor in self.anchors:
            # global lookup name
            anchor_path = os.path.normpath(os.path.join(path, anchor))

            if name == anchor:
                # special case where we only look up in global if anchor conflicts with our name
                anchor_value = global_params[anchor]
            else:
                # put params in global params temporarily for lookup
                gparams[path] = params

                if anchor in gparams.leaf_keys():
                    # look with a relative (local) path first
                    anchor_value = gparams[anchor]
                elif anchor_path in gparams.leaf_keys():
                    # look with an absolute path next
                    anchor_value = gparams[anchor_path]
                else:
                    raise ValueError(f"Anchor {anchor} missing from params!")

            assert not isinstance(anchor_value, Field), f"Cannot anchor to another field ({anchor})!"

            anchor_values.append(anchor_value)

        # mod on the list of values
        return self.mod_fn(*anchor_values)

