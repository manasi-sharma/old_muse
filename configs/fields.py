import os


class Field:
    """
    Specify a value in a config that depends on another value, with some output type

    A field will be only changeable through changing another value (rigidly tied together)
    """

    def __init__(self, anchor: str, mod_fn=lambda x: x):
        self.anchor = anchor
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
        # if name == self.anchor:
        #     lookup_dc = global_params
        # else:
        #     lookup_dc = params & global_params

        # global lookup name
        anchor_path = os.path.normpath(os.path.join(path, self.anchor))

        if name == self.anchor:
            # special case where we only look up in global if anchor conflicts with our name
            anchor_value = global_params[self.anchor]
        else:
            # put params in global params temporarily for lookup
            gparams = params & global_params.leaf_copy()
            gparams[path] = params

            if self.anchor in gparams.leaf_keys():
                # look with a relative (local) path first
                anchor_value = gparams[self.anchor]
            elif anchor_path in gparams.leaf_keys():
                # look with an absolute path next
                anchor_value = gparams[anchor_path]
            else:
                raise ValueError(f"Anchor {self.anchor} missing from params!")

        assert not isinstance(anchor_value, Field), f"Cannot anchor to another field ({self.anchor})!"

        return self.mod_fn(anchor_value)
