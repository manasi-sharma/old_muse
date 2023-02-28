

class Field:
    """
    Specify a value in a config that depends on another value, with some output type

    A field will be only changeable through changing another value (rigidly tied together)
    """

    def __init__(self, anchor: str, mod_fn=lambda x: x):
        self.anchor = anchor
        self.mod_fn = mod_fn

    def process(self, name, params, global_params):
        if name == self.anchor:
            lookup_dc = global_params
        else:
            lookup_dc = params & global_params

        assert self.anchor in lookup_dc.leaf_keys(), \
            f"Anchor {self.anchor} missing from params!"

        anchor_value = lookup_dc[self.anchor]

        assert not isinstance(anchor_value, Field), f"Cannot anchor to another field ({self.anchor})!"

        return self.mod_fn(anchor_value)
