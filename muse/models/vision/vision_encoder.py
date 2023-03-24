import numpy as np

from muse.experiments import logger
from muse.models.basic_model import BasicModel
from muse.models.vision.vision_core import VisionCore


class VisionEncoder(BasicModel):

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        # for each image, compute the shape that will be returned.
        self.image_shapes = []
        self.flipped_image_shapes = []
        # shapes that come out of net for each image
        self.output_shapes = []
        # shapes that come out of net.randomizers for each image
        self.randomizer_in_output_shapes = []

        # vision core should be the first layer
        for i, n in enumerate(self.inputs):
            assert len(img_shape := list(self.env_spec.names_to_shapes[n])) == 3, \
                f"Expected image inputs! {n} was not!"
            # (H x W x C)
            self.image_shapes.append(img_shape)
            # (C x H x W)
            flipped_img_shape = img_shape[2:] + img_shape[:2]
            self.flipped_image_shapes.append(flipped_img_shape)

            if self.call_separate and not self.use_shared_params:
                net = self.net[i]
            else:
                net = self.net

            assert isinstance(net, VisionCore), \
                f"{str(net)} \n^ Above net must be VisionCore!"

            self.output_shapes.append(net.get_output_shape(flipped_img_shape))
            self.randomizer_in_output_shapes.append(net.get_randomizer_in_output_shape(flipped_img_shape))

        self._add_outputs_to_spec()

    def _add_outputs_to_spec(self):
        if self.output not in self.env_spec.all_names:
            # get the output size by concatenating the output shapes
            temp_arrs = [np.empty([1, 1] + list(sh)) for sh in self.output_shapes]
            cat_arr = np.concatenate(temp_arrs, axis=self.concat_dim)
            logger.warn(f"Adding {self.output} to spec, with shape: {cat_arr.shape[2:]}")
            self.env_spec.add_nsld(self.output, list(cat_arr.shape[2:]), (-np.inf, np.inf), np.float32)
