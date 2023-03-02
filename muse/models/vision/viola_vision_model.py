import os

import numpy as np
from muse.models.vision.vision_core import PaddedCropRandomizer
import torch
import torch.nn.functional as F

from muse.experiments import logger
from muse.experiments.file_manager import FileManager
from muse.models.basic_model import BasicModel
from muse.models.pretrained.detic import DeticPredictor, get_proposal_cfg
from muse.models.vision import vision_encoders as ve
from muse.utils.general_utils import timeit
from muse.utils.torch_utils import combine_dims, split_dim, to_numpy
from muse.models.layers.common import SpatialProjection, BBoxTrueSinusoidalPositionEncodingFactor, RoIAlignWrapper
from muse.models.layers.functional import CombineDim

from attrdict import AttrDict
from attrdict.utils import get_with_default


# change this based on Detic install location, default is to use the same as VIOLA
DETIC_ROOT = os.path.join(FileManager.base_dir, '../VIOLA/third_party/Detic')


class VIOLAVisionModel(BasicModel):
    """
    VIOLA Vision model, which computes embeddings using image and ego image.

    Make sure not to put any extra randomizers in params.network. VIOLA handles that separately.
    """

    def _init_params_to_attrs(self, params):
        params.network = get_with_default(params, "network", ve.get_reset18_partial_embedding_layer())
        super()._init_params_to_attrs(params)
        # self.net should output partial embedding (still in image shape)

        assert len(self.inputs) == 2, "Accepts two images as inputs!"
        assert self.call_separate, "The two images must be processed separately"

        # config for detic
        self.rpn_predictor_cfg = get_with_default(params, 'rpn_predictor_cfg', get_proposal_cfg(DETIC_ROOT))

        # default, use image for RPN
        self.rpn_image_index = get_with_default(params, "rpn_image_index", 0)
        self.bbox_key = get_with_default(params, 'bbox_key', "centernet_bbox")
        self.preload_rpn = get_with_default(params, 'preload_rpn', True)
        self.rpn_input_format = get_with_default(params, 'rpn_input_format', "BGR")
        self.rpn_train_preload_path = get_with_default(params, 'rpn_train_preload_path', None)
        self.rpn_holdout_preload_path = get_with_default(params, 'rpn_holdout_preload_path', None)

        # extracting boxes from rpn output
        self.rpn_score_output_key = get_with_default(params, 'rpn_score_output_key', "scores")
        self.rpn_bbox_output_key = get_with_default(params, 'rpn_bbox_output_key', "boxes")

        # the projection size for the rpn features
        self.embed_size = get_with_default(params, "embed_size", 64)

        self.top_k = get_with_default(params, "top_k", 20)

        # for self.align
        self.alignment_output_size = list(get_with_default(params, "alignment_output_size", (6, 6)))

        # ordered strictly
        self.img_shape = self.env_spec.names_to_shapes[self.inputs[0]]
        self.ego_img_shape = self.env_spec.names_to_shapes[self.inputs[1]]
        
        self.downsample_frac = get_with_default(params, 'downsample_frac', 1.0)
        if self.downsample_frac < 1.0:
            self.downsampler = ve.get_downsampler(self.img_shape, self.downsample_frac)
            self.ego_downsampler = ve.get_downsampler(self.ego_img_shape, self.downsample_frac)
            self.img_shape = np.asarray(self.img_shape).copy()
            self.img_shape[:2] = np.round(self.img_shape[:2] * self.downsample_frac).astype(int)
            self.ego_img_shape = np.asarray(self.ego_img_shape).copy()
            self.ego_img_shape[:2] = np.round(self.ego_img_shape[:2] * self.downsample_frac).astype(int)
            logger.debug(f'Downsampling images by {self.downsample_frac}: to img={self.img_shape} and ego={self.ego_img_shape}')
            
        # augmentation
        self.crop_pad_len = 2
        self.img_randomizers = get_with_default(params, "img_randomizers", 
                                                [ve.get_padded_crop_randomizer(self.img_shape, self.crop_pad_len)])
        self.ego_img_randomizers = get_with_default(params, "ego_img_randomizers", 
                                                    [ve.get_padded_crop_randomizer(self.ego_img_shape, self.crop_pad_len)])

        # this is the shape post cropping, TODO make this better.
        if len(self.img_randomizers) > 0 and not isinstance(self.img_randomizers[0], PaddedCropRandomizer):
            raise NotImplementedError('Img Cannot use a non-padded crop randomizer bc it will change the size!')
        if len(self.ego_img_randomizers) > 0 and not isinstance(self.ego_img_randomizers[0], PaddedCropRandomizer):
            raise NotImplementedError('Ego Img cannot use a non-padded crop randomizer bc it will change the size!')

        self.img_h, self.img_w = self.img_shape[:2]

    def _init_setup(self):
        super()._init_setup()
        
        # what comes out of encoder, should be an image
        self.enc_out_c, self.enc_out_h, self.enc_out_w = ve.resnet18_compute_output_shape(self.img_shape, cut_last=4)
        self.ego_enc_out_c, self.ego_enc_out_h, self.ego_enc_out_w = ve.resnet18_compute_output_shape(self.ego_img_shape, cut_last=4)

        logger.debug(f"Assuming encoding of shape ({self.enc_out_c}, {self.enc_out_h}, {self.enc_out_w})")

        # TODO parameterize all of these.
        # extra layer for spatial + projection on images
        self.spatial_projection = SpatialProjection(
            (self.enc_out_c, self.enc_out_h, self.enc_out_w), num_kp=self.embed_size // 2, out_dim=self.embed_size)

        # likewise, but for projecting ego image
        if self.use_shared_params:
            self.ego_spatial_projection = self.spatial_projection
        else:
            self.ego_spatial_projection = SpatialProjection(
                (self.ego_enc_out_c, self.ego_enc_out_h, self.ego_enc_out_w), num_kp=self.embed_size // 2, out_dim=self.embed_size)

        # RPN, should not be loaded so we put it in a list (hacky)
        self.rpn_predictor_reference = [DeticPredictor(self.rpn_predictor_cfg, input_format=self.rpn_input_format, permuted=True)]
        self.rpn_predictor_reference[0].requires_grad_(False)

        # bbox encoding layers
        self.bbox_norm = BBoxTrueSinusoidalPositionEncodingFactor(channels=self.embed_size, pixel_var=1,
                                                                  factor_ratio=1.)

        # alignment btwn image and bboxes
        logger.debug(f"[VIOLA] ROI pooling using scale: {self.enc_out_h / self.img_h}")
        self.align = RoIAlignWrapper(output_size=self.alignment_output_size, spatial_scale=self.enc_out_h / self.img_h, sampling_ratio=-1)

        # project bounding boxes down to flat feature vectors.
        self.projection = torch.nn.Sequential(
            CombineDim(-3, 3),
            torch.nn.Linear(self.enc_out_c * int(np.prod(self.alignment_output_size)), self.embed_size)
        )

        # combining projected rpn features with bbox features
        self.bbox_position_embedding = lambda proj, nb: proj + nb

    @property
    def rpn_predictor(self):
        return self.rpn_predictor_reference[0]

    def _preproc_img(self, obs):
        # permute
        front_shape = obs.shape[:-3]
        obs = combine_dims(obs, 0, 2)
        obs = obs.permute((0, 3, 1, 2)).to(dtype=self.concat_dtype).contiguous()  # move channel to front
        
        return obs, front_shape

    def _postproc_arr(self, obs, front_shape):
        return split_dim(obs, 0, front_shape)

    def precompute_rpn_boxes(self, dataset, preload_path):
        loaded = False
        if preload_path is not None:
            try:
                all_boxes = np.load(preload_path)
                logger.debug(f"Loaded shape={all_boxes.shape}, from {preload_path}")
                loaded = True
            except OSError:
                logger.debug(f"Path {preload_path} does not exist. Computing manually")

        if not loaded:
            all_bbox = []
            B = 1
            self.eval()
            with torch.no_grad():
                for i in range(dataset.get_num_episodes()):
                    # (Hi x ...)
                    inputs, outputs = dataset.get_episode(i, None, split=True, torch_device=self.device)

                    for j in range(len(outputs['done']) // B):
                        # 1 x 1...
                        ins = inputs.leaf_apply(lambda arr: arr[None, B*j:B*(j+1)])
                        all_bbox.append(self(ins, do_rpn=True, rpn_only=True)[self.bbox_key])

                    logger.debug(f"ep {i} loaded")

            # return sum(H) x 4
            all_boxes = to_numpy(torch.cat(all_bbox, dim=1))[0]
            if preload_path is not None:
                logger.debug(f"Saving to {preload_path}")
                np.save(preload_path, all_boxes)

        assert list(all_boxes.shape) == [len(dataset), self.top_k, 4], [all_boxes.shape, len(dataset)]

        logger.debug(f"BBOX Min: {all_boxes.reshape(-1, 4).min(axis=0)}")
        logger.debug(f"BBOX Max: {all_boxes.reshape(-1, 4).max(axis=0)}")
        return all_boxes

    def pretrain(self, datasets_holdout=None):
        if self.preload_rpn:
            with timeit('pretrain/cache_rpn'):
                logger.debug("Loading RPN outputs for dataset_train...")
                bbox = self.precompute_rpn_boxes(self._dataset_train, self.rpn_train_preload_path)
                self._dataset_train.add_key_to_dataset(self.bbox_key, bbox)

            if datasets_holdout is not None and datasets_holdout:
                with timeit('preholdout/cache_rpn'):
                    logger.debug("Loading RPN outputs for dataset_holdout...")
                    bbox_h = self.precompute_rpn_boxes(datasets_holdout[0], self.rpn_holdout_preload_path)
                    datasets_holdout[0].add_key_to_dataset(self.bbox_key, bbox_h)

    def forward(self, inputs, training=False, preproc=True, postproc=True, timeit_prefix="viola_vision/",
                do_rpn=None, rpn_only=False, **kwargs):
        """ NOTE: slow when do_rpn=True """

        out = AttrDict()

        # Region Proposal
        if do_rpn is None:
            do_rpn = not inputs.has_leaf_key(self.bbox_key)

        assert do_rpn or not rpn_only, "cannot do RPN_ONLY if not doing rpn"

        # normalize
        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=self.concat_dtype)

        # preproc
        with timeit(f"{timeit_prefix}preproc"):
            if self._preproc_fn:
                inputs = self._preproc_fn(inputs)

        with torch.no_grad():
            # preproc (e.g. flip)
            img, ego_img = inputs.get_keys_required(self.inputs)
            img, front_shape = self._preproc_img(img)
            ego_img, _ = self._preproc_img(ego_img)
            
            if self.downsample_frac < 1.0:
                img = self.downsampler.forward_in(img)
                ego_img = self.ego_downsampler.forward_in(ego_img)

            if not rpn_only:
                # randomization on images
                for r in self.img_randomizers:
                    img = r.forward_in(img)
                for r in self.ego_img_randomizers:
                    ego_img = r.forward_in(ego_img)

            if do_rpn:
                with timeit("viola/rpn"):
                    # run RPN
                    self.rpn_predictor.eval()
                    in_img = [img, ego_img][self.rpn_image_index]
                    # need to run rpn in smaller increments at a time
                    B = min(in_img.shape[0], 10)
                    outputs = []
                    for j in range(in_img.shape[0] // B):
                        outputs.extend(self.rpn_predictor(in_img[B * j:B * (j+1)]))

                with timeit("viola/rpn_postproc"):
                    # go through batches, computing the valid ones
                    topk_boxes = []
                    for dc in outputs:
                        boxes = dc['proposals'].proposal_boxes
                        boxes = boxes[boxes.area() < (in_img.shape[-2] * in_img.shape[-1]) / 4]
                        boxes = boxes[boxes.area() > 4 * 4]
                        boxes = boxes.tensor
                        # pad zeros to the end
                        if len(boxes) < self.top_k:
                            topk_boxes.append(F.pad(boxes, (0, 0, 0, self.top_k - len(boxes)), 'constant', 0))
                        else:
                            topk_boxes.append(boxes[:self.top_k])

                    # top k boxes (B x K x ...)
                    bbox = torch.stack(topk_boxes)
            else:
                # get rpn bboxes from input, flatten first two dims
                bbox = combine_dims(inputs[self.bbox_key], 0, 2)

        if not rpn_only:
            # encode (partially) using net
            # run model and concatenate after (if calling separate)
            each_obs = [img, ego_img]
            with timeit(f'{timeit_prefix}forward'):
                out_ls = []
                for i, eo in enumerate(each_obs):
                    # index into module dict if separate
                    encoder_net_i = self.net if self.use_shared_params else self.net[i]
                    out_ls.append(encoder_net_i(eo))
                    assert len(out_ls[i].shape) == len(eo.shape), f"[VIOLAVision]: net {i} reduced the shape of the image!"

            # randomization on images (out)
            for r in self.img_randomizers[::-1]:
                out_ls[0] = r.forward_out(out_ls[0])
            for r in self.ego_img_randomizers[::-1]:
                out_ls[1] = r.forward_out(out_ls[1])

            # finishing projection for image and ego_image
            img_embed = self.spatial_projection(out_ls[0])
            ego_img_embed = self.ego_spatial_projection(out_ls[1])

            # pooling together the image and the bbox's
            aligned = self.align(out_ls[self.rpn_image_index], [b for b in bbox])

            # project the aligned features down
            projection = self.projection(aligned)

            # compute positional embedding
            normalized_bbox = self.bbox_norm(bbox)

            # combination of raw bbox info and projection info
            position_embedding_out = self.bbox_position_embedding(projection, normalized_bbox)

            # aggregate the image embedding with the positional one. this is a huge tensor
            # (n_embed + n_embed + K * n_embed)
            full_embedding = torch.cat([img_embed.unsqueeze(1), ego_img_embed.unsqueeze(1), position_embedding_out], dim=1)

            # outputs will be (B x H x c x h x w) for images/bboxes, and (B x H x D) for tensors
            out &= AttrDict.from_dict({
                f'aug_{self.inputs[0]}': self._postproc_arr(img, front_shape),
                f'aug_{self.inputs[1]}': self._postproc_arr(ego_img, front_shape),
                f'embed_{self.inputs[0]}': self._postproc_arr(img_embed, front_shape),
                f'embed_{self.inputs[1]}': self._postproc_arr(ego_img_embed, front_shape),
                f'position_embedding_out': self._postproc_arr(position_embedding_out, front_shape),
                self.output: self._postproc_arr(full_embedding, front_shape),
            })

        if do_rpn:
            out[self.bbox_key] = self._postproc_arr(bbox, front_shape)

        return self._postproc_fn(inputs, out) if self._postproc_fn else out
