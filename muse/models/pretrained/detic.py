import sys

import cv2
import numpy as np
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import os

import torch
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import timeit
from muse.experiments import logger


class DeticPredictor(torch.nn.Module):
    """
    NOTE: for viola, make sure detic from

    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    :
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg, input_format="BGR", permuted=False):
        super().__init__()
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            metadata = MetadataCatalog.get("proposal")
            metadata.thing_classes = [''] # Change here to try your own vocabularies!
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = torchvision.transforms.Resize(
            cfg.INPUT.MIN_SIZE_TEST, max_size=cfg.INPUT.MAX_SIZE_TEST
        )

        # what format will image be
        self.input_format = input_format

        # if true, don't need to reorder inputs
        self.permuted = permuted

        # what format will underlying predictor expect
        self.detic_input_format = cfg.INPUT.FORMAT

        assert self.input_format in ["RGB", "BGR"], self.input_format
        assert self.detic_input_format in ["RGB", "BGR"], self.detic_input_format

        logger.debug(f"Detic predictor: in={self.input_format}, detic_in={self.detic_input_format}")

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): images of shape (B, ..., H, W, C).
            permuted: if True, images come in shape (B, ..., C, H, W)
        Returns:
            predictions (AttrDict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        # convert to torch
        image = image.to(dtype=torch.float32)

        # -> (B, ... C, H, W)
        pre_shape = list(image.shape[:-3])
        image = image.view(-1, *image.shape[-3:])
        if not self.permuted:
            # augmentation expects in C,H,W order
            image = image.permute(0, 3, 1, 2)

        # Apply pre-processing to image.
        if self.detic_input_format != self.input_format:
            # whether the model expects BGR inputs or RGB
            image = torch.flip(image, [1])

        B, c, height, width = image.shape
        image = self.aug(image)

        with timeit('detectron_model'):
            # this is how detectron likes inputs...
            inputs = [
                {"image": image[i], "height": height, "width": width} for i in range(image.shape[0])
            ]
            predictions = self.model(inputs)

            # # need to format this better
            # predictions = AttrDict(
            #     scores=[dc['proposals'].scores for dc in out_dicts],
            #     pred_classes=[dc['proposals'].pred_classes for dc in out_dicts],
            #     objectness_logits=[dc['proposals'].objectness_logits for dc in out_dicts],
            #     boxes=[dc['proposals'].proposal_boxes.tensor for dc in out_dicts],
            # )

            # predictions.leaf_modify(lambda arr: split_dim(arr, 0, pre_shape))

        return predictions


def get_proposal_cfg(detic_root):
    detic_centernet_path = os.path.join(detic_root, 'third_party/CenterNet2/projects/CenterNet2')
    if detic_root not in sys.path:
        sys.path.insert(0, detic_root)
    if detic_centernet_path not in sys.path:
        sys.path.insert(0, detic_centernet_path)

    try:
        from centernet.config import add_centernet_config
        from detic.config import add_detic_config
    except ImportError as e:
        print("search path (missing centernet or detic): \n", sys.path)
        raise e

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)

    cfg.merge_from_file(f"{detic_root}/configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = f'{detic_root}/models/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth'
    cfg.MODEL.META_ARCHITECTURE = 'ProposalNetwork'
    cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 128
    # cfg.MODEL.CENTERNET.INFERENCE_TH = 0.1
    cfg.MODEL.CENTERNET.NMS_TH_TEST = 0.05
    # cfg.MODEL.DEVICE='cpu'
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True  # For better visualization purpose. Set to False for all classes.
    # cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3]
    # cfg.MODEL.RPN.IOU_THRESHOLDS = [0.7, 1.0]

    return cfg


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, input_format="BGR"):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DeticPredictor(cfg, input_format=input_format)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(torch.from_numpy(image)[None])[0]

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
            if "proposals" in predictions:
                instances = predictions["proposals"].to(self.cpu_device)
                instances.pred_boxes = instances.proposal_boxes
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            predictions = predictions[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)

        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(torch.from_numpy(frame)[None]))


if __name__ == '__main__':
    from muse.experiments.file_manager import FileManager
    from muse.experiments import logger
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--image_key', type=str, default='image')
    parser.add_argument('--input_format', type=str, default='BGR', choices=['RGB', 'BGR'])
    parser.add_argument('--first_n_images', type=int, default=1)
    parser.add_argument('--save_prefix', type=str, default="plots/detic/output", help="saves to [this]_[dataset]_[key]_[i]")
    args = parser.parse_args()

    DETIC_ROOT = os.path.join(FileManager.base_dir, '../Detic')
    vis = VisualizationDemo(get_proposal_cfg(DETIC_ROOT), input_format=args.input_format)

    # preds, vis_image = vis.run_on_image(cv2.imread("../desk.jpg"))
    # vis_image.save("desk_output.jpg")

    dataset_path = file_path_with_default_dir(args.dataset, FileManager.base_dir)

    assert os.path.exists(dataset_path)
    data = np.load(dataset_path, allow_pickle=True)

    imgs = data[args.image_key]

    logger.debug(f"Loaded images of shape: {imgs.shape}")

    data_name = os.path.basename(args.dataset)
    data_name = list(os.path.splitext(data_name))[0]

    img_key_name = args.image_key.replace('/', '-')

    out = []
    for i, img in enumerate(imgs[:args.first_n_images]):
        preds, vis_image = vis.run_on_image(img)
        path = args.save_prefix + f"_{data_name}_{img_key_name}_{i}.png"
        logger.debug(f"Writing to --> {path}")
        vis_image.save(path)


