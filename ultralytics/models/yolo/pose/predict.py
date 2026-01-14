# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops
from ultralytics.utils.postprocess_utils import decode_bbox, decode_kpts, separate_outputs_decode  # DG


class PosePredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific warnings
        for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"

    def postprocess(self, preds, img, orig_imgs, **kwargs):  # DG
        """Post-process predictions and return a list of Results objects. #DG

        This method handles both standard and separate_outputs format predictions.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.
        """
        save_feats = getattr(self, "_feats", None) is not None
        if self.separate_outputs:  # Hardware-optimized export with separated outputs #DG
            pred_order, nkpt = separate_outputs_decode(
                preds, self.args.task, self.model.kpt_shape[0] * self.model.kpt_shape[1]
            )
            pred_decoded = decode_bbox(pred_order, img.shape, self.device)
            nc = pred_decoded.shape[1] - 4
            kpt_shape = (nkpt.shape[-1] // 3, 3)
            kpts_decoded = decode_kpts(
                pred_order, img.shape, torch.permute(nkpt, (0, 2, 1)), kpt_shape, self.device, bs=1
            )
            pred_order = torch.cat([pred_decoded, kpts_decoded], 1)
            preds = nms.non_max_suppression(
                pred_order,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
                nc=nc,
            )
        else:
            preds = nms.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
                nc=len(self.model.names),
                return_idxs=save_feats,
            )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction, including keypoints.

        Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
                the number of detections, K is the number of keypoints, and D is the keypoint dimension.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and
                keypoints.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
