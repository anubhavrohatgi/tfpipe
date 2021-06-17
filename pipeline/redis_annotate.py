import tensorflow as tf
import numpy as np
from PIL import Image

from pipeline.pipeline import Pipeline
from core.utils import read_class_names, convert_redis


class RedisAnnotate(Pipeline):
    """ Pipeline task that formats prediction annotations for Redis publishing. """

    def __init__(self, dst, iou_thresh, score_thresh, classes):
        self.dst = dst
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.num_classes = len(read_class_names(classes))

        super().__init__()

    def map(self, data):
        self.annotate_predictions(data)

        return data

    def annotate_predictions(self, data):
        boxes, scores = data["predictions"]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_thresh,
            score_threshold=self.score_thresh
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                     valid_detections.numpy()]

        annotated_output = convert_redis(
            data["image_id"], data["image"].shape, self.num_classes, pred_bbox)

        data[self.dst] = annotated_output
