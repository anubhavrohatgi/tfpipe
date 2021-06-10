import tensorflow as tf
import numpy as np
from cv2 import cvtColor, COLOR_RGB2BGR
from PIL import Image

from pipeline.pipeline import Pipeline
from core.utils import read_class_names, draw_bbox


class AnnotateImage(Pipeline):
    """ Pipeline task for image annotation. """

    def __init__(self, dst, iou_thresh, score_thresh, classes):
        self.dst = dst
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.classes = read_class_names(classes)


        super().__init__()

    def map(self, data):
        self.annotate_predictions(data)

        return data

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]

        boxes = predictions[:, :, 0:4]
        conf = predictions[:, :, 4:]

        boxes = tf.reshape(boxes, (boxes.shape[0], -1, 1, 4))
        scores = tf.reshape(
            conf, (boxes.shape[0], -1, tf.shape(conf)[-1]))

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_thresh,
            score_threshold=self.score_thresh
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        annotated_image = draw_bbox(data["image"].copy(), pred_bbox, self.classes)
        annotated_image = Image.fromarray(annotated_image.astype(np.uint8))

        data[self.dst] = annotated_image
