from cv2 import cvtColor, COLOR_RGB2BGR
import tensorflow as tf
import numpy as np
from PIL import Image

from pipeline.pipeline import Pipeline
from pipeline.utils.utils import draw_bbox


class AnnotateImage(Pipeline):
    """ Pipeline task for image annotation. """

    def __init__(self, dst, iou_thresh, score_thresh):
        self.dst = dst
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

        super().__init__()

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image

        self.annotate_predictions(data)

        return data

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return

        # print(data["predictions"])
        # for i, (key, value) in enumerate(data["predictions"].items()):
        #     print(value.shape, key, i)

        for predictions in data["predictions"].values():
            boxes = predictions[:, :, 0:4]
            conf = predictions[:, :, 4:]

        boxes = tf.reshape(boxes, (boxes.shape[0], -1, 1, 4))
        scores = tf.reshape(
            conf, (boxes.shape[0], -1, tf.shape(conf)[-1]))

        result = tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_thresh,
            score_threshold=self.score_thresh
        )

        def f(x): return x.numpy()

        boxes, scores, classes, valid_detections = map(f, result)

        # print(classes)
        # print(valid_detections)

        annotated_images = list()
        for i, image in enumerate(data["image"]):
            pred_bbox = [boxes[i:i+1], scores[i:i+1],
                         classes[i:i+1], valid_detections[i:i+1]]
            annotated_image = draw_bbox(image, pred_bbox)
            annotated_image = Image.fromarray(annotated_image.astype(np.uint8))

            annotated_images.append(annotated_image)

        data[self.dst] = annotated_images
