import tensorflow as tf

from tfpipe.pipeline.pipeline import Pipeline
from tfpipe.core.utils import read_class_names, draw_bbox, get_meta, filter_boxes, fbox


class AnnotateImage(Pipeline):
    """ Pipeline task for image annotation. """

    def __init__(self, dst, iou_thresh, score_thresh, meta, classes):
        self.dst = dst
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.meta = meta
        self.classes = read_class_names(classes)

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

        # Metadata
        if self.meta:
            metadata = get_meta(
                data["image"].shape, data["image_path"], data["image_id"], pred_bbox, self.classes)
            data["meta"] = metadata

        annotated_image = draw_bbox(
            data["image"].copy(), pred_bbox, self.classes)

        data[self.dst] = annotated_image
