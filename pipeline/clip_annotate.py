import tensorflow as tf

from tfpipe.core.utils import read_class_names
from tfpipe.pipeline.pipeline import Pipeline

class ClipAnnotate(Pipeline):
    """ Pipeline task for video frame annotation. """

    def __init__(self, iou_thresh, score_thresh, meta, classes):
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

        # Metadata
        if self.meta:
            pass
        
        data["detection"] = bool(valid_detections[0]) # False if 0, True if nonzero

        # data[self.dst] = annotated_image