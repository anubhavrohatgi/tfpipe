import tensorflow as tf

from tfpipe.pipeline.pipeline import Pipeline
from tfpipe.core.utils import read_class_names, draw_bbox, get_meta


def filter_boxes(box_xywh, scores, input_shape, score_threshold=0.4):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(
        scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(
        scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


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

        # filter
        boxes, scores = filter_boxes(boxes, scores, tf.constant([416, 416]))

        boxes = tf.reshape(boxes, (1, -1, 1, 4))
        scores = tf.reshape(scores, (1, -1, tf.shape(scores)[-1]))
        ####

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
                data["image"].shape, pred_bbox, self.classes)
            data["meta"] = metadata

        annotated_image = draw_bbox(
            data["image"].copy(), pred_bbox, self.classes)

        data[self.dst] = annotated_image
