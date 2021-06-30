import tensorflow as tf
from tensorflow.python.eager.monitoring import BoolGaugeCell
from tensorflow.python.util.deprecation import _PRINTED_WARNING

from tfpipe.core.utils import load_config, filter_boxes
from tfpipe.core.yolo import YOLO, decode
from tfpipe.pipeline.pipeline import Pipeline


class CreateModel(Pipeline):
    """ Pipeline task for creating the model. """

    def __init__(self, input_size: int, classes: str, framework: str,
                 model_name: str, is_tiny: bool):
        self.input_size = input_size
        self.classes = classes
        self.framework = framework
        self.model_name = model_name
        self.is_tiny = is_tiny

    def map(self, _):
        """ Creates and returns the model. """

        input_layer = tf.keras.layers.Input(
            (self.input_size, self.input_size, 3))

        predictions = self.get_predictions(input_layer)

        model = tf.keras.Model(input_layer, predictions)

        return model

    def get_predictions(self, input_layer):
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = load_config(
            self.classes, self.model_name, self.is_tiny)
        feature_maps = YOLO(input_layer, NUM_CLASS,
                            self.model_name, self.is_tiny)

        bbox_tensors, prob_tensors = [], []
        if self.is_tiny:
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    output_tensors = decode(
                        fm, self.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                else:
                    output_tensors = decode(
                        fm, self.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
        else:
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    output_tensors = decode(
                        fm, self.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                elif i == 1:
                    output_tensors = decode(
                        fm, self.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                else:
                    output_tensors = decode(
                        fm, self.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
        pred_bbox = tf.concat(bbox_tensors, axis=1)
        pred_prob = tf.concat(prob_tensors, axis=1)
        if self.framework == 'tflite':
            pred = (pred_bbox, pred_prob)
        else:
            # boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=0.2, input_shape=tf.constant([self.input_size, self.input_size]))
            # pred = tf.concat([boxes, pred_conf], axis=-1)
            pred = (pred_bbox, pred_prob)

        return pred
