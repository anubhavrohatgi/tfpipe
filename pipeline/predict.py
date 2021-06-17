import tensorflow as tf
import numpy as np
from cv2 import imread

from core.config import cfg
from pipeline.pipeline import Pipeline


class Predict(Pipeline):
    """ The pipeline task for single-process predicting. """

    def __init__(self, args):
        self.size = args.size
        self.iou_thresh = args.iou
        self.score_thresh = args.score

        if args.framework == 'tflite':
            pass  # come back if we need tflite
        else:
            self.model = tf.keras.models.load_model(args.weights)
            self.predict = self.model.predict

        super().__init__()

    def is_working(self):
        """ Returns False because predictions are done linearly. """

        return False

    def map(self, data):
        self.make_predictions(data)

        return data

    def make_predictions(self, data):
        if data == Pipeline.Empty:
            return Pipeline.Skip

        predictions = self.predict(data["predictions"])

        conf = predictions[:, :, 4:]

        data["predictions"] = (tf.reshape(predictions[:, :, 0:4], (1, -1, 1, 4)),
                               tf.reshape(conf, (1, -1, tf.shape(conf)[-1])))

    def cleanup(self):
        pass
