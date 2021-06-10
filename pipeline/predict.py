import tensorflow as tf
import numpy as np
from cv2 import resize
from tensorflow.python.saved_model import tag_constants

from pipeline.pipeline import Pipeline


class Predict(Pipeline):
    """ The pipeline task for single-process predicting. """

    def __init__(self, weights: str, framework: str = "tf", size: int = 416):
        self.size = size
        self.weights = weights
        self.framework = framework

        # if framework == 'tflite':
        #     pass  # come back if we need tflite
        # else:
        #     self.model = tf.saved_model.load(
        #         weights, tags=[tag_constants.SERVING])
        #     self.predictor = self.model.signatures['serving_default']

        super().__init__()

    def is_working(self):
        """ Returns False because predictions are done linearly. """

        return False

    def map(self, data):
        self.make_predictions(data)

        return data

    def make_predictions(self, data):
        images = [resize(image, (self.size, self.size)) /
                  255.0 for image in data["image"]]
        images = np.asanyarray(images).astype(np.float32)
        images = tf.constant(images)

        self.model = tf.saved_model.load(
            self.weights, tags=[tag_constants.SERVING])
        infer = self.model.signatures['serving_default']

        predictions = infer(images)
        data["predictions"] = predictions

    def cleanup(self):
        pass
