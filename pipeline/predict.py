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

        if framework == 'tflite':
            pass  # come back if we need tflite
        else:
            self.model = tf.keras.models.load_model(weights)
            self.predict = self.model.predict

        super().__init__()

    def is_working(self):
        """ Returns False because predictions are done linearly. """

        return False

    def map(self, data):
        self.make_predictions(data)

        return data

    def make_predictions(self, data):
        image = [resize(data["image"], (self.size, self.size)) / 255.0]
        image = np.asanyarray(image).astype(np.float32)
        image = tf.constant(image)


        predictions = self.predict(image)
        data["predictions"] = predictions

    def cleanup(self):
        pass
