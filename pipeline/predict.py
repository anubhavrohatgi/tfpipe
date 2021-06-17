import tensorflow as tf
import cv2
import numpy as np

from core.config import cfg
from pipeline.pipeline import Pipeline


class Predict(Pipeline):
    """ The pipeline task for single-process predicting. """

    def __init__(self, args):
        if args.framework == 'tflite':
            pass  # come back if we need tflite
        else:
            print("Loading model...")
            self.model = tf.keras.models.load_model(args.weights)
            self.predict = self.model.predict

        print("Inferencing Test Image...")
        image = cv2.cvtColor(cv2.imread(cfg.INIT_IMG), cv2.COLOR_BGR2RGB)
        image = [cv2.resize(image, (args.size, args.size)) / 255.0]
        image = np.asanyarray(image).astype(np.float32)
        self.predict(tf.constant(image))

        super().__init__()

    def infer_ready(self):
        """ Returns True when model is ready for inference. Always True since 
        model is created when object is initialized. """

        return True

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
