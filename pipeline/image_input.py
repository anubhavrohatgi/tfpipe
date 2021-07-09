import cv2
import tensorflow as tf
import numpy as np
from tensorflow import constant
from collections import deque

from tfpipe.core.config import cfg
from tfpipe.core.utils import images_from_dir, build_preproc
from tfpipe.pipeline.pipeline import Pipeline


class ImageInput(Pipeline):
    """ Pipeline task to capture images. """

    def __init__(self, path, size, meta):
        self.size = size
        self.meta = meta

        images = images_from_dir(path)

        self.images = deque(images)

        self.preprocess = build_preproc(size)

        with tf.device("CPU:0"):
            print("Preprocessing Test Image...")
            test_image = cv2.imread(cfg.INIT_IMG)
            pp = tf.image.resize(test_image, (self.size, self.size)) / 255.0
            self.preprocess(pp)

        super().__init__()

    def is_working(self):
        """ Returns True if there are images yet to be captured. """

        return len(self.images) > 0

    def image_ready(self):
        """ Returns True if the next image is ready. """

        return True

    def map(self, _=None):
        """ Returns the image content of the next image in the input. """

        # print("im input")
        with tf.device("CPU:0"):
            if not self.image_ready() or not self.is_working():
                return Pipeline.Empty

            image_file = self.images.popleft()
            image_id = 0
            if isinstance(image_file, list):
                image_file, image_id = image_file

            # print("Current File: " + image_file)

            image = cv2.imread(image_file)

            if image is None:
                try:
                    image = np.reshape(
                        np.fromfile(image_file, dtype=np.uint8), cfg.MTAUR_DIMENSIONS)
                except Exception as e:
                    print(f"Got Exception: {e}")
                    print(
                        f"*** Error: byte length not recognized or file: {image_file} ***")
                    return Pipeline.Empty

            pp = tf.image.resize(image, (self.size, self.size)) / 255.0
            preproc_image = self.preprocess(pp)

            data = {
                "image_path": image_file,
                "image_id": image_id,
                "image": image,
                "predictions": preproc_image,
                "meta": None
            }

            return data
