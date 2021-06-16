import os
import cv2
import numpy as np

from collections import deque
from threading import Thread

from core.config import cfg
from core.utils import images_from_dir
from pipeline.pipeline import Pipeline


class ImageInput(Pipeline):
    """ Pipeline task to capture images. """

    def __init__(self, path, size, meta):
        self.size = size
        self.meta = meta

        images = images_from_dir(path)

        self.images = deque(images)

        super().__init__()


    def is_working(self):
        """ Returns True if there are images yet to be captured. """

        return len(self.images) > 0

    def image_ready(self):
        """ Returns True if the next image is ready. """

        return True

    def map(self, _):
        """ Returns the image content of the next image in the input. """
        
        if not self.image_ready():
            return Pipeline.Skip

        image_file = self.images.popleft()

        print("Current File: " + image_file)

        image = cv2.imread(image_file)

        if image is None:
            try:
                image = np.reshape(
                    np.fromfile(image_file, dtype=np.uint8), cfg.MTAUR_DIMENSIONS)
            except Exception as e:
                print(f"Got Exception: {e}")
                print(f"*** Error: byte length not recognized or file: {image_file} ***")
                return Pipeline.Skip

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.meta:
            image = cv2.resize(image, (self.size, self.size))

        data = {
            "image_id": image_file,
            "image": image
        }

        return data
