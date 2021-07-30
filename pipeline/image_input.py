import cv2
import tensorflow as tf
import numpy as np
from collections import deque

from tfpipe.core.config import cfg
from tfpipe.core.utils import images_from_dir, gen_from_cap, build_preproc
from tfpipe.pipeline.pipeline import Pipeline

# PTDiag
from ptdiag import PTProcess

class ImageInput(Pipeline):
    """ Pipeline task to capture images. """

    def __init__(self, path, size, meta, input_ready=None):
        self.size = size
        self.meta = meta
        if input_ready:
            self.input_ready = input_ready

        images = images_from_dir(path)

        # self.images = deque(images)
        self.cap = None

        self.finished = False

        self.preprocess = build_preproc(size)

        print("Preprocessing Test Image...")
        test_image = cv2.imread(cfg.INIT_IMG)
        pp = tf.image.resize(test_image, (self.size, self.size)) / 255.0
        self.preprocess(pp)

        pp_images = list()
        for image_file, image_id, image in self.generator(deque(images)):
            pp = tf.image.resize(image, (self.size, self.size)) / 255.0
            pp = self.preprocess(pp)

            data = {
                "image_path": image_file,
                "image_id": image_id,
                "image": image,
                "predictions": pp,
                "meta": None
            }
            pp_images.append(data)

        print(len(pp_images))

        super().__init__(source=pp_images)
        # super().__init__(source=self.generator(deque(images)))

        self.ptp = PTProcess("image input")

    def is_working(self):
        """ Returns True if there are images yet to be captured. """

        # return len(self.images) > 0 or self.cap is not None
        return not self.finished

    def input_ready(self):
        """ Returns True if the next image is ready. """

        return True
    

    def generator(self, images):
        """ Generator that yields next image to be processed. """

        ptp = PTProcess("image gen")

        while len(images) > 0:
            while not self.input_ready():
                yield Pipeline.Empty
            ptp.on()
            image_id = 0
            image_file = images.popleft()

            if isinstance(image_file, list):
                image_file, image_id = image_file
                if isinstance(image_id, cv2.VideoCapture):
                    for pkg in gen_from_cap(image_file, image_id):
                        while not self.input_ready():
                            yield Pipeline.Empty
                        yield pkg
            else:
                image = cv2.imread(image_file)
                ptp.off()
                yield image_file, image_id, image

        # self.finished = True

        # while True:
        #     yield Pipeline.Empty
        
    def map(self, _=None):
        """ Returns the image content of the next image in the input. """

        self.ptp.on()
        if any(self.source):
            vi = self.source.pop()
            self.ptp.off()
            return vi
        # if video_info == Pipeline.Empty:
        else:
            self.ptp.off()
            self.finished = True
            return Pipeline.Empty
        
        self.ptp.on()

        image_file, image_id, image = video_info

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

        self.ptp.off()

        return data
