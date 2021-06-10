import os
import cv2

from collections import deque
from ujson import load

from pipeline.pipeline import Pipeline


class ImageInput(Pipeline):
    """ Pipeline task to capture images. """

    def __init__(self, path, valid_exts=(".jpg", ".png")):
        self.valid_exts = valid_exts

        if os.path.isdir(path):
            images = self.images_from_dir(path)
        else:
            images = self.images_from_file(path)

        # print("Images: " + str(images))

        self.images = deque(images)

        super().__init__()

    def valid_extension(self, path):
        """ Returns True if path has a valid image extension. """

        return os.path.splitext(path)[-1] in self.valid_exts

    def images_from_file(self, path: str):
        """ Returns list of image paths from a file path. """

        if path.endswith('.json'):
            images = load(open(path, 'r'))
        else:
            images = [path] if self.valid_extension(path) else []
        
        return images

    def images_from_dir(self, path: str):
        """ Returns a list of paths to valid images. """

        images = list()
        for root, _, files in os.walk(path):
            for img in files:
                if self.valid_extension(img):
                    images.append(os.path.join(root, img))

        return images

    def is_working(self):
        """ Returns True if there are images yet to be captured. """

        return len(self.images) > 0

    def image_ready(self):
        """ Returns True if the next image is ready. """

        return True

    def map(self, _):
        """ Returns the image content and metadata of the next image in the input. """

        if not self.image_ready():
            return Pipeline.Skip

        image_file = self.images.popleft()
        # print("Current Image: " + image_file)
        image = cv2.cvtColor(
            cv2.imread(image_file), cv2.COLOR_BGR2RGB)

        data = {
            "image_id": image_file,
            "image": image
        }

        return data
