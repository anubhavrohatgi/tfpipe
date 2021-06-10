import os
from collections import deque
import cv2

from pipeline.pipeline import Pipeline


class ImageInput(Pipeline):
    """ Pipeline task to capture images. """

    def __init__(self, path, batch_size, valid_exts=(".jpg", ".png")):
        self.valid_exts = valid_exts
        self.batch_size = batch_size

        if os.path.isdir(path):
            images = self.images_from_dir(path)
        else:
            images = [path] if self.valid_extension(path) else []

        print("Images: " + str(images))

        self.images = deque(images)

        super().__init__()

    def valid_extension(self, path):
        """ Returns True if path has a valid image extension. """
        return os.path.splitext(path)[-1] in self.valid_exts

    def images_from_dir(self, path: str) -> list[str]:
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

        image_ids = list()
        image_data = list()
        while len(image_ids) < self.batch_size and len(self.images) > 0:
            image_file = self.images.popleft()

            print("Current Image: " + image_file)

            image = cv2.cvtColor(
                cv2.imread(image_file), cv2.COLOR_BGR2RGB)

            image_ids.append(image_file)
            image_data.append(image)

        data = {
            "image_id": image_ids,
            "image": image_data
        }

        return data

        # image_file = self.images.popleft()
        # print("Current Image: " + image_file)
        # image = cv2.cvtColor(
        #     cv2.imread(image_file), cv2.COLOR_BGR2RGB)

        # data = {
        #     "image_id": image_file,
        #     "image": image
        # }

        # return data
