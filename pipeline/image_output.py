import os
import cv2
from numpy import array

from pipeline.pipeline import Pipeline


class ImageOutput(Pipeline):
    """ Pipeline task that delivers the images to some output. """

    def __init__(self, dst, dir, show_img, full_base,
                 image_ext=".jpg", jpg_quality=None, png_compression=None):

        self.dst = dst
        self.dir = dir
        self.show_img = show_img
        self.full_base = full_base
        self.image_ext = image_ext

        # 0 - 100 (higher means better). Default is 95.
        self.jpg_quality = jpg_quality

        # 0 - 9 (higher means a smaller size and longer compression time). Default is 3.
        self.png_compression = png_compression

        super().__init__()

    def map(self, data):

        # for image_id, image in zip(data["image_id"], data[self.dst]):
        #     self.export_img(image_id, image)
        print("hi")
        for _ in map(self.export_img, data["image_id"], data[self.dst]):
            pass

        return data

    def export_img(self, image_id, image):
        """ Saves image to a file. Also displays the image if self.show is True. """

        if self.show_img:
            image.show_img()

        image = cv2.cvtColor(array(image), cv2.COLOR_BGR2RGB)

        # Prepare output for image based on image_id
        dirname, basename = os.path.split(image_id)

        dirname = os.path.join(
            self.dir, dirname) if self.full_base else self.dir
        os.makedirs(dirname, exist_ok=True)

        basename = os.path.splitext(basename)[0] + self.image_ext

        path = os.path.join(dirname, basename)

        print("Saving Image: " + path)

        if self.image_ext == ".jpg":
            cv2.imwrite(path, image,
                        (cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality) if self.jpg_quality else None)
        elif self.image_ext == ".png":
            cv2.imwrite(path, image,
                        (cv2.IMWRITE_PNG_COMPRESSION, self.png_compression) if self.png_compression else None)
        else:
            raise Exception("Unsupported image format")
