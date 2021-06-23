import cv2
import numpy as np
from tensorflow import constant
from multiprocessing import Process, Queue

from tfpipe.core.config import cfg
from tfpipe.core.utils import images_from_dir
from tfpipe.pipeline.pipeline import Pipeline


class RedisCapture(Pipeline):
    """ Pipeline task to capture images from Redis. """

    class _InputStream(Process):
        def __init__(self, redis, image_queue):
            self.pub = redis.pubsub()
            self.image_queue = image_queue

            try:
                self.pub.subscribe("frames")
            except:
                print("*** Failed to connect to Redis channel. Exiting... ***")
                exit()

            super().__init__(daemon=True)

        def run(self):
            """ Polls the Redis stream and adds file paths to the image 
            queue when they are available. """

            for msg in self.pub.listen():
                print(msg)
                data = msg['data']

                if data == 1:
                    print("*** Connected to Redis Channel! ***")
                else:
                    for image_path in images_from_dir(data):
                        self.image_queue.put(image_path)

    def __init__(self, redis):
        image_queue = Queue()

        # Used to complete overhead stemming from the first inference before Redis connection
        # image_queue.put(cfg.RED_INIT_IMG)

        self._worker = self._InputStream(redis, image_queue)

        super().__init__(source=image_queue)

        self._worker.start()

    def is_working(self):
        """ Indicates if the pipeline should stop processing because 
        the input stream has ended. """

        return self._worker.is_alive() or not self.source.empty()

    def image_ready(self):
        """ Returns True if the next image is ready. """

        return not self.source.empty()

    def map(self, _=None):
        """ Returns the image content of the next image in the Redis stream. """

        if not self.image_ready():
            return Pipeline.Empty

        image_file = self.source.get()

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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pre_proc = [cv2.resize(image, (self.size, self.size)) / 255.0]
        pre_proc = constant(np.asanyarray(pre_proc).astype(np.float32))

        data = {
            "image_path": image_file,
            "image": image,
            "predictions": pre_proc
        }

        return data

    def cleanup(self):
        """ Closes video file or capturing device. This function should be 
        triggered after the pipeline completes its tasks. """

        pass
