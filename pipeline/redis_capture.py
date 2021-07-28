import cv2
import tensorflow as tf
import numpy as np
from tensorflow import constant
from multiprocessing import Process, Queue, Value

from tfpipe.core.config import cfg
from tfpipe.core.utils import images_from_dir, create_redis, build_preproc
from tfpipe.pipeline.pipeline import Pipeline


class RedisCapture(Pipeline):
    """ Pipeline task to capture images from Redis. """

    class _InputStream(Process):
        def __init__(self, redis_info):
            self.redis_info = redis_info
            self.image_queue = Queue()

            self.ready_val = Value('i', 0)

            super().__init__(daemon=True)


        def run(self):
            """ Polls the Redis stream and adds file paths to the image 
            queue when they are available. """

            host, port, ch_in = self.redis_info
            redis = create_redis(host, port)

            pub = redis.pubsub()
            try:
                pub.subscribe(ch_in)
                self.ready_val.value = 1
            except:
                print("*** Failed to connect to Redis channel. Exiting... ***")
                self.ready_val.value = -1
            else:
                index = 0
                for msg in pub.listen():
                    print(f"Message Received: {msg} | Index: {index}")
                    index += 1
                    data = msg['data']

                    if data == 1:
                        print("*** Connected to Redis Channel! ***")
                        continue

                    try: 
                        images = images_from_dir(data)
                    except FileNotFoundError:
                        print(f"Got FileNotFoundError from data: {data}")
                        images = list()

                    for image_path in images:
                        self.image_queue.put(image_path)
        

        @property
        def ready(self):
            
            return self.ready_val.value

        def empty(self):
        
            return self.image_queue.empty()

        def get(self):

            return self.image_queue.get()

    def __init__(self, redis_info, size, input_ready=None):

        self._worker = self._InputStream(redis_info)
        self.size = size

        self.preprocess = build_preproc(size)

        print("Preprocessing Test Image...")
        test_image = cv2.imread(cfg.INIT_IMG)
        pp = tf.image.resize(test_image, (self.size, self.size)) / 255.0
        self.preprocess(pp)

        super().__init__()

        self._worker.start()

        while not self._worker.ready:
            pass

        if self._worker.ready == -1:
            raise Exception("Redis Connection Issue")

    def is_working(self):
        """ Indicates if the pipeline should stop processing because 
        the input stream has ended. """

        return self._worker.is_alive() or not self._worker.empty()

    def input_ready(self):
        """ Returns True if the next image is ready. """

        return not self._worker.empty()

    def map(self, _=None):
        """ Returns the image content of the next image in the Redis stream. """

        if not self.input_ready() or not self.is_working():
            return Pipeline.Empty

        image_file = self._worker.get()
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
            "image": image,
            "predictions": preproc_image,
            "meta": None
        }

        return data

    def cleanup(self):
        """ Closes video file or capturing device. This function should be 
        triggered after the pipeline completes its tasks. """

        pass
