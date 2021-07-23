import cv2
import tensorflow as tf

from tfpipe.core.config import cfg
from tfpipe.core.utils import video_from_file, gen_from_cap, build_preproc
from tfpipe.pipeline.pipeline import Pipeline

class ClipInput(Pipeline):
    """ Pipeline task for ingesting clips. """
    
    def __init__(self, size, meta):
        self.size = size
        self.meta = meta

        self._exit = False
        self._finished = False

        self.preprocess = build_preproc(size)

        with tf.device("CPU:0"):
            print("Preprocessing Test Image...")
            test_image = cv2.imread(cfg.INIT_IMG)
            pp = tf.image.resize(test_image, (self.size, self.size)) / 255.0
            self.preprocess(pp)

        super().__init__(source=None)

    
    @property
    def exit(self):
        """ Determines when to exit input loop. """

        return self._exit

    def is_working(self):
        """ Returns false when no images are left in the feed. """

        return not self._finished
    
    def input(self):
        """ Receives a console input and checks the path for a video. If a video 
        exists, a cv2.VideoCapture object is created and info about the video 
        relevant to the output is returned. """

        path = input("Please provide the path to a video or type `exit` to quit the tool:\n> ")
        if path == "exit":
            self._exit = True
            
            return Pipeline.Skip

        video = video_from_file(path)
        if video is None:
            
            return Pipeline.Skip
        
        fps = int(video.get(cv2.CAP_PROP_FPS))
        resolution = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source = gen_from_cap(path, video)

        return path, fps, resolution

    def map(self, _=None):
        """ Returns the image content of the next image in the input. """

        with tf.device("CPU:0"):

            try:
                image_file, image_id, image = next(self.source)
            except StopIteration:
                self._finished = True
                return Pipeline.Empty

            pp = tf.image.resize(image, (self.size, self.size)) / 255.0
            preproc_image = self.preprocess(pp)

            data = {
                "image_path": image_file,
                "image_id": image_id,
                "image": image,
                "predictions": preproc_image,
                "meta": None,
                "detection": False
            }

            return data