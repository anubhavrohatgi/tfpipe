import cv2
import tensorflow as tf
from tqdm import tqdm

from tfpipe.core.config import cfg
from tfpipe.core.utils import video_from_file, gen_from_cap, build_preproc
from tfpipe.pipeline.pipeline import Pipeline

class ClipInput(Pipeline):
    """ Pipeline task for ingesting clips. """
    
    def __init__(self, size, meta, input_ready=None):
        self.size = size
        self.meta = meta
        if input_ready is not None:
            self.input_ready = input_ready

        self._exit = False
        self._finished = False

        self.preprocess = build_preproc(size)

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
    
    def input_ready(self):
        """ Pass function into __init__ to overload. """

        return True
    
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
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        del self.source # clear previous source
        self._finished = False
        self.source = self.generator(total_frames, path, video)

        return path, fps, resolution

    def generator(self, total_frames, path, video):
        video_gen = gen_from_cap(path, video)

        for video_info in tqdm(video_gen, total=total_frames):
            while not self.input_ready():
                yield Pipeline.Empty
            yield video_info
        
        # Guarantee that video is finished
        try:
            next(video_gen)
        except StopIteration:
            self._finished = True
        
        assert self._finished, "*** Error: Video generator yielded more frames than intended. ***"
        while True:
            yield Pipeline.Empty

    def map(self, _=None):
        """ Returns the image content of the next image in the input. """

        video_info = next(self.source)
        if video_info == Pipeline.Empty:
            return Pipeline.Empty

        image_file, image_id, image = video_info

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