import tensorflow as tf

from tfpipe.core.utils import get_init_img, build_predictor
from tfpipe.pipeline.pipeline import Pipeline

from time import time

# PTDiag
from ptdiag import PTProcess

class Predict(Pipeline):
    """ The pipeline task for single-process predicting. """

    def __init__(self, args):

        print("starting")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        gpu_cfg = [tf.config.LogicalDeviceConfiguration(memory_limit=args.vram)]
        tf.config.set_logical_device_configuration(gpu, gpu_cfg)

        vgpu = tf.config.list_logical_devices('GPU')[0]
        print(vgpu)
        self.device = vgpu.name

        with tf.device(self.device):
            print("Loading model...")
            self.predict, self.model = build_predictor(
                args.framework, args.weights, args.size, args.quick_load)

            print("Inferencing Test Image...")

            self.predict(get_init_img(args.size))

        super().__init__()

        self.index = self.loop_time = self.gpu_time = 0

        self.ptp = PTProcess("predict")

    def infer_ready(self):
        """ Returns True when model is ready for inference. Always True since
        model is created when object is initialized. """

        return True
    
    def input_ready(self):
        """ Returns True if GPU is ready for next frame. """

        return True

    def is_working(self):
        """ Returns False because predictions are done linearly. """

        return False

    def map(self, data):
        # print('prd')
        if data == Pipeline.Empty:
            return Pipeline.Skip
            
        with tf.device(self.device):
            # t = time()
            self.ptp.on()
            data["predictions"] = self.predict(data["predictions"])
            self.ptp.off()
            # self.gpu_time += time() - t
            # self.index += 1
        return data

    def cleanup(self):
        # print(f"GPU: {self.device} | Images: {self.index} | Runtime: {self.gpu_time} s | FPS: {self.index/self.gpu_time}")

        pass
