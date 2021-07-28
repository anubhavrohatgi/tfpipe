import tensorflow as tf
from multiprocessing import Process, Queue, Value
from ujson import loads

from tfpipe.core.utils import get_init_img, build_predictor
from tfpipe.pipeline.pipeline import Pipeline

from time import time


class AsyncPredictor(Process):
    """ The asynchronous predictor. """

    def __init__(self, args, device, vram, task_queue, result_queue, quick_load=False):
        self.size = args.size
        self.weights = args.weights
        self.framework = args.framework
        self.device = device
        self.vram = vram
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.quick_load = quick_load

        self.ready = Value('i', 0)

        super().__init__(daemon=False)

    def run(self):
        """ The main prediction loop. """

        gpu = tf.config.list_physical_devices("GPU")[self.device]
        tf.config.set_visible_devices([gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpu, True)
        gpu_cfg = [tf.config.LogicalDeviceConfiguration(memory_limit=self.vram)]
        tf.config.set_logical_device_configuration(gpu, gpu_cfg)

        vgpu = tf.config.list_logical_devices("GPU")[0]
        with tf.device(vgpu.name):

            # Create the model and prediction function
            print(f"Building Model for Device: {self.device}")
            predict, model = build_predictor(self.framework, self.weights, self.size, self.quick_load)

            print(f"Inferencing Test Image: {self.device}")
            predict(get_init_img(self.size))

            # Set ready flag
            self.ready.value = 1
            print(f"Ready: {self.device}")

            # index = 0
            # t = time()
            
            while True:
                data = self.task_queue.get()
                if data == Pipeline.Exit:
                    break

                data["predictions"] = predict(data["predictions"])
                # print(f"Device: {self.device} | Time: {time() - t} s")

                self.result_queue.put(data)

                # print(self.device, "done")
                # index += 1
            
            # runtime = time() - t
            # print(f"GPU: {self.device} | Images: {index} | Runtime: {runtime} | FPS: {index/runtime}")


class AsyncPredict(Pipeline):
    """ The pipeline task for multi-process predicting. """

    def __init__(self, args, is_redis):
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices([], "GPU")
        
        gpu_spec = loads(args.gpu_spec)
        num_gpus = len(gpus) if args.gpus == "all" or gpu_spec else int(args.gpus)
        assert num_gpus > 0, "Must specify number of gpus greater than 0"
        assert not [1 for gpu_id in gpu_spec if gpu_id >= num_gpus], "Must specify valid GPU"

        # Number of inputs and outputs
        self.inx = self.outx = 0

        self.task_queue = Queue()
        self.result_queue = Queue()
        self.cache = dict()
        self.workers = list()


        # Create GPU Predictors
        for gpu_id in (range(num_gpus) if not gpu_spec else gpu_spec):
            worker = AsyncPredictor(
                args, gpu_id, args.vram, self.task_queue, self.result_queue, args.quick_load)
            self.workers.append(worker)

        self.num_gpus = len(self.workers)

        # Start the Jobs
        for w in self.workers:
            w.start()

        super().__init__()

    def map(self, data):
        if data != Pipeline.Empty:
            data["c_id"] = self.inx
            self.inx += 1
            self.put(data)

        # return self.get() if self.output_ready() else Pipeline.Skip

        # Guarantee output order...
        if self.output_ready():
            data = self.get()
            if data["c_id"] == self.outx:
                self.outx += 1
                return data
            else:
                self.cache[data["c_id"]] = data

        data = self.cache.pop(self.outx, None)
        if data is not None:
            self.outx += 1
            return data
        else:
            return Pipeline.Skip
        

    def put(self, data, block=False):
        """ Puts data in the task queue. """

        self.task_queue.put(data, block)

    def get(self):
        """ Returns first element in the output queue. """

        return self.result_queue.get()

    def input_ready(self):
        """ Returns True if GPUs are ready for next frame. """

        return self.inx - self.outx < 2 * self.num_gpus

    def output_ready(self):
        """ Returns True if there is element in the output queue. """

        return not self.result_queue.empty()

    def infer_ready(self):
        """ Returns True when each of the predictors are ready for inference. """

        return all([worker.ready.value for worker in self.workers])

    def is_working(self):
        """ Working while num inputs != num outputs and while queues are not empty. """

        return self.inx != self.outx or not self.task_queue.empty() or not self.result_queue.empty()

    def cleanup(self):
        """ Kills all predictors. """

        for _ in self.workers:
            self.task_queue.put(Pipeline.Exit)
