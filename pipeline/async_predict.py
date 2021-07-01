import tensorflow as tf
from multiprocessing import Process, Queue, Value

from tfpipe.core.utils import get_devices, get_init_img, build_predictor
from tfpipe.pipeline.pipeline import Pipeline

from time import time


class AsyncPredictor(Process):
    """ The asynchronous predictor. """

    def __init__(self, args, device, task_queue, result_queue):
        self.size = args.size
        self.weights = args.weights
        self.framework = args.framework
        self.device = device
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.ready = Value('i', 0)

        super().__init__(daemon=True)

    def run(self):
        """ The main prediction loop. """
        print("starting")
        gpu = tf.config.list_physical_devices('GPU')[self.device]
        tf.config.experimental.set_memory_growth(gpu, True)
        gpu_cfg = [tf.config.LogicalDeviceConfiguration(memory_limit=2000)]
        tf.config.set_logical_device_configuration(gpu, gpu_cfg)

        vgpu = tf.config.list_logical_devices('GPU')[self.device]
        print(vgpu)

        print("with")
        # with tf.device(self.device):
        with tf.device(vgpu.name):
            # with tf.device("/device:GPU:0"):

            # Create the model and prediction function
            print(f"Building Model for Device: {self.device}")
            predict = build_predictor(self.framework, self.weights, self.size)

            print(f"Inferencing Test Image: {self.device}")
            predict(get_init_img(self.size))

            # Set ready flag
            self.ready.value = 1
            print(f"Ready: {self.device}")
            while True:
                data = self.task_queue.get()
                if data == Pipeline.Exit:
                    break

                t = time()
                data["predictions"] = predict(data["predictions"])
                # print(f"Device: {self.device} | Time: {time() - t} s")

                self.result_queue.put(data)

                # print(self.device, "done")


class AsyncPredict(Pipeline):
    """ The pipeline task for multi-process predicting. """

    def __init__(self, args):
        print("listing devices")
        gpus, cpus = get_devices()
        num_gpus = min(len(gpus), args.gpus)
        num_cpus = min(len(cpus), args.cpus)
        assert num_gpus > 0 or num_cpus > 0, "Must specify number of gpus or cpus"

        # Number of inputs and outputs
        self.inx = self.outx = 0

        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = list()

        # create as many as you want,
        # doesn't matter what the name is because diff name space... maybe

        # Create GPU Predictors
        for gpu_id in range(num_gpus):
            worker = AsyncPredictor(
                args, gpu_id, self.task_queue, self.result_queue)
            self.workers.append(worker)

        # Create CPU Predictors
        for cpu_id in range(num_cpus):
            worker = AsyncPredictor(
                args, f"CPU:{cpu_id}", self.task_queue, self.result_queue)
            self.workers.append(worker)

        # Start the Jobs
        for w in self.workers:
            w.start()

        super().__init__()

    def map(self, data):
        if data != Pipeline.Empty:
            self.put(data)

        return self.get() if self.output_ready() else Pipeline.Skip

    def put(self, data):
        """ Puts data in the task queue. """

        self.inx += 1
        self.task_queue.put(data)

    def get(self):
        """ Returns first element in the output queue. """

        self.outx += 1
        return self.result_queue.get()

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
