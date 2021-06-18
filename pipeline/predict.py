from tfpipe.core.utils import get_init_img, build_predictor
from tfpipe.pipeline.pipeline import Pipeline


class Predict(Pipeline):
    """ The pipeline task for single-process predicting. """

    def __init__(self, args):
        print("Loading model...")
        self.predict = build_predictor(args.framework, args.weights, args.size)

        print("Inferencing Test Image...")

        self.predict(get_init_img(args.size))

        super().__init__()

    def infer_ready(self):
        """ Returns True when model is ready for inference. Always True since
        model is created when object is initialized. """

        return True

    def is_working(self):
        """ Returns False because predictions are done linearly. """

        return False

    def map(self, data):
        data["predictions"] = self.predict(data["predictions"])

        return data

    def cleanup(self):
        pass
