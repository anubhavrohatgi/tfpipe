
from tfpipe.core.utils import load_weights
from tfpipe.pipeline.pipeline import Pipeline


class ConvertWeights(Pipeline):
    """ Pipeline task for converting and saving the model. """

    def __init__(self, weights, output, model_name, is_tiny):
        self.weights = weights
        self.output = output
        self.model_name = model_name
        self.is_tiny = is_tiny

    def map(self, model):
        """ Converts and saves the model. """

        load_weights(model, self.weights, self.model_name, self.is_tiny)
        model.summary()
        model.save(self.output)

        return model
