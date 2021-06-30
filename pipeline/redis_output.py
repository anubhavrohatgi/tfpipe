
from tfpipe.pipeline.pipeline import Pipeline


class RedisOutput(Pipeline):
    """ Pipeline task for publishing predictions. """

    def __init__(self, redis, ch_out, dst):
        self.redis = redis
        self.ch_out = ch_out
        self.dst = dst

        super().__init__()

    def map(self, data):
        self.redis.publish(self.ch_out, data[self.dst])

        return data
