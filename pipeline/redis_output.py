
from pipeline.pipeline import Pipeline


class RedisOutput(Pipeline):
    """ Pipeline task for publishing predictions. """

    def __init__(self, redis, dst):
        self.redis = redis
        self.dst = dst

        super().__init__()

    def map(self, data):
        self.redis.publish('bbox', data[self.dst])

        return data