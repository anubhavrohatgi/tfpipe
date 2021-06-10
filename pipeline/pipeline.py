class Pipeline(object):
    """Pipeline class for individual input-based submission to the pipeline."""

    class Empty:
        pass

    class Skip:
        pass

    class Stop:
        pass

    def __init__(self, source=list()):
        self.source = source

    def __rshift__(self, other):
        """Allows Pipeline objects to connect using the `>>` operator."""

        if other is not None:
            other.map = self.create_map(other.map)
            return other
        else:
            return self

    def create_map(self, other_map):
        """Returns a new map function that calls self.map and forward its return value to other_map."""

        def map(data):

            data = self.map(data)

            return other_map(data) if data != Pipeline.Skip else Pipeline.Empty

        return map

    def map(self, data):
        """Overwrite to map the pipeline data."""

        return data

    def filter(self, data):
        """Overwrite to filter out the pipeline data."""

        return True

    def is_working(self):
        """Overwrite to determine how execution terminates."""

        return True


if __name__ == '__main__':
    DEBUG = True

    class TestPipe(Pipeline):
        def __init__(self, source):
            super().__init__(source)

        def map(self, data):
            if DEBUG:
                print("Source: " + str(self.source) + " | Data: " + str(data))

            return (data + self.source) ** self.source

    p1 = TestPipe(1)
    p2 = TestPipe(2)
    p3 = TestPipe(3)
    pl1 = p1 >> p2 >> p3

    p1 = TestPipe(1)
    p2 = TestPipe(2)
    p3 = TestPipe(3)
    pl2 = p1 >> p3 >> p2

    print('--')
    test_val = 2

    res1 = pl1.map(test_val)
    res2 = pl2.map(test_val)

    print("To show that commutativity is not preserved:")
    print(f"pl1 = p1 >> p2 >> p3; pl1({test_val}) = {res1}")
    print(f"pl2 = p1 >> p3 >> p2; pl2({test_val}) = {res2}")
