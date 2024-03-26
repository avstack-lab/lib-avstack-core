class ReferenceFrameError(Exception):
    def __init__(self, frame):
        pass


class FrameEquivalenceError(Exception):
    def __init__(self, frame1, frame2):
        self.frame1 = frame1
        self.frame2 = frame2

        super(FrameEquivalenceError, self).__init__(
            "{} and {} need to be the same but are not".format(frame1, frame2)
        )

    def __reduce__(self):
        return (FrameEquivalenceError, (self.arg1, self.arg2))
