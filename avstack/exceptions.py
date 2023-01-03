

class BoundingBoxError(Exception):
    def __init__(self, box):
        self.box = box

        super(BoundingBoxError, self).__init__('Bounding box is of an incorrect format {}'.format(box))

    def __reduce__(self):
        return (BoundingBoxError, (self.box))


class CompatibilityError(Exception):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

        super(CompatibilityError, self).__init__('{}, is not compatible with {}'.format(arg1, arg2))

    def __reduce__(self):
        return (CompatibilityError, (self.arg1, self.arg2))
