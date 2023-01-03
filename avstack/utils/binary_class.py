# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-03-20
# @Filename: binary_class.py
# @Last modified by:   spencer
# @Last modified time: 2021-03-20
"""
Binary classification statistics utilities
"""

import numpy as np


class ConfusionMatrix():
    def __init__(self, TP, TN, FP, FN, thresh=None):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.nP = TP + FN
        self.nN = FP + TN
        self.thresh = thresh
        self.matrix = np.asarray([[TP, FP], [FN, TN]])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Confusion Matrix:\n' + str(self.matrix)
