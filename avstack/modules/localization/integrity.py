# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-21
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""

import numpy as np


# ============================================================
# Base Class For Integrity
# ============================================================


class _SensorIntegrity:
    """
    Base class for all integrity monitoring algorithms
    """

    def __init__(self, name):
        """Inits defined in the subclasses"""
        self.name = name
        self.test_pass = True

    def __call__(self, *args, **kwargs):
        return self.test(*args, **kwargs)

    def test(self):
        """Test defined in the subclasses"""
        raise NotImplementedError


# ============================================================
# Chi 2 Integrity
# ============================================================


class Chi2Integrity(_SensorIntegrity):
    """
    Classic chi-2 threshold for anomaly detection

    Takes in innovation along with innovation covariance and performs test
    Has options to perform instantaneous or windowed chi square detection
    """

    def __init__(self, p_thresh=0.95):
        """
        Initialize an anomaly detection module

        Inputs:
        df - degrees of freedom of the distribution
        p_thresh - probability threshold for the chi-square statistic
        """
        super().__init__(name="Chi2")
        # Set up the threshold values beforehand (expensive to do real time)
        from scipy.stats.distributions import chi2

        # Compute a threshold mapping
        self.thresh = {i: chi2.ppf(p_thresh, i) for i in range(6)}
        self.g = 0.0
        self.test_pass = True

    def test(self, y, S):
        """
        Perform the chi square integrity test
        """
        df = len(y)
        if df not in self.thresh:
            from scipy.stats.distributions import chi2

            self.thresh[df] = chi2.ppf(p_thresh, df)

        # Compute chi square statistic
        self.g = float(np.transpose(y) @ np.linalg.inv(S) @ y)

        # Determine if this exceeds threshold
        if self.g > self.thresh[df]:
            self.test_pass = False
        else:
            self.test_pass = True
        return self.test_pass
