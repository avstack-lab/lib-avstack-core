import numpy as np

from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


# ============================================================
# Base Class For Integrity
# ============================================================


class _SensorIntegrity(BaseModule):
    """
    Base class for all integrity monitoring algorithms
    """

    def __init__(self, name, *args, **kwargs):
        """Inits defined in the subclasses"""
        super().__init__(*args, **kwargs)
        self.name = name
        self.test_pass = True

    @apply_hooks
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

    def __init__(self, p_thresh=0.95, *args, **kwargs):
        """
        Initialize an anomaly detection module

        Inputs:
        df - degrees of freedom of the distribution
        p_thresh - probability threshold for the chi-square statistic
        """
        super().__init__(name="Chi2", *args, **kwargs)
        # Set up the threshold values beforehand (expensive to do real time)
        from scipy.stats.distributions import chi2

        # Compute a threshold mapping
        self.p_thresh = p_thresh
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

            self.thresh[df] = chi2.ppf(self.p_thresh, df)

        # Compute chi square statistic
        self.g = float(np.transpose(y) @ np.linalg.inv(S) @ y)

        # Determine if this exceeds threshold
        if self.g > self.thresh[df]:
            self.test_pass = False
        else:
            self.test_pass = True
        return self.test_pass
