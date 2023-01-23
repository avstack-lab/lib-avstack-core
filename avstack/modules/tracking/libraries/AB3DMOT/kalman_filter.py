# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox3D, info, framerate):
        """
        Initialises a tracker using initial bounding box.

        States:
        [x, y, z, yaw, l, w, h, vx, vy, vz]
        """
        # define constant velocity model
        self.dt = 1.0 / framerate
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, self.dt, 0, 0],  # state transition matrix
                [0, 1, 0, 0, 0, 0, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # -- measurement uncertainty
        self.kf.R = np.diag([1, 1, 1, 10 * np.pi / 180, 0.5, 0.5, 0.5]) ** 2

        # -- state uncertainties independent of framerate
        self.kf.P = np.diag([5, 5, 5, 20 * np.pi / 180, 2, 2, 2, 10, 10, 10]) ** 2

        # -- process noise matrix DOES depend on framerate
        self.kf.Q = (
            np.diag([3, 3, 3, 10 * np.pi / 180, 0.5, 0.5, 0.5, 10, 10, 10]) * self.dt
        ) ** 2
        self.kf.x[:7] = bbox3D.reshape((7, 1))

        # -- parameters
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info associated

    @property
    def n_updates(self):
        return self.hits

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if (
            abs(new_theta - predicted_theta) > np.pi / 2.0
            and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0
        ):  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the rage
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.info = info

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))

    def get_velocity(self):
        return self.kf.x[7:10].reshape((3,))
