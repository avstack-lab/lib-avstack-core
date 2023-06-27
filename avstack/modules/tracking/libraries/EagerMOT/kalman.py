# @Author: spencer hallyburton <Spencer>
# @Date:   2022-06-02T14:38:19-04:00
# @Last modified by:   Spencer
# @Last modified time: 2022-06-02T14:38:32-04:00


import numpy as np
import quaternion
from filterpy.kalman import KalmanFilter

from avstack.geometry import Attitude, Box2D, Box3D, Position
from avstack.geometry import transformations as tforms


class EagerMOTTrack:
    count = 0

    def __init__(self, box2d, box3d):
        """A new track for eagermot"""
        # Set up the 3D kalman filter
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
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

        self.kf.P = np.diag([4, 4, 4, 2, 2, 2, 1, 10, 10, 10]) ** 2
        self.kf.R = np.diag([1, 1, 1, 0.5, 0.5, 0.5, 0.1]) ** 2
        self.t_last_predict = 0

        self.time_since_update = 0
        self.id = EagerMOTTrack.count
        EagerMOTTrack.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.box3d_coast = 0
        self.box2d_coast = 0

        # Set the track fields
        self.box2d = box2d
        if box3d is None:
            self.box3d_origin = None
            self.box3d_initialized = False
            self.box3d_n_confirmed = 0
        else:
            bb = box3d
            bbox3D = np.array([bb.t[0], bb.t[1], bb.t[2], bb.yaw, bb.h, bb.w, bb.l])
            self.box3d_origin = box3d.reference
            self.box3d_initialized = True
            self.kf.x[:7] = bbox3D[:, None]
            self.box3d_n_confirmed = 1
        if box2d is None:
            self.camera_calibration = None
            self.box2d_confirmed = False
            self.box2d_n_confirmed = 0
        else:
            self.camera_calibration = box2d.calibration
            self.box2d_confirmed = True
            self.box2d_n_confirmed = 1

    @property
    def n_updates(self):
        return self.hits

    @property
    def box3d(self):
        if self.box3d_initialized:
            x, y, z, yaw, h, w, l = self.kf.x[:7, 0]
            q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            pos = Position(np.array([x, y, z]), self.box3d_origin)
            rot = Attitude(q, self.box3d_origin)
            return Box3D(pos, rot, [h, w, l])
        else:
            return None

    @property
    def yaw(self):
        if self.box3d_initialized:
            return self.kf.x[3, 0]
        else:
            return None

    def set_F(self, t):
        dt = t - self.t_last_predict
        self.kf.F = np.eye(10)
        self.kf.F[:3, 7:10] = dt * np.eye(3)

    def set_Q(self, t):
        dt = t - self.t_last_predict
        self.kf.Q = (np.diag([10, 10, 10, 1, 1, 1, 0.5, 10, 10, 10]) ** 2) * dt

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"EagerMOT track:\nbox3d -- {self.box3d_n_confirmed} confirmed: {self.box3d}\nbox2d -- {self.box2d_n_confirmed} confirmed: {self.box2d}"

    def predict(self, t):
        """Predict the states forward a frame"""
        if self.box3d_initialized:
            # -- predict the 3d box first
            self.set_F(t)
            self.set_Q(t)
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

            # -- predict the 2d box next
            if self.camera_calibration is not None:
                self.box2d = self.box3d.project_to_2d_bbox(self.camera_calibration)

    def update(self, detection_2d, detection_3d):
        """Perform update with detections"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

            # -- 2d update first
            if detection_2d is not None:
                self.box2d_coast = 0
                self.box2d = detection_2d
                self.box2d_confirmed = True
                self.box2d_n_confirmed += 1
                self.camera_calibration = detection_2d.calibration

            # -- 3d update
            if detection_3d is not None:
                self.box3d_n_confirmed += 1
                self.box3d_coast = 0
                if self.box3d_origin is None:
                    self.box3d_origin = detection_3d.reference
                bb = detection_3d
                bbox3D = np.array([bb.t[0], bb.t[1], bb.t[2], bb.yaw, bb.h, bb.w, bb.l])
                if not self.box3d_initialized:
                    self.kf.x[:7] = bbox3D[:, None]
                    self.box3d_initialized = True
                else:
                    ######################### orientation correction
                    if self.kf.x[3] >= np.pi:
                        self.kf.x[3] -= np.pi * 2
                    if self.kf.x[3] < -np.pi:
                        self.kf.x[3] += np.pi * 2
                    new_theta = bbox3D[3]
                    if new_theta >= np.pi:
                        new_theta -= np.pi * 2
                    if new_theta < -np.pi:
                        new_theta += np.pi * 2
                    bbox3D[3] = new_theta

                    predicted_theta = self.kf.x[3]
                    if (
                        abs(new_theta - predicted_theta) > np.pi / 2.0
                        and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0
                    ):
                        self.kf.x[3] += np.pi
                        if self.kf.x[3] > np.pi:
                            self.kf.x[3] -= np.pi * 2
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
                        self.kf.x[3] -= np.pi * 2
                    if self.kf.x[3] < -np.pi:
                        self.kf.x[3] += np.pi * 2
