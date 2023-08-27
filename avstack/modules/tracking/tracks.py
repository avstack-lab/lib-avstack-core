import json

import numpy as np

from avstack.datastructs import DataContainerDecoder
from avstack.environment.objects import VehicleState
from avstack.geometry import (
    Attitude,
    BoxDecoder,
    Position,
    ReferenceDecoder,
    ReferenceFrame,
    Velocity,
)
from avstack.geometry.bbox import Box2D, Box3D
from avstack.geometry.transformations import (
    cartesian_to_spherical,
    razelrrt_to_xyzvel,
    spherical_to_cartesian,
    transform_orientation,
    xyzvel_to_razelrrt,
)


zero3 = np.zeros((3, 3))
eye3 = np.eye(3)

std_tracks = ["xyfromraztrack", "xyzfromrazeltrack", "xyzfromrazelrrttrack"]
box_tracks = ["basicboxtrack2d", "basicboxtrack3d", "basicjointboxtrack"]


class TrackEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, _TrackBase):
            t_dict = {
                "obj_type": o.obj_type,
                "t0": o.t0,
                "t": o.t,
                "ID": o.ID,
                "coast": o.coast,
                "n_updates": o.n_updates,
                "age": o.age,
                "x": o.x.tolist(),
                "P": o.P.tolist(),
                "reference": o.reference.encode(),
            }
        else:
            raise NotImplementedError(f"{type(o)}, {o}")
        if isinstance(o, (BasicBoxTrack2D, BasicBoxTrack3D, BasicJointBoxTrack)):
            t_dict["box"] = o.box.encode()
            t_dict["v"] = o.velocity.x.tolist()
        if isinstance(o, (BasicJointBoxTrack)):
            t_dict["track2d"] = o.track_2d.encode() if o.track_2d is not None else None
        return {type(o).__name__.lower(): t_dict}


class TrackDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        try:
            reference = json.loads(
                list(json_object.values())[0]["reference"], cls=ReferenceDecoder
            )
        except Exception:
            pass
        if any([st in json_object for st in std_tracks]):
            if "xyfromraztrack" in json_object:
                json_object = json_object["xyfromraztrack"]
                factory = XyFromRazTrack
            elif "xyzfromrazeltrack" in json_object:
                json_object = json_object["xyzfromrazeltrack"]
                factory = XyzFromRazelTrack
            elif "xyzfromrazelrrttrack" in json_object:
                json_object = json_object["xyzfromrazelrrttrack"]
                factory = XyzFromRazelRrtTrack
            else:
                raise NotImplementedError(json_object)
            out = factory(
                json_object["t0"],
                None,
                reference=reference,
                obj_type=json_object["obj_type"],
                ID_force=json_object["ID"],
                x=np.array(json_object["x"]),
                P=np.array(json_object["P"]),
                t=json_object["t"],
                coast=json_object["coast"],
                n_updates=json_object["n_updates"],
                age=json_object["age"],
            )
        elif any([bt in json_object for bt in box_tracks]):
            if "basicboxtrack2d" in json_object:
                json_object = json_object["basicboxtrack2d"]
                factory = BasicBoxTrack2D
            elif "basicboxtrack3d" in json_object:
                json_object = json_object["basicboxtrack3d"]
                factory = BasicBoxTrack3D
            elif "basicjointboxtrack" in json_object:
                json_object = json_object["basicjointboxtrack"]
                factory = BasicBoxTrack3D
            else:
                raise NotImplementedError(json_object)
            box = json.loads(json_object["box"], cls=BoxDecoder)
            out = factory(
                json_object["t0"],
                box,
                reference=reference,
                obj_type=json_object["obj_type"],
                ID_force=json_object["ID"],
                v=np.array(json_object["v"]),
                P=np.array(json_object["P"]),
                t=json_object["t"],
                coast=json_object["coast"],
                n_updates=json_object["n_updates"],
                age=json_object["age"],
            )
        else:
            return json_object
        return out


class TrackContainerDecoder(DataContainerDecoder):
    data_decoder = TrackDecoder


def KF_update(x, P, hx, H, z, R):
    y = z - hx
    Sinv = np.linalg.inv(H @ P @ H.T + R)
    K = P @ H.T @ Sinv
    return x + K @ y, (np.eye(P.shape[0]) - K @ H) @ P


class _TrackBase:
    ID_counter = 0

    def __init__(
        self,
        t0,
        x,
        P,
        reference,
        obj_type,
        ID_force=None,
        t=None,
        coast=0,
        n_updates=1,
        age=0,
    ) -> None:
        if ID_force is None:
            ID = _TrackBase.ID_counter
            _TrackBase.ID_counter += 1
        else:
            ID = ID_force
        self.obj_type = obj_type
        self.coast = coast
        self.active = True
        self.n_updates = n_updates
        self.age = age
        self.ID = int(ID)
        self.t0 = t0
        self.t = t0 if t is None else t
        self.t_last_predict = self.t
        self.x = x
        self.P = P
        self.reference = reference

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, ReferenceFrame):
            raise ValueError(f"Reference frame type not appropriate, {type(reference)}")
        self._reference = reference

    @property
    def position(self):
        return Position(self.x[self.idx_pos], self.reference)

    @position.setter
    def position(self, position: Position):
        self.x[self.idx_pos] = position.x

    @property
    def velocity(self):
        return Velocity(self.x[self.idx_vel], self.reference)

    @velocity.setter
    def velocity(self, velocity: Velocity):
        self.x[self.idx_vel] = velocity.x

    @staticmethod
    def f(x, dt):
        raise NotImplementedError

    @staticmethod
    def F(x, dt):
        raise NotImplementedError

    @staticmethod
    def h(x):
        raise NotImplementedError

    @staticmethod
    def H(x):
        raise NotImplementedError

    @staticmethod
    def Q(dt):
        raise NotImplementedError

    def encode(self):
        return json.dumps(self, cls=TrackEncoder)

    def _predict(self, t):
        dt = t - self.t_last_predict
        self.x = self.f(self.x, dt)
        F = self.F(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q(dt)
        self.t_last_predict = t
        self.age += 1
        self.coast += 1

    def _update(self, z, R):
        y = z - self.h(self.x)
        H = self.H(self.x)
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        self.coast = 0
        self.n_updates += 1

    def predict(self, t):
        """Can override this in subclass"""
        self._predict(t)

    def update(self, z, R):
        """Can override this in subclass"""
        self._update(z, R)

    def R_old_to_new(self, reference):
        diff = self.reference.differential(reference)
        R_old_to_new = transform_orientation(diff.q, "quat", "dcm")
        return R_old_to_new

    def change_reference(self, reference, inplace: bool):
        raise NotImplementedError


class XyFromRazTrack(_TrackBase):
    """Tracking on raz measurements

    IMPORTANT: assumes we are in a sensor-relative coordinate frame.
    This assumption allows us to say that the sensor is always
    facing 'forward' which simplifies the calculations. If we are
    wanting to track in some other coordinate frame, we will need to
    explicitly incorporate the sensor's pointing angle and position
    offset in the calculations."""

    NAME = "raztrack"

    def __init__(
        self,
        t0,
        raz,
        reference,
        obj_type,
        ID_force=None,
        x=None,
        P=None,
        t=None,
        coast=0,
        n_updates=1,
        age=0,
        *args,
        **kwargs,
    ):
        """
        Track state is: [x, y, vx, vy]
        Measurement is: [range, azimuth]
        """
        # Position can be initialized fairly well
        # Velocity can only be initialized along the range rate
        if x is None:
            x = np.array([raz[0] * np.cos(raz[1]), raz[0] * np.sin(raz[1]), 0, 0])
        if P is None:
            r_sig = 5
            v_sig = 30
            P = np.diag([r_sig, r_sig, v_sig, v_sig]) ** 2
        self.idx_pos = [0, 1]
        self.idx_vel = [2, 3]
        super().__init__(
            t0, x, P, reference, obj_type, ID_force, t, coast, n_updates, age
        )

    @staticmethod
    def H(x):
        """Partial derivative of the measurement function w.r.t x at x hat

        NOTE: assumes we are in a sensor-relative coordinate frame
        """
        H = np.zeros((2, 4))
        r = np.linalg.norm(x[:2])
        H[0, :2] = x[:2] / r
        H[1, 0] = -x[1] / r**2
        H[1, 1] = x[0] / r**2
        return H

    @staticmethod
    def h(x):
        """Measurement function

        NOTE: assumes we are in a sensor-relative coordinate frame
        """
        return np.array([np.linalg.norm(x[:2]), np.arctan2(x[1], x[0])])

    @staticmethod
    def F(x, dt):
        """Partial derivative of the propagation function w.r.t. x at x hat"""
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def f(x, dt):
        """State propagation function"""
        return np.array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]])

    @staticmethod
    def Q(dt):
        """This is most definitely not optimal and should be tuned in the future"""
        return (np.diag([2, 2, 2, 2]) * dt) ** 2

    def update(self, z, R=np.diag([10, 1e-2]) ** 2):
        self._update(z, R)

    def change_reference(self, reference, inplace: bool):
        vec = Position(np.array([self.x[0], self.x[1], 0]), self.reference)
        vec.change_reference(reference, inplace=True)
        if inplace:
            self.x[:2] = vec.x[:2]
            self.reference = reference
        else:
            raise NotImplementedError


class XyzFromRazelTrack(_TrackBase):
    """Tracking on razel measurements

    IMPORTANT: assumes we are in a sensor-relative coordinate frame.
    This assumption allows us to say that the sensor is always
    facing 'forward' which simplifies the calculations. If we are
    wanting to track in some other coordinate frame, we will need to
    explicitly incorporate the sensor's pointing angle and position
    offset in the calculations."""

    NAME = "razeltrack"

    def __init__(
        self,
        t0,
        razel,
        reference,
        obj_type,
        ID_force=None,
        x=None,
        P=None,
        t=None,
        coast=0,
        n_updates=1,
        age=0,
        *args,
        **kwargs,
    ):
        """
        Track state is: [x, y, z, vx, vy, vz]
        Measurement is: [range, azimuth, elevation, range rate]
        """
        # Position can be initialized fairly well
        # Velocity can only be initialized along the range rate
        if x is None:
            x = np.array([*spherical_to_cartesian(razel), 0, 0, 0])
        if P is None:
            r_sig = 5
            v_sig = 30
            P = np.diag([r_sig, r_sig, r_sig, v_sig, v_sig, v_sig]) ** 2
        self.idx_pos = [0, 1, 2]
        self.idx_vel = [3, 4, 5]
        super().__init__(
            t0, x, P, reference, obj_type, ID_force, t, coast, n_updates, age
        )

    @staticmethod
    def H(x):
        """Partial derivative of the measurement function w.r.t x at x hat

        NOTE: assumes we are in a sensor-relative coordinate frame
        """
        H = np.zeros((3, 6))
        r = np.linalg.norm(x[:3])
        r2d = np.linalg.norm(x[:2])
        H[0, :3] = x[:3] / r
        H[1, 0] = -x[1] / r2d**2
        H[1, 1] = x[0] / r2d**2
        H[2, 0] = -x[0] * x[2] / (r**2 * r2d)
        H[2, 1] = -x[1] * x[2] / (r**2 * r2d)
        H[2, 2] = r2d / r**2
        return H

    @staticmethod
    def h(x):
        """Measurement function

        NOTE: assumes we are in a sensor-relative coordinate frame
        """
        return cartesian_to_spherical(x[:3])

    @staticmethod
    def F(x, dt):
        """Partial derivative of the propagation function w.r.t. x at x hat"""
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def f(x, dt):
        """State propagation function"""
        return np.array(
            [x[0] + x[3] * dt, x[1] + x[4] * dt, x[2] + x[5] * dt, x[3], x[4], x[5]]
        )

    @staticmethod
    def Q(dt):
        """This is most definitely not optimal and should be tuned in the future"""
        return (np.diag([2, 2, 2, 2, 2, 2]) * dt) ** 2

    def update(self, z, R=np.diag([4, 1e-2, 5e-2]) ** 2):
        self._update(z, R)

    def change_reference(self, reference, inplace: bool):
        if inplace:
            RO2N = self.R_old_to_new(reference)
            self.position = self.position.change_reference(reference, inplace=False)
            self.velocity = self.velocity.change_reference(reference, inplace=False)
            RO2N_B = np.block([[RO2N, zero3], [zero3, RO2N]])
            self.P = RO2N_B @ self.P @ RO2N_B.T
            self.P[self.P < 1e-5] = 0
            self.reference = reference
        else:
            raise NotImplementedError("Need to implement this")


class XyzFromRazelRrtTrack(_TrackBase):
    """Tracking on razel rrt measurements

    NOTE: to reduce the nonlinearities of the range rate,
    we use the pseudo-measurement of range * rrt.
    See Farina and Studer, Radar Data Processing for an explanation

    NOTE: Miller and Leskiw in Nonlinear Estimation with Radar Observations
    showed it is better to process updates in the order of
    azimuth, elevation, and range and range rate (simultaneously) and
    NOT all at once...but we do not do this here...

    IMPORTANT: assumes we are in a sensor-relative coordinate frame.
    This assumption allows us to say that the sensor is always
    facing 'forward' which simplifies the calculations. If we are
    wanting to track in some other coordinate frame, we will need to
    explicitly incorporate the sensor's pointing angle and position
    offset in the calculations.

    IMPORTANT: range rate is defined as positive moving away from sensor
    """

    NAME = "razelrrttrack"

    def __init__(
        self,
        t0,
        razelrrt,
        reference,
        obj_type,
        ID_force=None,
        x=None,
        P=None,
        t=None,
        coast=0,
        n_updates=1,
        age=0,
        *args,
        **kwargs,
    ):
        """
        Track state is: [x, y, z, vx, vy, vz]
        Measurement is: [range, azimuth, elevation, range rate]
        """
        # Position can be initialized fairly well
        # Velocity can only be initialized along the range rate
        if x is None:
            x = razelrrt_to_xyzvel(razelrrt)
        if P is None:
            # Note the uncertainty on transverse velocities is larger (see note above)
            v_unit = x[:3] / razelrrt[0]
            r_sig = 5
            v_sig = 30
            rrt_p_max = v_sig  # complete uncertainty gives 10, total certainty gives 2
            rrt_p_min = 10
            P = (
                np.diag(
                    [
                        r_sig,
                        r_sig,
                        r_sig,
                        *np.maximum(rrt_p_min, rrt_p_max * (1 - v_unit)),
                    ]
                )
                ** 2
            )
        self.idx_pos = [0, 1, 2]
        self.idx_vel = [3, 4, 5]
        super().__init__(
            t0, x, P, reference, obj_type, ID_force, t, coast, n_updates, age
        )

    @property
    def rrt(self):
        return self.velocity @ self.position.unit()

    @staticmethod
    def H(x):
        """Partial derivative of the measurement function w.r.t x at x hat

        NOTE: we are using the pseudo measurement for range rate
        NOTE: assumes we are in a sensor-relative coordinate frame
        """
        H = np.zeros((4, 6))
        r = np.linalg.norm(x[:3])
        r2d = np.linalg.norm(x[:2])
        H[0, :3] = x[:3] / r
        H[1, 0] = -x[1] / r2d**2
        H[1, 1] = x[0] / r2d**2
        H[2, 0] = -x[0] * x[2] / (r**2 * r2d)
        H[2, 1] = -x[1] * x[2] / (r**2 * r2d)
        H[2, 2] = r2d / r**2
        # range rate pseudo measurement
        # range*rrt = vx*x + vy*y + vz*z
        H[3, :3] = x[3:6]
        H[3, 3:6] = x[:3]
        return H

    @staticmethod
    def h(x):
        """Measurement function

        NOTE: we are using the pseudo measurement for range rate
        NOTE: assumes we are in a sensor-relative coordinate frame
        """
        zhat = xyzvel_to_razelrrt(x)
        zhat[3] = zhat[0] * zhat[3]
        return zhat

    @staticmethod
    def F(x, dt):
        """Partial derivative of the propagation function w.r.t. x at x hat"""
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def f(x, dt):
        """State propagation function"""
        return np.array(
            [x[0] + x[3] * dt, x[1] + x[4] * dt, x[2] + x[5] * dt, x[3], x[4], x[5]]
        )

    @staticmethod
    def Q(dt):
        """This is most definitely not optimal and should be tuned in the future"""
        return (np.diag([2, 2, 2, 2, 2, 2]) * dt) ** 2

    def update(self, z, R=np.diag([1, 1e-2, 5e-2, 10]) ** 2):
        """Construct the pseudo measurement for range rate"""
        z = z.copy()  # copy bc we are doing pseudo measurement manipulation
        z[3] = z[0] * z[3]
        self._update(z, R)

    def change_reference(self, reference, inplace: bool):
        if inplace:
            RO2N = self.R_old_to_new(reference)
            self.position = self.position.change_reference(reference, inplace=False)
            self.velocity = self.velocity.change_reference(reference, inplace=False)
            RO2N_B = np.block([[RO2N, zero3], [zero3, RO2N]])
            self.P = RO2N_B @ self.P @ RO2N_B.T
            self.P[self.P < 1e-5] = 0
            self.reference = reference
        else:
            raise NotImplementedError("Need to implement this")


class BasicBoxTrack3D(_TrackBase):

    NAME = "boxtrack3d"

    def __init__(
        self,
        t0,
        box3d,
        reference,
        obj_type,
        ID_force=None,
        v=None,
        P=None,
        t=None,
        coast=0,
        n_updates=1,
        age=0,
    ):
        """Box state is: [x, y, z, h, w, l, vx, vy, vz] w/ yaw as attribute"""
        if v is None:
            v = np.array([0, 0, 0])
        x = np.array(
            [
                box3d.t[0],
                box3d.t[1],
                box3d.t[2],
                box3d.h,
                box3d.w,
                box3d.l,
                v[0],
                v[1],
                v[2],
            ]
        )
        if P is None:
            P = np.diag([5, 5, 5, 2, 2, 2, 10, 10, 10]) ** 2
        self.idx_pos = [0, 1, 2]
        self.idx_vel = [6, 7, 8]
        super().__init__(
            t0, x, P, reference, obj_type, ID_force, t, coast, n_updates, age
        )
        self.where_is_t = box3d.where_is_t
        self.q = box3d.q

    @staticmethod
    def f(x, dt):
        return np.array(
            [
                x[0] + x[6] * dt,
                x[1] + x[7] * dt,
                x[2] + x[8] * dt,
                x[3],
                x[4],
                x[5],
                x[6],
                x[7],
                x[8],
            ]
        )

    @staticmethod
    def F(x, dt):
        F = np.eye(9)
        F[:3, 6:9] = dt * np.eye(3)
        return F

    @staticmethod
    def h(x):
        return x[:6]

    @staticmethod
    def H(x):
        H = np.zeros((6, 9))
        H[:6, :6] = np.eye(6)
        return H

    @staticmethod
    def Q(dt):
        return (np.diag([2, 2, 2, 0.2, 0.2, 0.2, 3, 3, 3]) * dt) ** 2

    @property
    def attitude(self):
        return self.q

    @attitude.setter
    def attitude(self, attitude: Attitude):
        self.q = attitude

    @property
    def box3d(self):
        hwl = self.x[3:6]
        return Box3D(self.position, self.attitude, hwl, where_is_t=self.where_is_t)

    @property
    def box(self):
        return self.box3d

    @property
    def yaw(self):
        return self.box3d.yaw

    def update(self, box3d, R=np.diag([1, 1, 1, 0.25, 0.25, 0.25]) ** 2):
        if box3d.reference != self.reference:
            raise RuntimeError("Should have converted the box location before this...")
        if self.where_is_t != box3d.where_is_t:
            raise NotImplementedError(
                "Differing t locations not implemented: {}, {}".format(
                    self.where_is_t, box3d.where_is_t
                )
            )
        det = np.array([box3d.t[0], box3d.t[1], box3d.t[2], box3d.h, box3d.w, box3d.l])
        self._update(det, R)
        self.q = box3d.q

    def as_object(self):
        vs = VehicleState(obj_type=self.obj_type, ID=self.ID)
        vs.set(
            t=self.t,
            position=self.position,
            box=self.box3d,
            velocity=self.velocity,
            acceleration=None,
            attitude=self.attitude,
            angular_velocity=None,
        )
        return vs

    def change_reference(self, reference, inplace: bool):
        if inplace:
            RO2N = self.R_old_to_new(reference)
            self.position = self.position.change_reference(reference, inplace=False)
            self.velocity = self.velocity.change_reference(reference, inplace=False)
            self.attitude = self.attitude.change_reference(reference, inplace=False)
            RO2N_B = np.block(
                [[RO2N, zero3, zero3], [zero3, eye3, zero3], [zero3, zero3, RO2N]]
            )
            self.P = RO2N_B @ self.P @ RO2N_B.T
            self.P[self.P < 1e-5] = 0
            self.reference = reference
        else:
            raise NotImplementedError("Need to implement this")


class BasicBoxTrack2D(_TrackBase):
    NAME = "boxtrack2d"

    def __init__(
        self,
        t0,
        box2d,
        reference,
        obj_type,
        ID_force=None,
        v=None,
        P=None,
        t=None,
        coast=0,
        n_updates=1,
        age=0,
    ):
        """Box state is: [x, y, w, h, vx, vy]"""
        if v is None:
            v = np.array([0, 0])
        x = np.array(
            [
                box2d.center[0],
                box2d.center[1],
                box2d.w,
                box2d.h,
                v[0],
                v[1],
            ]
        )
        if P is None:
            P = np.diag([10, 10, 10, 10, 10, 10]) ** 2
        self.idx_pos = [0, 1]
        self.idx_vel = [4, 5]
        super().__init__(
            t0, x, P, reference, obj_type, ID_force, t, coast, n_updates, age
        )
        self.calibration = box2d.calibration

    @property
    def width(self):
        return self.x[2]

    @property
    def height(self):
        return self.x[3]

    @property
    def xmin(self):
        return self.position[0] - self.width / 2

    @property
    def xmax(self):
        return self.position[0] + self.width / 2

    @property
    def ymin(self):
        return self.position[1] - self.height / 2

    @property
    def ymax(self):
        return self.position[1] + self.height / 2

    @property
    def box2d(self):
        return Box2D([self.xmin, self.ymin, self.xmax, self.ymax], self.calibration)

    @property
    def box(self):
        return self.box2d

    @staticmethod
    def f(x, dt):
        """Box state is: [x, y, w, h, vx, vy]"""
        return np.array([x[0] + x[4] * dt, x[1] + x[5] * dt, x[2], x[3], x[4], x[5]])

    @staticmethod
    def F(x, dt):
        F = np.eye(6)
        F[:2, 4:6] = dt * np.eye(2)
        return F

    @staticmethod
    def h(x):
        return x[:4]

    @staticmethod
    def H(x):
        H = np.zeros((4, 6))
        H[:4, :4] = np.eye(4)
        return H

    @staticmethod
    def Q(dt):
        return (np.diag([2, 2, 2, 2, 2, 2]) * dt) ** 2

    def change_reference(self, reference, inplace: bool):
        if inplace:
            self.reference = reference
        else:
            raise NotImplementedError

    def update(self, box2d, R=np.diag([5, 5, 5, 5]) ** 2):
        # if box2d.source_identifier != self.source_identifier:
        #     raise NotImplementedError("Sensor sources must be the same for now")
        det = np.array([box2d.center[0], box2d.center[1], box2d.w, box2d.h])
        self._update(det, R)


class BasicJointBoxTrack(_TrackBase):
    NAME = "boxtrack2d3d"

    def __init__(self, t0, box2d, box3d, reference, obj_type):
        self.track_2d = (
            BasicBoxTrack2D(t0, box2d, reference, obj_type)
            if box2d is not None
            else None
        )
        self.track_3d = (
            BasicBoxTrack3D(t0, box3d, reference, obj_type)
            if box3d is not None
            else None
        )

    @property
    def ID(self):
        return self.track_3d.ID

    @property
    def box(self):
        return self.box3d

    @property
    def x(self):
        return self.track_3d.x

    @property
    def P(self):
        return self.track_3d.P

    @property
    def obj_type(self):
        return self.track_3d.obj_type

    @property
    def idx_pos(self):
        return self.track_3d.idx_posok

    # avapi.visualize.replay.replay_track_results(track_res_frames, fig_width=8)

    @property
    def idx_vel(self):
        return self.track_3d.idx_vel

    @property
    def t0(self):
        return self.track_3d.t0

    @property
    def t(self):
        return self.track_3d.t

    @property
    def coast(self):
        return self.track_3d.coast

    @property
    def n_updates(self):
        return self.track_3d.n_updates

    @property
    def age(self):
        try:
            return max(self.track_2d.age, self.track_3d.age)
        except Exception:
            return self.track_3d.age

    @property
    def box2d(self):
        return self.track_2d.box2d if self.track_2d is not None else None

    @property
    def box3d(self):
        return self.track_3d.box3d if self.track_3d is not None else None

    @property
    def yaw(self):
        return self.box3d.yaw

    @property
    def n_updates_2d(self):
        return self.track_2d.n_updates if self.track_2d is not None else 0

    @property
    def n_updates_3d(self):
        return self.track_3d.n_updates if self.track_3d is not None else 0

    @property
    def coast_2d(self):
        return self.track_2d.coast if self.track_2d is not None else 0

    @property
    def coast_3d(self):
        return self.track_3d.coast if self.track_3d is not None else 0

    @property
    def reference(self):
        return self.track_3d.reference if self.track_3d is not None else None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"BasicJointBoxTrack w/ {self.track_2d.n_updates} 2D n_updates: {self.box2d}; {self.track_3d.n_updates} 3D n_updates: {self.box3d}"

    def update(self, box, obj_type, reference):
        if isinstance(box, tuple):
            b2 = box[0]
            b3 = box[1]
        elif isinstance(box, Box2D):
            b2 = box
            b3 = None
        elif isinstance(box, Box3D):
            b2 = None
            b3 = box
        else:
            raise NotImplementedError(type(box))

        # -- update 2d
        if self.track_2d is None and b2 is not None:
            self.track_2d = BasicBoxTrack2D(
                self.track_3d.t, b2, obj_type=obj_type, reference=reference
            )
        elif b2 is not None:
            self.track_2d.update(b2)

        # -- update 3d
        if self.track_3d is None and b3 is not None:
            self.track_3d = BasicBoxTrack3D(
                self.track_2d.t, b3, obj_type=obj_type, reference=reference
            )
        elif b3 is not None:
            self.track_3d.update(b3)

    def predict(self, t):
        if self.track_2d is not None:
            self.track_2d.predict(t)
        if self.track_3d is not None:
            self.track_3d.predict(t)

    def as_object(self):
        if self.track_3d is not None:
            return self.track_3d.as_object()
        else:
            raise RuntimeError("No 3d track to convert to object")

    def change_reference(self, reference, inplace: bool):
        if inplace:
            if self.track_2d is not None:
                self.track_2d.change_reference(reference, inplace=inplace)
            if self.track_3d is not None:
                self.track_3d.change_reference(reference, inplace=inplace)
        else:
            raise NotImplementedError
