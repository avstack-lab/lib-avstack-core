import numpy as np

from avstack.datastructs import DataContainer
from avstack.environment.objects import VehicleState
from avstack.geometry.bbox import Box2D, Box3D, get_box_from_line
from avstack.geometry.transformations import xyzvel_to_razelrrt, razelrrt_to_xyzvel, spherical_to_cartesian, cartesian_to_spherical


def format_data_container_as_string(DC):
    trks_strings = " ".join(["TRACK " + trk.format_as_string() for trk in DC.data])
    return (
        f"datacontainer {DC.frame} {DC.timestamp} {DC.source_identifier} "
        f"{trks_strings}"
    )


def get_track_from_line(line):
    items = line.split()
    trk_type = items[0]
    n_prelim = 9
    obj_type, t0, t, ID, coast, n_updates, age, n_dim = items[1:n_prelim]
    if "box" in trk_type:
        if trk_type == "boxtrack3d":
            n_add = 3
            factory = BasicBoxTrack3D
        elif trk_type == "boxtrack2d":
            n_add = 2
            factory = BasicBoxTrack2D
        elif trk_type == "boxtrack2d3d":
            raise NotImplementedError
        else:
            raise NotImplementedError(trk_type)
        v = items[n_prelim : (n_prelim + n_add)]
        P = items[(n_prelim + n_add) : (n_prelim + n_add + int(n_dim) ** 2)]
        box = items[(n_prelim + n_add + int(n_dim) ** 2) : :]
        v = np.array([float(vi) for vi in v])
        P = np.reshape(np.array([float(pi) for pi in P]), (int(n_dim), int(n_dim)))
        box = get_box_from_line(" ".join(box))
        trk = factory(
            float(t0),
            box,
            obj_type,
            ID_force=int(ID),
            v=v,
            P=P,
            t=float(t),
            coast=float(coast),
            n_updates=int(n_updates),
            age=float(age),
        )
    else:
        raise NotImplementedError(trk_type)
    return trk


def get_data_container_from_line(line, identifier_override=None):
    items = line.split()
    assert items[0] == "datacontainer"
    frame = int(items[1])
    timestamp = float(items[2])
    source_identifier = items[3]
    detections = [get_track_from_line(det) for det in line.split("TRACK")[1:]]
    source_identifier = (
        source_identifier if identifier_override is None else identifier_override
    )
    return DataContainer(frame, timestamp, detections, source_identifier)


class _TrackBase:
    ID_counter = 0
    def __init__(self, t0, x, P, obj_type, ID_force=None, t=None, coast=0, n_updates=1, age=0) -> None:
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
        self.ID = ID
        self.t0 = t0
        self.t = t0 if t is None else t
        self.t_last_predict = self.t
        self.x = x
        self.P = P

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


class XyzFromRazelTrack(_TrackBase):
    """Tracking on razel measurements
    
    IMPORTANT: assumes we are in a sensor-relative coordinate frame.
    This assumption allows us to say that the sensor is always 
    facing 'forward' which simplifies the calculations. If we are
    wanting to track in some other coordinate frame, we will need to
    explicitly incorporate the sensor's pointing angle and position
    offset in the calculations."""
    def __init__(
        self,
        t0,
        razel,
        obj_type,
        ID_force=None,
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
        x = np.array([*spherical_to_cartesian(razel), 0, 0, 0])
        if P is None:
            P = np.diag([5, 5, 5, 10, 10, 10]) ** 2
        super().__init__(t0, x, P, obj_type, ID_force, t, coast, n_updates, age)

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:6]
    
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
        H[1, 1] =  x[0] / r2d**2
        H[2, 0] = -x[0]*x[2] / (r**2 * r2d)
        H[2, 1] = -x[1]*x[2] / (r**2 * r2d)
        H[2, 2] =  r2d / r**2
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
        return np.array([[1, 0, 0, dt,  0,  0],
                         [0, 1, 0,  0, dt,  0],
                         [0, 0, 1,  0,  0, dt],
                         [0, 0, 0,  1,  0,  0],
                         [0, 0, 0,  0,  1,  0],
                         [0, 0, 0,  0,  0,  1]])

    @staticmethod
    def f(x, dt):
        """State propagation function"""
        return np.array([x[0] + x[3]*dt,
                         x[1] + x[4]*dt,
                         x[2] + x[5]*dt,
                         x[3], x[4], x[5]])
    
    @staticmethod
    def Q(dt):
        """This is most definitely not optimal and should be tuned in the future"""
        return (np.diag([2, 2, 2, 2, 2, 2]) * dt) ** 2
    
    def update(self, z, R=np.diag([10, 1e-2, 5e-2])**2):
        self._update(z, R)


class XyzFromRazelRrtTrack(_TrackBase):
    """Tracking on razel rrt measurements
    
    NOTE: to reduce the nonlinearities of the range rate,
    we use the pseudo-measurement of range * rrt.
    See https://arxiv.org/pdf/1412.5524.pdf for an explanation

    IMPORTANT: assumes we are in a sensor-relative coordinate frame.
    This assumption allows us to say that the sensor is always 
    facing 'forward' which simplifies the calculations. If we are
    wanting to track in some other coordinate frame, we will need to
    explicitly incorporate the sensor's pointing angle and position
    offset in the calculations."""
    def __init__(
        self,
        t0,
        razelrrt,
        obj_type,
        ID_force=None,
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
        x = razelrrt_to_xyzvel(razelrrt)
        if P is None:
            # Note the uncertainty on transverse velocities is larger (see note above)
            v_unit = x[:3] / razelrrt[0]
            rrt_p_max = 10  # complete uncertainty gives 10, total certainty gives 2
            rrt_p_min = 2
            P = np.diag([5, 5, 5, *np.maximum(rrt_p_min, rrt_p_max * (1-v_unit))]) ** 2
        super().__init__(t0, x, P, obj_type, ID_force, t, coast, n_updates, age)

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:6]
    
    @property
    def rrt(self):
        return self.velocity @ self.position / np.linalg.norm(self.position)
    
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
        H[1, 1] =  x[0] / r2d**2
        H[2, 0] = -x[0]*x[2] / (r**2 * r2d)
        H[2, 1] = -x[1]*x[2] / (r**2 * r2d)
        H[2, 2] =  r2d / r**2
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
        zhat[3] = zhat[0]*zhat[3]
        return zhat
    
    @staticmethod
    def F(x, dt):
        """Partial derivative of the propagation function w.r.t. x at x hat"""
        return np.array([[1, 0, 0, dt,  0,  0],
                         [0, 1, 0,  0, dt,  0],
                         [0, 0, 1,  0,  0, dt],
                         [0, 0, 0,  1,  0,  0],
                         [0, 0, 0,  0,  1,  0],
                         [0, 0, 0,  0,  0,  1]])

    @staticmethod
    def f(x, dt):
        """State propagation function"""
        return np.array([x[0] + x[3]*dt,
                         x[1] + x[4]*dt,
                         x[2] + x[5]*dt,
                         x[3], x[4], x[5]])
    
    @staticmethod
    def Q(dt):
        """This is most definitely not optimal and should be tuned in the future"""
        return (np.diag([2, 2, 2, 2, 2, 2]) * dt) ** 2
    
    def update(self, z, R=np.diag([10, 1e-2, 5e-2, 2])**2):
        """Construct the pseudo measurement for range rate"""
        z[3] = z[0]*z[3]
        self._update(z, R)


class BasicBoxTrack3D(_TrackBase):
    def __init__(
        self,
        t0,
        box3d,
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
        super().__init__(t0, x, P, obj_type, ID_force, t, coast, n_updates, age)
        self.origin = box3d.origin
        self.where_is_t = box3d.where_is_t
        self.q = box3d.q

    @staticmethod
    def f(x, dt):
        return np.array([x[0] + x[6]*dt,
                         x[1] + x[7]*dt,
                         x[2] + x[8]*dt,
                         x[3], x[4], x[5],
                         x[6], x[7], x[8]])
    
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
        return (np.diag([2, 2, 2, 0.5, 0.5, 0.5, 3, 3, 3]) * dt) ** 2

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[6:9]

    @property
    def box3d(self):
        as_l = [el for el in self.x[3:6]]
        as_l.extend([el for el in self.x[:3]])
        as_l.append(self.q)
        return Box3D(as_l, self.origin, where_is_t=self.where_is_t)

    @property
    def box(self):
        return self.box3d

    @property
    def yaw(self):
        return self.box3d.yaw

    def update(self, box3d, R=np.diag([1, 1, 1, 0.5, 0.5, 0.5])**2):
        if box3d.origin != self.origin:
            box3d.change_origin(self.origin)
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
            attitude=self.q,
            angular_velocity=None,
            origin=self.origin,
        )
        return vs

    def format_as_string(self):
        v_str = " ".join(map(str, self.x[6:9]))
        P_str = " ".join(map(str, self.P.ravel()))
        return (
            f"boxtrack3d {self.obj_type} {self.t0} {self.t} {self.ID} "
            f"{self.coast} {self.n_updates} {self.age} "
            f"{len(self.x)} {v_str} {P_str} {self.box3d.format_as_string()}"
        )


class BasicBoxTrack2D(_TrackBase):

    def __init__(
        self,
        t0,
        box2d,
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
        super().__init__(t0, x, P, obj_type, ID_force, t, coast, n_updates, age)
        self.calibration = box2d.calibration

    @property
    def position(self):
        return self.x[:2]

    @property
    def velocity(self):
        return self.x[4:6]

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
        return np.array([x[0] + x[4]*dt,
                         x[1] + x[5]*dt,
                         x[2], x[3],
                         x[4], x[5]])
    
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

    def update(self, box2d, R=np.diag([5, 5, 5, 5])**2):
        # if box2d.source_identifier != self.source_identifier:
        #     raise NotImplementedError("Sensor sources must be the same for now")
        det = np.array([box2d.center[0], box2d.center[1], box2d.w, box2d.h])
        self._update(det, R)

    def format_as_string(self):
        v_str = " ".join(map(str, self.x[4:6]))
        P_str = " ".join(map(str, self.P.ravel()))
        return (
            f"boxtrack2d {self.obj_type} {self.t0} {self.t} {self.ID} "
            f"{self.coast} {self.n_updates} {self.age} "
            f"{len(self.x)} {v_str} {P_str} {self.box2d.format_as_string()}"
        )


class BasicJointBoxTrack(_TrackBase):
    def __init__(self, t0, box2d, box3d, obj_type):
        self.track_2d = (
            BasicBoxTrack2D(t0, box2d, obj_type) if box2d is not None else None
        )
        self.track_3d = (
            BasicBoxTrack3D(t0, box3d, obj_type) if box3d is not None else None
        )

    @property
    def ID(self):
        return self.track_3d.ID

    @property
    def box(self):
        return self.box3d

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"BasicJointBoxTrack w/ {self.track_2d.n_updates} 2D n_updates: {self.box2d}; {self.track_3d.n_updates} 3D n_updates: {self.box3d}"

    def update(self, box, obj_type):
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
            self.track_2d = BasicBoxTrack2D(self.track_3d.t, b2, obj_type)
        elif b2 is not None:
            self.track_2d.update(b2)

        # -- update 3d
        if self.track_3d is None and b3 is not None:
            self.track_3d = BasicBoxTrack3D(self.track_2d.t, b3, obj_type)
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

    def format_as(self, format_):
        return self.as_object().format_as(format_)
