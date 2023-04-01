import numpy as np
from filterpy.kalman import KalmanFilter

from avstack.datastructs import DataContainer
from avstack.environment.objects import VehicleState
from avstack.geometry.bbox import Box2D, Box3D, get_box_from_line


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


class _BoxTrackBase:
    def __init__(self, t0, ID, obj_type, t=None, coast=0, n_updates=1, age=0) -> None:
        self.obj_type = obj_type
        self.coast = coast
        self.active = True
        self.n_updates = n_updates
        self.age = age
        self.ID = ID
        self.t0 = t0
        self.t = t0 if t is None else t

    @property
    def x(self):
        return self.kf.x

    @property
    def P(self):
        return self.kf.P

    def set_F(self, t):
        raise NotImplementedError

    def set_Q(self, t):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"BasicBoxTrack w/ {self.n_updates} updates: {self.x}"

    def update(self):
        raise NotImplementedError

    def predict(self, t):
        self.set_F(t)
        self.set_Q(t)
        self.kf.predict()
        self.t_last_predict = t
        self.age += 1
        self.coast += 1


class BasicBoxTrack3D(_BoxTrackBase):
    ID_counter = 0

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
        if ID_force is None:
            ID = BasicBoxTrack3D.ID_counter
            BasicBoxTrack3D.ID_counter += 1
        else:
            ID = ID_force
        super().__init__(t0, ID, obj_type, t, coast, n_updates, age)
        self.origin = box3d.origin
        self.where_is_t = box3d.where_is_t

        # -- initialize filter
        self.kf = KalmanFilter(dim_x=9, dim_z=6)
        if P is None:
            P = np.diag([5, 5, 5, 2, 2, 2, 10, 10, 10]) ** 2
        self.kf.P = P
        self.kf.H = np.zeros((6, 9))
        self.kf.H[:6, :6] = np.eye(6)
        self.kf.R = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) ** 2
        if v is None:
            v = np.array([0, 0, 0])
        self.kf.x = np.array(
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
        self.q = box3d.q
        self.t_last_predict = t0

    @property
    def position(self):
        return self.kf.x[:3]

    @property
    def velocity(self):
        return self.kf.x[6:9]

    @property
    def box3d(self):
        as_l = [el for el in self.kf.x[3:6]]
        as_l.extend([el for el in self.kf.x[:3]])
        as_l.append(self.q)
        return Box3D(as_l, self.origin, where_is_t=self.where_is_t)

    @property
    def box(self):
        return self.box3d

    @property
    def yaw(self):
        return self.box3d.yaw

    def set_F(self, t):
        dt = t - self.t_last_predict
        self.kf.F = np.eye(9)
        self.kf.F[:3, 6:9] = dt * np.eye(3)

    def set_Q(self, t):
        dt = t - self.t_last_predict
        self.kf.Q = (np.diag([2, 2, 2, 0.5, 0.5, 0.5, 3, 3, 3]) * dt) ** 2

    def update(self, box3d):
        if box3d.origin != self.origin:
            box3d.change_origin(self.origin)
        if self.where_is_t != box3d.where_is_t:
            raise NotImplementedError(
                "Differing t locations not implemented: {}, {}".format(
                    self.where_is_t, box3d.where_is_t
                )
            )
        self.coast = 0
        self.n_updates += 1
        det = np.array([box3d.t[0], box3d.t[1], box3d.t[2], box3d.h, box3d.w, box3d.l])
        self.kf.update(det)
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
        v_str = " ".join(map(str, self.kf.x[6:9]))
        P_str = " ".join(map(str, self.kf.P.ravel()))
        return (
            f"boxtrack3d {self.obj_type} {self.t0} {self.t} {self.ID} "
            f"{self.coast} {self.n_updates} {self.age} "
            f"{len(self.kf.x)} {v_str} {P_str} {self.box3d.format_as_string()}"
        )


class BasicBoxTrack2D(_BoxTrackBase):
    ID_counter = 0

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
        if ID_force is None:
            ID = BasicBoxTrack2D.ID_counter
            BasicBoxTrack2D.ID_counter += 1
        else:
            ID = ID_force
        super().__init__(t0, ID, obj_type, t, coast, n_updates, age)
        self.calibration = box2d.calibration
        # self.source_identifier = box2d.source_identifier

        # -- initialize filter
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        if P is None:
            P = np.diag([10, 10, 10, 10, 10, 10]) ** 2
        self.kf.P = P
        self.kf.H = np.zeros((4, 6))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R = np.diag([5, 5, 5, 5]) ** 2
        if v is None:
            v = np.array([0, 0])
        self.kf.x = np.array(
            [
                box2d.center[0],
                box2d.center[1],
                box2d.w,
                box2d.h,
                v[0],
                v[1],
            ]
        )
        self.t_last_predict = t0

    @property
    def position(self):
        return self.kf.x[:2]

    @property
    def velocity(self):
        return self.kf.x[4:6]

    @property
    def w(self):
        return self.kf.x[2]

    @property
    def h(self):
        return self.kf.x[3]

    @property
    def xmin(self):
        return self.position[0] - self.w / 2

    @property
    def xmax(self):
        return self.position[0] + self.w / 2

    @property
    def ymin(self):
        return self.position[1] - self.h / 2

    @property
    def ymax(self):
        return self.position[1] + self.h / 2

    @property
    def box2d(self):
        return Box2D([self.xmin, self.ymin, self.xmax, self.ymax], self.calibration)

    @property
    def box(self):
        return self.box2d

    def set_F(self, t):
        dt = t - self.t_last_predict
        self.kf.F = np.eye(6)
        self.kf.F[:2, 4:6] = dt * np.eye(2)

    def set_Q(self, t):
        dt = t - self.t_last_predict
        self.kf.Q = (np.diag([2, 2, 2, 2, 2, 2]) * dt) ** 2

    def update(self, box2d):
        # if box2d.source_identifier != self.source_identifier:
        #     raise NotImplementedError("Sensor sources must be the same for now")
        self.coast = 0
        self.n_updates += 1
        det = np.array([box2d.center[0], box2d.center[1], box2d.w, box2d.h])
        self.kf.update(det)

    def format_as_string(self):
        v_str = " ".join(map(str, self.kf.x[4:6]))
        P_str = " ".join(map(str, self.kf.P.ravel()))
        return (
            f"boxtrack2d {self.obj_type} {self.t0} {self.t} {self.ID} "
            f"{self.coast} {self.n_updates} {self.age} "
            f"{len(self.kf.x)} {v_str} {P_str} {self.box2d.format_as_string()}"
        )


class BasicJointBoxTrack(_BoxTrackBase):
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
