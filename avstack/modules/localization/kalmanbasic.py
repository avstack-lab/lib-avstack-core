import numpy as np
import quaternion

from avstack.config import ALGORITHMS
from avstack.geometry import Attitude, GlobalOrigin3D, Position, Velocity
from avstack.geometry import transformations as tforms
from avstack.sensors import DataBuffer, ImuBuffer

from .base import _LocalizationAlgorithm


def KF_linear_update(xp, Pp, y, H, R, S=None):
    if S is None:
        S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)
    x = xp + K @ y
    P = (np.eye(Pp.shape[0]) - K @ H) @ Pp
    return x, P, K, S


def KF_linear_propagate(x, P, F, Q):
    xp = F @ x
    Pp = F @ P @ F.T + Q
    return xp, Pp


# =============================================================
# WITHOUT IMU
# =============================================================


@ALGORITHMS.register_module()
class BasicGpsKinematicKalmanLocalizer(_LocalizationAlgorithm):
    """
    Extremely simple state estimator

    Uses only GPS measurments with kinematic state prediction

    Does not necessarily perform state estimation in any coordinate frame.
    ECEF/ENU/NED all the same.
    """

    def __init__(
        self,
        t_init,
        ego_init,
        rate,
        integrity,
        integrity_delay=5,
        predict_model="cv",
        sigma_m=1,
        tau_m=5,
        reference=GlobalOrigin3D,
    ):
        super().__init__(t_init, ego_init, rate)
        self.ego_template = ego_init
        self.integrity_delay = integrity_delay
        vR = 5**2
        vV = 10**2
        vA = 4**2

        # Integrity
        self.integrity = integrity
        self.reference = reference

        # Dynamics model
        self.predict_model = predict_model
        self.q = 2 * sigma_m**2 * tau_m
        if self.predict_model == "cv":
            self.n_states = 6
            self.P0 = np.diag([vR, vR, vR, vV, vV, vV])
            self.F = lambda dt: np.kron(np.array([[1, dt], [0, 1]]), np.eye(3))
            self.Q = lambda dt: np.kron(
                self.q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]),
                np.eye(3),
            )
        else:
            raise NotImplementedError

        # Measurement model -- only GPS
        self.n_meas = 3
        self.H = np.zeros((self.n_meas, self.n_states))
        np.fill_diagonal(self.H, 1)

        # initialize
        self._t_last_update = None
        if ego_init is not None:
            x0 = np.concatenate((ego_init.position.x, np.zeros((3,))), axis=0)
            self._initialize(t_init, x0)

    def _initialize(self, t0, x0):
        self.initialized = True
        self.t0 = t0
        self.x0 = x0
        self.t_met = 0
        self._t_last_update = t0
        self.t = t0
        self.x = x0
        self.P = self.P0

    def _propagate(self, t):
        dt = t - self.t
        assert dt >= 0, dt
        self.x, self.P = KF_linear_propagate(self.x, self.P, self.F(dt), self.Q(dt))
        self.t = t
        self.t_met += dt

    def _update(self, t, z, R):
        self._t_last_update = t
        self.y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        if (self.integrity is not None) and (t - self.t0 > self.integrity_delay):
            pass_integrity = self.integrity(y=self.y, S=S)
        else:
            pass_integrity = True
        if pass_integrity:
            self.x, self.P, self.K, self.S = KF_linear_update(
                self.x, self.P, self.y, self.H, R, S=S
            )
        else:
            print("msmt rejected")

    def execute(self, t, gps_data, *args, **kwargs):
        # Pull off data
        if gps_data is not None:
            assert (
                np.linalg.norm(t - gps_data.timestamp) <= 2e-1
            ), f"{t}, {gps_data.timestamp}"
            t = gps_data.timestamp
            if self.attitude is not None:
                z = gps_data.data["z"] - self.attitude.R.T @ gps_data.levar
            else:
                z = gps_data.data["z"]
            R = gps_data.data["R"]
        else:
            z = R = None

        # Run filter
        if (z is None) and (not self.initialized):
            ego_loc = None
            if self._t_last_update is None:
                self._t_last_update = t
        else:
            if (z is None) and (self.initialized):
                self._propagate(t)
            elif not self.initialized:
                self._initialize(t, np.concatenate((z, np.zeros((3,)))))
            else:
                self._propagate(t)
                self._update(t, z, R)

            # Make vehicle state object
            self.position = Position(self.x[:3], self.reference)  # in ENU
            self.velocity = Velocity(self.x[3:6], self.reference)  # in ENU
            self.acceleration = None
            if self.velocity.norm() > 0:
                forward = self.velocity.x / np.linalg.norm(self.velocity.x)  # in ENU
            elif self.attitude is None:
                forward = np.array([1, 0, 0])
            else:
                forward = self.attitude.forward_vector
            up = np.array([0, 0, 1])
            left = np.cross(up, forward)
            up = np.cross(forward, left)
            R_enu_2_body = np.array([forward, left, up])  # R_enu_2_body
            self.attitude = Attitude(
                tforms.transform_orientation(R_enu_2_body, "dcm", "quat"),
                self.reference,
            )  # from velocity
            self.angular_velocity = None
            self.ego_template.set(
                self.t,
                self.position,
                self.box,
                self.velocity,
                self.acceleration,
                self.attitude,
                self.angular_velocity,
            )
            ego_loc = self.ego_template

        # Error checking
        if (t - self._t_last_update) > 10:
            raise RuntimeError("Has not been updated in too long.")

        return ego_loc


# =============================================================
# WITH IMU GYRO AND ACCEL
# =============================================================


@ALGORITHMS.register_module()
class BasicGpsImuErrorStateKalmanLocalizer(_LocalizationAlgorithm):
    """
    Simple state estimator

    Uses GPS and IMU measurements to perform navigation
    """

    def __init__(self, t_init, ego_init, integrity, reference):
        rate = 1e4  # some large number
        super().__init__(t_init, ego_init, rate)
        self.ego_template = ego_init
        self.reference = reference
        self.n_states = 9  # p-v-attitude
        vR = 10**2
        vV = 20**2
        vAtt = 1e-3**2
        self.P0 = np.diag([vR, vR, vR, vV, vV, vV, vAtt, vAtt, vAtt])
        self.min_prop_rate = 5  # at least propagate at this rate
        self.cov_prop_rate = 2  # ONLY propagate covariance at this rate

        # Measurement model -- only GPS
        self.n_meas = 3
        self.H = np.zeros((self.n_meas, self.n_states))
        np.fill_diagonal(self.H, 1)

        # initialize
        self._t_last_update = None
        eulers = tforms.transform_orientation(ego_init.attitude.q, "quat", "euler")
        x0 = np.concatenate((ego_init.position.x, np.zeros((3,)), eulers), axis=0)
        self._initialize(t_init, x0)
        self.imu_buffer = ImuBuffer()
        self.gps_buffer = DataBuffer()
        self.n_msmt_updates = 0
        self.n_cov_prop = 0

    @property
    def rE(self):
        return self.x[:3]

    @rE.setter
    def rE(self, rE):
        self.x[:3] = rE

    @property
    def vE(self):
        return self.x[3:6]

    @vE.setter
    def vE(self, vE):
        self.x[3:6] = vE

    @property
    def qN2B(self):
        return tforms.transform_orientation(self.x[6:9], "euler", "quat")

    @qN2B.setter
    def qN2B(self, qN2B):
        self.x[6:9] = tforms.transform_orientation(qN2B, "quat", "euler")

    def _initialize(self, t0, x0):
        self.initialized = True
        self.t0 = t0
        self.x0 = x0
        self.xerr = np.zeros((self.n_states,))
        self.t_met = 0
        self._t_last_update = t0
        self._t_last_cov_prop = t0
        self.t = t0
        self.x = x0
        self.P = self.P0

    def _propagate(self, t):
        """Apply everything in imu integrator up to time t"""
        if len(self.imu_buffer) == 0:
            return
        imu_data = self.imu_buffer.integrate_up_to(t)

        # Pull off state values
        rE = self.rE  # if you get an error here, you must not be initialized yet
        vE = self.vE
        qN2B = self.qN2B
        qE2N = tforms.get_q_ecef_to_ned((rE, "ecef"))
        qB2E = (qN2B * qE2N).conjugate()

        # --------------------
        # Update covariance
        # --------------------
        if (t - self._t_last_cov_prop) >= (1 / self.cov_prop_rate):
            F = PhiFromIMU(
                imu_data.data["dt"], imu_data.data["dv"], imu_data.data["dth"], rE, qB2E
            )
            Q = QFromImu(imu_data.data["dt"], imu_data.data["R"])
            self.P = F @ self.P @ F.T + Q
            self.xerr = F @ self.xerr
            self.t_cov_prop = t
            self.n_cov_prop += 1

        # --------------------
        # Update states
        # --------------------
        self.t += imu_data.data["dt"]

        # ----- velocity
        v_old = self.vE
        self.vE = v_old + imu_data.data["dv"]
        v_avg = (v_old + vE) / 2

        # ----- position
        self.rE = self.rE + v_avg * imu_data.data["dt"]

        # ----- attitude
        q_Bkm1_2_Bk = quaternion.from_rotation_vector(imu_data.data["dth"])
        q_Bkm1_2_E = qB2E
        qB2E = q_Bkm1_2_E * q_Bkm1_2_Bk.conjugate()
        qE2N = quaternion.from_rotation_matrix(
            tforms.get_R_ecef_to_ned((self.rE, "ecef"))
        )
        self.qN2B = (qE2N * qB2E).conjugate()

    def _update(self, gps_data):
        """Update with gps measurement at time t"""
        assert (
            abs(gps_data.timestamp - self.t) < 1e-2
        ), f"{gps_data.timestamp}, {self.t}"
        z, R = gps_data.data["z"], gps_data.data["R"]
        y = z - self.rE  # z - H @ self.x

        # -- update error state
        self.xerr, self.P, self._K, self._S = KF_linear_update(
            self.xerr, self.P, y, self.H, R
        )
        # -- correct error states
        self.rE += self.xerr[0:3]
        self.vE += self.xerr[3:6]
        q_Bkm1_2_Bk = quaternion.from_rotation_vector(self.xerr[6:9])
        self.qN2B = q_Bkm1_2_Bk * self.qN2B
        self.xerr = np.zeros((9,))

        self.t = gps_data.timestamp
        self.n_msmt_updates += 1

    def execute(self, t, gps_data, imu_data, *args, **kwargs):
        # Pull off data
        # -- imu data
        if imu_data is not None:
            self.imu_buffer.push(imu_data.timestamp, imu_data)

        # -- gps data
        if gps_data is not None:
            self.gps_buffer.push(gps_data.timestamp, gps_data)

        # -- process all gps measurements for low-rate
        while not self.gps_buffer.empty():
            t_gps, gps_data = self.gps_buffer.pop(with_priority=True)
            if t_gps >= self.t:
                self._propagate(t_gps)
                self._update(gps_data)

        # -- ensure propagation up to date for high-rate
        if abs(t - self.t) > 1e-1:
            raise RuntimeError(f"Time out of date {t} {self.t}")

        # -- set outputs
        position = Position(self.rE, self.reference)
        velocity = Velocity(self.vE, self.reference)
        attitude = Attitude(self.qN2B, self.reference)
        acceleration = None
        angular_velocity = None
        self.ego_template.set(
            self.t,
            position,
            self.box,
            velocity,
            acceleration,
            attitude,
            angular_velocity,
        )

        return self.ego_template


def PhiFromIMU(dt, dv, dth, rE, qB2E):
    """
    Get state transition and process noise matrices from IMU data
    """
    grav_grad = np.zeros((3, 3))
    fRaw_i_i = dv / dt
    OmRaw_b_b = dth / dt

    # Frame transformations
    ce2i = np.eye(3)
    ci2b = quaternion.as_rotation_matrix(qB2E).T @ ce2i.T

    # State transitions
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)  # kinematics
    A[3:6, 0:3] = grav_grad  # gravity gradient
    A[3:6, 6:9] = -skew(fRaw_i_i)

    # -----------------
    # Accel/Gyro state transitions
    # TODO
    # -----------------

    # Get Phi and Q matrices
    Phi = A_to_STM(A, dt)
    return Phi


def QFromImu(dt, R):
    # Process noise
    B = np.zeros((9, 6))
    B[3:6, 0:3] = np.eye(3)
    B[6:9, 3:6] = np.eye(3)
    Q = B @ (R * dt) @ B.T
    return Q


def A_to_STM(A, dt, order=2):
    """Get state transition matrix from dynamics matrix, A"""
    assert order <= 2
    assert order >= 0

    STM = np.eye(A.shape[0])

    if order >= 1:
        STM += 0.5 * np.linalg.matrix_power(A, 2) * dt**2

    if order >= 2:
        STM += (1 / 6) * np.linalg.matrix_power(A, 3) * dt**3

    return STM


def skew(v):
    """Make skew-symmetric matrix from vector"""
    v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return v
