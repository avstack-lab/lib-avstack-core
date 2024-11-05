import pickle

import numpy as np
import quaternion

from avstack.environment.objects import VehicleState
from avstack.geometry import Attitude, GlobalOrigin3D, Position, Velocity
from avstack.geometry import transformations as tforms
from avstack.modules import localization
from avstack.sensors import GpsData, ImuData


imu_ID = 1
gps_ID = 2


def run_gps_localization(L, ego_true):
    t = 0
    dt = 0.1
    t_max = 100
    A = np.array([[1, dt, 1 / 2 * dt**2], [0, 1, dt], [0, 0, 1]])
    F = lambda dt: np.kron(A, np.eye(3))
    rs = 4
    R = rs**2 * np.eye(3)
    last_ego = None
    frame = 0
    levar = np.zeros((3,))
    while t <= t_max:
        ego_true = F(dt) @ ego_true
        z = ego_true[:3] + rs * np.random.randn(3)
        gps_data = GpsData(
            t, frame, {"z": z, "R": R}, GlobalOrigin3D, gps_ID, levar=levar
        )
        ego_est = L(t, gps_data)
        if ego_est is not None:
            last_ego = ego_est
        t += dt
        frame += 1

    assert last_ego is not None
    assert np.linalg.norm(last_ego.position.x - ego_true[:3]) <= 6


def run_gps_imu_localization(L, trajectory, origin_ecef):
    imu_rate = np.inf
    imu_interval = 1.0 / imu_rate
    gps_rate = 10
    gps_interval = 1.0 / gps_rate
    t0 = trajectory["t"][0]
    t_last_imu = t0
    t_last_gps = t0

    # noise
    rs_dv = 0
    rs_dth = 0
    R_imu = np.diag([rs_dv, rs_dv, rs_dv, rs_dth, rs_dth, rs_dth])
    rs_gps = 4
    R_gps = rs_gps**2 * np.eye(3)
    levar = np.zeros((3,))
    # run trajectory
    last_ego = None
    for i in range(len(trajectory["t"])):
        if i > 0:
            r_k = origin_ecef + np.array(
                [trajectory["x"][i], trajectory["y"][i], trajectory["z"][i]]
            )
            v_k = np.array(
                [trajectory["vx"][i], trajectory["vy"][i], trajectory["vz"][i]]
            )
            q_N_2_Bk = quaternion.from_rotation_vector(
                np.array([0, 0, np.pi / 180 * trajectory["yaw"][i]])
            )
            q_E_2_N = tforms.get_q_ecef_to_ned((r_k, "ecef"))
            q_E_2_Bk = q_N_2_Bk * q_E_2_N

            # -- make sensor data
            t = trajectory["t"][i]
            if (t - t_last_imu) >= (imu_interval - 1e-5):
                dt = t - t_last_imu
                dv = v_k - v_km1
                q_Bkm1_2_Bk = q_E_2_Bk * q_E_2_Bkm1.conjugate()
                dth = tforms.transform_orientation(q_Bkm1_2_Bk, "quat", "euler")
                imu_data = ImuData(
                    t,
                    i,
                    {"dt": dt, "dv": dv, "dth": dth, "R": R_imu},
                    GlobalOrigin3D,
                    imu_ID,
                )
                t_last_imu = t
            else:
                imu_data = None
            if (t - t_last_gps) >= (gps_interval - 1e-5):
                z = r_k + rs_gps * np.random.randn(3)
                gps_data = GpsData(
                    t, i, {"z": z, "R": R_gps}, GlobalOrigin3D, gps_ID, levar=levar
                )
                t_last_gps = t
            else:
                gps_data = None

            # -- process sensor data
            ego_est = L(t, gps_data, imu_data)
            if ego_est is not None:
                last_ego = ego_est
        r_km1 = origin_ecef + np.array(
            [trajectory["x"][i], trajectory["y"][i], trajectory["z"][i]]
        )
        v_km1 = np.array(
            [trajectory["vx"][i], trajectory["vy"][i], trajectory["vz"][i]]
        )
        att_km1 = np.array([0, 0, np.pi / 180 * trajectory["yaw"][i]])
        q_E_2_Bkm1 = quaternion.from_rotation_vector(att_km1)

    assert last_ego is not None
    assert np.linalg.norm(last_ego.position.x - r_km1) <= 2
    assert np.linalg.norm(last_ego.velocity.x - v_km1) <= 2
    # assert np.linalg.norm(last_ego.attitude.q - att_km1) <= 1e-2  # this doesn't work yet


def test_basic_kalman_filter_with_init():
    reference = GlobalOrigin3D
    ego_init = VehicleState("Car")
    rate = 10
    t_init = 0.0
    ego_true = np.array([1000, 100, -2455, 10, -20, 5, -0.1, -0.2, 2])  # p-v-a
    att = np.eye(3)
    ego_init.set(
        t=0,
        position=Position(ego_true[:3], reference),
        box=None,
        velocity=Velocity(ego_true[3:6], reference),
        acceleration=None,
        attitude=Attitude(att, reference),
        angular_velocity=None,
    )
    L = localization.BasicGpsKinematicKalmanLocalizer(
        t_init, ego_init, rate, integrity=None
    )
    run_gps_localization(L, ego_true)


def tests_gps_imu_kalman_filter():
    with open("tests/data/vehicle_truth_data_v1.p", "rb") as f:
        trajectory = pickle.load(f)
    reference = GlobalOrigin3D
    ego_init = VehicleState("Car")
    t_init = trajectory["t"][0]
    # random origin
    origin_ecef = tforms.transform_point(np.array([0, 0, 0]), "lla", "ecef")[:, 0]
    pos = origin_ecef + np.array(
        [trajectory["x"][0], trajectory["y"][0], trajectory["z"][0]]
    )
    vel = np.array([trajectory["vx"][0], trajectory["vy"][0], trajectory["vz"][0]])
    att = tforms.transform_orientation(
        np.array([0, 0, np.pi / 180 * trajectory["yaw"][0]]), "euler", "dcm"
    )
    ego_init.set(
        t=0,
        position=Position(pos, reference),
        box=None,
        velocity=Velocity(vel, reference),
        acceleration=None,
        attitude=Attitude(att, reference),
        angular_velocity=None,
    )
    L = localization.BasicGpsImuErrorStateKalmanLocalizer(
        t_init, ego_init, reference=reference, integrity=None
    )
    run_gps_imu_localization(L, trajectory, origin_ecef)
