import numpy as np

from avstack.geometry import (
    BoundingBox3D,
    Pose,
    Rotation,
    Vector,
    WorldFrame,
    conversions,
)
from avstack.modules import control
from avstack.modules.planning import Waypoint, WaypointPlan
from avstack.objects import VehicleState


def test_pid_base():
    # -- p only
    pid = control.PIDBase(1.0, 0, 0)
    c1 = pid(1.0, -5)
    c2 = pid(2.0, -10)
    c3 = pid(3.0, 20, 0, 4.5)
    assert c1 == -5
    assert c2 == -10
    assert c3 == 4.5


def test_vehicle_pid_control():
    pos = Vector(np.zeros((3,)), WorldFrame)
    rot = Rotation(np.quaternion(1), WorldFrame)
    box_obj = BoundingBox3D(pos, rot, [2, 2, 5])
    ego_state = VehicleState("car")
    lat = {"K_P": 1.5, "K_D": 0.02, "K_I": 0.01}
    lon = {"K_P": 1.0, "K_D": 0.01, "K_I": 0.05}
    controller = control.vehicle.VehiclePIDController(lat, lon)

    # Initialize
    t = 0
    tmax = 10
    dt = 0.1
    speed_target = 20
    imsize = [200, 200]
    lane_width = 3.7
    yaw = lambda t: 1 / 4 * np.sin(t / 3)

    pos = Vector(np.zeros(3), WorldFrame)
    rot = Rotation(
        conversions.transform_orientation([0, 0, yaw(t)], "euler", "dcm"),
        WorldFrame,
    )
    v = 10
    vel = Vector(v * rot.forward_vector, WorldFrame)
    acc = Vector(np.zeros(3), WorldFrame)
    ang_vel = Vector(np.zeros(3), WorldFrame)

    # Run looop
    pos_all = []
    WP = WaypointPlan()
    while t < tmax:
        t += dt
        # -- reference
        rot = Rotation(
            conversions.transform_orientation([0, 0, yaw(t)], "euler", "dcm"),
            WorldFrame,
        )
        vel_new = Vector(v * rot.forward_vector, WorldFrame)
        pos = pos + (vel + vel_new) * dt / 2
        vel = vel_new

        # -- ego state
        pos_ego = pos + np.random.randn(3)
        vel_ego = vel + np.random.randn(3)
        acc_ego = acc
        rot_ego = Rotation(
            conversions.transform_orientation(
                [0, 0, yaw(t) + np.random.randn(1)], "euler", "dcm"
            ),
            WorldFrame,
        )
        ego_state.set(t, pos_ego, box_obj, vel_ego, acc_ego, rot_ego, ang_vel)

        # -- control
        WP.update(ego_state)
        if WP.needs_waypoint():
            w = Waypoint(Pose(pos, rot), speed_target)
            d = pos.distance(pos_ego)
            WP.push(d, w)
        ctrl = controller(ego_state, WP)
        assert np.sign(ctrl.throttle - ctrl.brake) == np.sign(
            speed_target - vel_ego.norm()
        )
        assert (ctrl.throttle + ctrl.brake) > 0
