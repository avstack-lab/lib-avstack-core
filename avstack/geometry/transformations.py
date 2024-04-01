import math

import numpy as np
import quaternion
from numba import jit

from avstack.geometry.coordinates import StandardCoordinates


# WGS84 Constants:
WGS_a = 6378137
WGS_a_2 = WGS_a**2

WGS_e = 8.1819190842622 * (10**-2)
WGS_e_2 = WGS_e**2

WGS_b = math.sqrt(WGS_a_2 * (1 - WGS_e_2))
WGS_b_2 = WGS_b**2

WGS_ep = math.sqrt((WGS_a_2 - WGS_b_2) / WGS_b_2)
WGS_ep_2 = WGS_ep**2


# =============================================
# SPHERICAL
# =============================================


@jit(nopython=True, fastmath=True)
def matrix_cartesian_to_spherical(M):
    """
    Matrix is of shape N x 3

    Outputs values in ranges:
    azimuth: [-pi, +pi]
    elevation: [-pi/2, +pi/2]
    """
    M_ = M.copy()
    for i in range(M.shape[0]):  # numba is ok with loop
        M_[i, 0] = np.linalg.norm(M[i, :3])
    M_[:, 1] = np.arctan2(M[:, 1], M[:, 0])
    M_[:, 2] = np.arcsin(M[:, 2] / M_[:, 0])
    return M_


@jit(nopython=True, fastmath=True)
def matrix_spherical_to_cartesian(M):
    """
    Matrix is of shape N x 3

    Assumes elevations are in ranges:
    azimuth:   [-pi, +pi]
    elevation: [-pi/2, +pi/2]
    """
    M_ = M.copy()
    c2 = np.cos(M[:, 2])
    M_[:, 0] = M[:, 0] * np.cos(M[:, 1]) * c2
    M_[:, 1] = M[:, 0] * np.sin(M[:, 1]) * c2
    M_[:, 2] = M[:, 0] * np.sin(M[:, 2])
    return M_


def spherical_to_cartesian(razel, coordinates=StandardCoordinates):
    x = razel[0] * np.cos(razel[1]) * np.cos(razel[2])
    y = razel[0] * np.sin(razel[1]) * np.cos(razel[2])
    z = razel[0] * np.sin(razel[2])
    return StandardCoordinates.convert(np.array([x, y, z]), coordinates)


def cartesian_to_spherical(v, coordinates=StandardCoordinates):
    v2 = coordinates.convert(v, StandardCoordinates)
    rng = np.linalg.norm(v2)
    az = np.arctan2(v2[1], v2[0])
    el = np.arcsin(v2[2] / rng)
    return np.array([rng, az, el])


def xyzvel_to_razelrrt(xyzvel):
    rng, az, el = cartesian_to_spherical(xyzvel[:3])
    rrt = xyzvel[3:6] @ xyzvel[:3] / rng
    return np.array([rng, az, el, rrt])


def razelrrt_to_xyzvel(razelrrt):
    x, y, z = spherical_to_cartesian(razelrrt[:3])
    v_unit = np.array([x, y, z]) / razelrrt[0]
    vx, vy, vz = v_unit * razelrrt[3]
    return np.array([x, y, z, vx, vy, vz])


# ===========================================
# Global wrapper function
# ===========================================


def transform_point(x, c_from, c_to, origin=None, t_unix=None):
    """
    :c_from - coordinate frame of x
    :c_to - coordinate from to transform to
    """
    FROM = c_from.lower()
    TO = c_to.lower()

    if isinstance(x, list):
        x = np.asarray(x)
    if len(x.shape) == 1:
        x = x[:, None]

    # Handle required arguments
    if (FROM in ["ned"]) or (TO in ["ned"]):
        assert origin is not None

        if len(origin[0].shape) == 1:
            org_updated = (origin[0][:, None], origin[1])
        else:
            org_updated = origin

        if org_updated[1] == "lla":
            check_angle_radians(org_updated[0])

    if (FROM == "eci") or (TO == "eci"):
        assert t_unix is not None

    # Process result
    if FROM == "ecef":
        if TO == "eci":
            x_out = ecef_to_eci(x, t_unix)
        elif TO == "lla":
            x_out = ecef_to_lla(x)
        elif TO == "ned":
            x_out = ecef_to_ned(x, org_updated)
        elif TO == "ecef":
            x_out = x
        else:
            raise NotImplementedError
    elif FROM == "eci":
        if TO == "ecef":
            x_out = eci_to_ecef(x, t_unix)
        elif TO == "lla":
            x_out = eci_to_lla(x, t_unix)
        elif TO == "ned":
            x_out = eci_to_ned(x, org_updated, t_unix)
        elif TO == "eci":
            x_out = x
        else:
            raise NotImplementedError
    elif FROM == "lla":
        if TO == "ecef":
            x_out = lla_to_ecef(x)
        elif TO == "eci":
            x_out = lla_to_eci(x, t_unix)
        elif TO == "ned":
            x_out = lla_to_ned(x, org_updated)
        elif TO == "lla":
            x_out = x
        else:
            raise NotImplementedError
    elif FROM == "ned":
        if TO == "ecef":
            x_out = ned_to_ecef(x, org_updated)
        elif TO == "eci":
            x_out = ned_to_eci(x, org_updated, t_unix)
        elif TO == "lla":
            x_out = ned_to_lla(x, org_updated)
        elif TO == "ned":
            x_out = x
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return x_out


# ===========================================
# 3D Transform vectors of coordinates
# ===========================================


def ecef_to_eci(x, t_unix):
    raise NotImplementedError


def ecef_to_ned(x, origin):
    """
    :x - 3D vector (or multiple vectors) (3 X N) of point in ECEF (cartesian)
    :origin - tuple of (vector, coordinate_frame)

    if coordinate frame is lla, should be in radians
    """
    x_origin_ecef = transform_point(origin[0], origin[1], "ecef")
    x_centered = x - x_origin_ecef
    R_ecef_to_ned = get_R_ecef_to_ned(origin)
    return R_ecef_to_ned @ x_centered


def ecef_to_lla(x):
    """
    :x - 3D vector (or multiple vectors) (3 X N) of point in ECEF (cartesian)
    """

    p = np.sqrt(x[0, :] ** 2 + x[1, :] ** 2)
    th = np.arctan2(WGS_a * x[2, :], WGS_b * p)

    lat = np.arctan2(
        x[2, :] + WGS_ep_2 * WGS_b * np.sin(th) ** 3,
        p - WGS_e_2 * WGS_a * np.cos(th) ** 3,
    )
    lon = np.arctan2(x[1, :], x[0, :])
    N = WGS_a / np.sqrt(1 - WGS_e_2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    which_unstable = np.where((np.abs(x[0, :]) < 1) & (np.abs(x[1, :]) < 1))
    alt[which_unstable] = np.abs(x[2, which_unstable] - WGS_b)
    return np.concatenate((lat[:, None].T, lon[:, None].T, alt[:, None].T), axis=0)


def eci_to_ecef(x, t_unix):
    raise NotImplementedError


def eci_to_lla(x, t_unix):
    raise NotImplementedError


def eci_to_ned(x, origin, t_unix):
    raise NotImplementedError


def lla_to_ecef(x):
    N = WGS_a / np.sqrt(1 - WGS_e_2 * np.sin(x[0, :]) ** 2)
    return np.concatenate(
        (
            ((N + x[2, :]) * np.cos(x[1, :]) * np.cos(x[0, :]))[:, None].T,
            ((N + x[2, :]) * np.sin(x[1, :]) * np.cos(x[0, :]))[:, None].T,
            (((1 - WGS_e_2) * N + x[2, :]) * np.sin(x[0, :]))[:, None].T,
        ),
        axis=0,
    )


def lla_to_eci(x, t_unix):
    raise NotImplementedError


def lla_to_ned(x, origin):
    x_ecef = lla_to_ecef(x)
    return ecef_to_ned(x_ecef, origin)


def ned_to_ecef(x, origin):
    x_origin_ecef = transform_point(origin[0], origin[1], "ecef")
    R_ned_to_ecef = get_R_ned_to_ecef(origin)
    x_ecef_uncenter = R_ned_to_ecef @ x
    return x_ecef_uncenter + x_origin_ecef


def ned_to_eci(x, origin, t_unix):
    raise NotImplementedError


def ned_to_lla(x, origin):
    x_ecef = ned_to_ecef(x, origin)
    return ecef_to_lla(x_ecef)


# ===========================================
# 2D Transform vectors of coordinates
# ===========================================


# ===========================================
# Obtain rotation/transformation matrices
# ===========================================


def get_q_ecef_to_eci():
    raise NotImplementedError


def get_R_ecef_to_eci():
    raise NotImplementedError


def get_q_ned_to_ecef(origin):
    return transform_orientation(get_R_ned_to_ecef(origin), "dcm", "quat")


def get_R_ned_to_ecef(origin):
    return get_R_ecef_to_ned(origin).T


def get_q_ecef_to_ned(origin):
    return get_q_ned_to_ecef(origin).conjugate()


def get_R_ecef_to_ned(origin):
    """
    :origin - tuple of (x, coordinate_frame)
    """
    x_o_lla = transform_point(origin[0], origin[1], "lla")
    R_ecef_to_ned = np.array(
        [
            [
                -np.sin(x_o_lla[0, 0]) * np.cos(x_o_lla[1, 0]),
                -np.sin(x_o_lla[0, 0]) * np.sin(x_o_lla[1, 0]),
                np.cos(x_o_lla[0, 0]),
            ],
            [-np.sin(x_o_lla[1, 0]), np.cos(x_o_lla[1, 0]), 0],
            [
                -np.cos(x_o_lla[0, 0]) * np.cos(x_o_lla[1, 0]),
                -np.cos(x_o_lla[0, 0]) * np.sin(x_o_lla[1, 0]),
                -np.sin(x_o_lla[0, 0]),
            ],
        ]
    )
    return R_ecef_to_ned


def get_R_ecef_to_lla():
    raise NotImplementedError


@jit(nopython=True, fastmath=True)
def _make_euler(R):
    R = np.clip(R, -1.0, 1.0)
    roll = np.arctan2(R[1, 2], R[2, 2])
    pitch = -np.arcsin(R[0, 2])
    yaw = np.arctan2(R[0, 1], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def R_to_euler(R):
    """
    Convert direction cosine matrix to euler angles

    :R - direction cosine matrix - generally will be R_local_to_body (e.g., R_ned_to_body)
    assumes R = Rx(phi) * Ry(theta) * Rz(psi)
    where - phi on [-pi, pi] (roll)
          - theta on [-pi/2, pi/2] (pitch)
          - psi on [-pi, pi] (yaw/heading)
    """
    if len(R.shape) == 2:
        euler = _make_euler(R)
    else:
        euler = np.zeros((3, R.shape[2]))
        for i in range(R.shape[2]):
            euler[:, i] = _make_euler(R[:, :, i])
    return euler


@jit(nopython=True, fastmath=True)
def _make_DCM(euler):
    R = float(euler[0])
    P = float(euler[1])
    Y = float(euler[2])
    # This is DCM of global level --> body
    DCM = np.array(
        [
            [
                np.cos(P) * np.cos(Y),
                np.sin(R) * np.sin(P) * np.cos(Y) - np.cos(R) * np.sin(Y),
                np.cos(R) * np.sin(P) * np.cos(Y) + np.sin(R) * np.sin(Y),
            ],
            [
                np.cos(P) * np.sin(Y),
                np.sin(R) * np.sin(P) * np.sin(Y) + np.cos(R) * np.cos(Y),
                np.cos(R) * np.sin(P) * np.sin(Y) - np.sin(R) * np.cos(Y),
            ],
            [-np.sin(P), np.sin(R) * np.cos(P), np.cos(R) * np.cos(P)],
        ]
    )
    return DCM.T


def euler_to_R(euler: np.ndarray):
    """
    Convert angles to direction cosine matrix

    :euler - tuple of (roll, pitch, yaw) / (phi, theta, psi) angles - generally will be
             angles from the local frame to the body

    output will generally be R_global_to_body (e.g., R_ned_to_body)
    assumes R = Rx(phi) * Ry(theta) * Rz(psi)
    where - phi on [-pi, pi] (roll)
          - theta on [-pi/2, pi/2] (pitch)
          - psi on [-pi, pi] (yaw/heading)

    could also get this by composing individual axis rotations in sequence, but
    explicitly writing it out is more efficient
    """
    if len(euler.shape) == 1:
        DCM = _make_DCM(euler)
    else:
        DCM = np.zeros((3, 3, euler.shape[1]), dtype=float)
        for i in range(euler.shape[1]):
            DCM[:, :, i] = _make_DCM(euler[:, i])

    return DCM


# ===========================================
# Obtain rotation/transformation matrices
# ===========================================


def transform_orientation(x, a_from, a_to):
    """
    Transform angles from different representations
    eulers must be represented in (roll, pitch, yaw) order

    :x - vector or matrix describing the orientation
    :a_from - {'euler', 'quat', 'DCM'}
    :a_to - {'euler', 'quat', 'DCM'}
    """
    FROM = a_from.lower()
    TO = a_to.lower()

    q_list = ["quat", "quaternion"]
    e_list = ["euler"]
    d_list = ["dcm"]

    if isinstance(x, list) and len(x) == 3:
        x = np.asarray([float(xi) for xi in x])

    if FROM in e_list:
        if TO in q_list:
            DCM = euler_to_R(x)
            if len(DCM.shape) == 3:
                DCM = np.transpose(DCM, (2, 0, 1))
            x_out = quaternion.from_rotation_matrix(DCM)
        elif TO in d_list:
            x_out = euler_to_R(x)
        elif TO in e_list:
            x_out = x
        else:
            raise NotImplementedError
    elif FROM in q_list:
        if TO in d_list:
            x_out = quaternion.as_rotation_matrix(x)
            if len(x_out.shape) == 3:
                x_out = np.transpose(x_out, (1, 2, 0))
        elif TO in e_list:
            DCM = quaternion.as_rotation_matrix(x)
            if len(DCM.shape) == 3:
                DCM = np.transpose(DCM, (1, 2, 0))
            x_out = R_to_euler(DCM)
        elif TO in q_list:
            x_out = x
        else:
            raise NotImplementedError
    elif FROM in d_list:
        if TO in e_list:
            x_out = R_to_euler(x)
        elif TO in q_list:
            if len(x.shape) == 3:
                x = np.transpose(x, (2, 0, 1))
            x_out = quaternion.from_rotation_matrix(x)
        elif TO in d_list:
            x_out = x
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if TO in q_list:
        if x_out.w < 0:
            x_out *= -1

    # error checking
    if ((TO in q_list) and (np.isnan(x_out.w) or np.isnan(x_out.vec).any())) or (
        (TO not in q_list) and np.isnan(x_out).any()
    ):
        import pdb

        pdb.set_trace()
        raise RuntimeError(x_out)

    return x_out


# ===========================================
# Utilities
# ===========================================


def check_angle_radians(vector):
    assert np.all(vector < (2 * math.pi))


def sign_corrected_q(q0, q1):
    """Adjust the sign of q1 to be as close as possible to q0"""
    if abs(q0 - q1) > 1.4142135623730951:
        return -q1
    return q1


def align_x_to_vec(vec):
    """
    Generates rotation matrix to align x axis to the specified vector

    Effectively, align along range to target in a range-az-el system
    """
    r2d = np.linalg.norm(vec[0:2])
    if r2d == 0:
        az = 0
    else:
        az = math.acos(vec[0] / r2d)
    el = math.asin(vec[2] / np.linalg.norm(vec))

    # rotate about z for az, then about -y for el
    R1 = rotz(az)
    R2 = roty(-el)
    return R2 @ R1


def cart2hom(pts_3d):
    """Input: nx3 points in Cartesian
    Oupput: nx4 points in Homogeneous by appending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def get_yaw_from_bev_corners(bev_corners):
    """
    Get yaw angle from bird's eye view 4 corners

    bev_corners: 4x2
    coordinates defined as: x - forward, y - left, z - up

    0 yaw is defined as the short side perpendicular to the y axis in velo
    yaw can't be more than 90 degrees here
    """

    # Center coordinates
    centered = bev_corners - np.mean(bev_corners, axis=0)[:, None].T
    side_1 = np.linalg.norm(centered[0, :] - centered[1, :])
    side_2 = np.linalg.norm(centered[0, :] - centered[2, :])

    # The shorter side is the front/back sides
    if side_1 <= side_2:
        idx_front_to_back = [(0, 1), (2, 3)]
    else:
        idx_front_to_back = [(0, 2), (1, 3)]

    # Get the line to the center of the short side
    vec_mid = np.mean(centered[idx_front_to_back[0], :], axis=0)

    # Yaw calculation
    return np.arctan2(vec_mid[0], vec_mid[1])


def rotx(t):
    """3D Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotz_2d(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s], [s, c]])


def get_rot_yaw_matrix(rot, up):
    if up == "+y":
        M = roty(-rot)
    elif up == "-y":
        M = roty(rot)
    elif up == "+z":
        M = rotz(-rot)
    elif up == "-z":
        M = rotz(rot)
    else:
        raise NotImplementedError
    return M


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def project_to_image(pts_3d, P):
    """Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def project_cartesian_to_5_channel_spherical(point_cloud, nchan_h=64, nchan_w=512):
    """
    Takes a point cloud and converts to spherical representation with features

    Features are used from the LU-Net paper and are (x, y, z, intensity, r)
    Standard shape: (64, 512, 5)
    https://arxiv.org/pdf/1710.07368.pdf

    Coordinate system: LiDAR is x forward, y left, z up
    """
    # Project to spherical
    r_3d = np.linalg.norm(point_cloud[:, :3], axis=1)
    r_2d = np.linalg.norm(point_cloud[:, :2], axis=1)
    zen = np.arcsin(point_cloud[:, 2] / r_3d)
    az = np.arcsin(point_cloud[:, 1] / r_2d)

    # Only include within front-view (define as a 90 degree azimuth pyramid??)
    half_ang_az = math.pi / 180 * 45

    # Define the bin width
    del_w = 2 * half_ang_az / nchan_w
    del_h = (max(zen) - min(zen)) / nchan_h  # just do empirically

    # Assign bins
    bin_w = np.floor((az + math.pi / 4) / del_w)
    bin_h = np.floor((zen - min(zen)) / del_h)

    # Build the matrix by looping
    spher_img = np.zeros((nchan_h, nchan_w, 5))
    for ipt in range(point_cloud.shape[0]):
        # To filter by viewing angle, ignore bins outside of desired range
        if (
            (bin_w[ipt] < 0)
            or (bin_w[ipt] >= nchan_w)
            or (bin_h[ipt] < 0)
            or (bin_h[ipt] >= nchan_h)
        ):
            continue

        # Flip x and y by convention (?)
        row = nchan_h - int(bin_h[ipt]) - 1
        col = nchan_w - int(bin_w[ipt]) - 1

        # Assign features
        spher_img[row, col, 0:3] = point_cloud[ipt, 0:3]
        spher_img[row, col, 3] = point_cloud[ipt, 3]
        spher_img[row, col, 4] = r_3d[ipt]

    return spher_img


def project_to_bev():
    raise NotImplementedError
