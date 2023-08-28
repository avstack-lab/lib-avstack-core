# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-02-17
# @Filename: maskfilters.py
# @Last modified by:   spencer
# @Last modified time: 2021-08-04
"""
A collection of slicers, masks, and filtering operations for perception data

The code is independent of data source so long as format is standard
"""

from copy import deepcopy

import numpy as np
from numba import jit

import avstack.geometry.bbox as bbox
from avstack.geometry import transformations as tforms


# ==============================================================================
# FILTERS
# ==============================================================================


def _get_extents_filter(loc_data, extents):
    """
    Returns the extent filter for data of [N x 3] or [N x 4]
    """
    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (
        (loc_data[:, 0] > x_extents[0])
        & (loc_data[:, 0] < x_extents[1])
        & (loc_data[:, 1] > y_extents[0])
        & (loc_data[:, 1] < y_extents[1])
        & (loc_data[:, 2] > z_extents[0])
        & (loc_data[:, 2] < z_extents[1])
    )

    return extents_filter


def filter_points(point_cloud, extents, ground_plane=None, offset_dist=2.0):
    """
    Creates a point filter using the 3D extents and ground plane
    :param point_cloud: Point cloud in the standard for mof [N, 3] or [N,4]
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d) - in the lidar frame
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """

    pc2 = np.asarray(point_cloud[:, 0:3])

    # Filter points within certain xyz range
    extents_filter = _get_extents_filter(pc2, extents)

    # Add ground plane filter
    if ground_plane is not None:
        ground_plane = np.array(ground_plane)

        # Calculate filter using ground plane
        ones_col = np.ones(pc2.shape[0])
        padded_points = np.hstack([pc2, ones_col[:, None]])

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        # Create plane filter
        dot_prod = np.dot(offset_plane, np.transpose(padded_points))
        plane_filter = dot_prod < 0

        # Combine the two filters
        point_filter = np.logical_and(extents_filter, plane_filter)
    else:
        # Only use the extents for filtering
        point_filter = extents_filter

    return point_filter


def filter_boxes_range(boxes, range_max):
    """
    Return mask for which boxes are inside the range
    """
    box_centers = np.asarray([b.t for b in boxes])
    return np.linalg.norm(box_centers, axis=1) < range_max


def filter_points_range(point_cloud, range_min, range_max):
    """
    Return mask for which points are inside the range
    """
    try:
        rng = np.linalg.norm(point_cloud.data, axis=1)
    except AttributeError as e:
        rng = np.linalg.norm(point_cloud, axis=1)
    return (range_min <= rng) & (rng <= range_max)


def filter_boxes_extent(boxes, extent):
    """
    Filter if the box center is outside the extents

    extent is usually defined in the lidar frame of reference
    """
    box_centers = np.asarray([b.t.vector for b in boxes])
    box_filter = _get_extents_filter(box_centers, extent)
    return box_filter


def filter_points_in_image(point_cloud, im_size, calib):
    """Filter points based on if they fall within the image specified"""
    # Filter points based on the image coordinates
    point_in_im = calib.project_velo_to_image(point_cloud[:, 0:3])
    point_in_im_mask = (
        (point_in_im[:, 0] > 0)
        & (point_in_im[:, 0] < im_size[1])
        & (point_in_im[:, 1] > 0)
        & (point_in_im[:, 1] < im_size[0])
        & (point_cloud[:, 0] > 0)
    )
    return point_in_im_mask


def filter_points_in_cone(point_cloud, v_unit, half_angle):
    """Filter points defined by a cone"""
    assert np.isclose(np.linalg.norm(v_unit), 1)
    rng_pts = np.linalg.norm(point_cloud.data, axis=1)
    cos_th = np.dot(point_cloud.data[:, :3], v_unit) / rng_pts
    cos_ha = np.cos(half_angle)
    return cos_th >= cos_ha  # reverse since instead of arccos


def filter_points_in_image_frustum(point_cloud, box2d, camera_calib):
    """
    Gets all lidar points in a frustum defined by the 2d box in image space

    point_cloud - velo coordinates
    """
    lidar_in_img_ref = point_cloud.data.change_calibration(camera_calib, inplace=False)
    lidar_mask0 = lidar_in_img_ref[:, 2] >= 0  # assumes z is forward
    lidar_image = point_cloud.project(camera_calib)
    lidar_mask1 = lidar_image.data[:, 0] > box2d.box2d[0]
    lidar_mask2 = lidar_image.data[:, 1] > box2d.box2d[1]
    lidar_mask3 = lidar_image.data[:, 0] < box2d.box2d[2]
    lidar_mask4 = lidar_image.data[:, 1] < box2d.box2d[3]
    frustum_filter = lidar_mask0 & lidar_mask1 & lidar_mask2 & lidar_mask3 & lidar_mask4
    return frustum_filter


def filter_points_in_pillar(point_cloud, box_bev):
    """
    Take a 3d box and expand the vertical axis to +/- infinity to create a pillar

    box_bev is 4x2 and is in m
    box_bev is defined as:

    x - forward
    y - left
    z - up
    """
    # Project point cloud to bev -- z is up so is reduced
    pc_velo_bev = np.delete(point_cloud.data[:, 0:3], 2, axis=1)

    # Get points in hull
    return bbox.in_hull(pc_velo_bev, box_bev)


# def filter_points_in_shadow(point_cloud, box2d, box3d, box_calib, camera_calib):
#     """
#     Gets all lidar points in the shadow which is the frustum behind the object
#     """

#     # Filter within frustum
#     frustum_filter = filter_points_in_image_frustum(point_cloud, box2d, camera_calib)

#     # Filter by bounding box
#     box_bev = box3d.get_corners_bev_velo(calib=point_cloud.calibration)
#     pillar_filter = filter_points_in_pillar(point_cloud, box_bev)

#     # Filter by range
#     box_center_velo = box_calib.transform_3d_to_3d(box3d.t, origin=point_cloud.calibration.reference)
#     range_filter = filter_points_range(point_cloud, 0, np.linalg.norm(box_center_velo))

#     return frustum_filter & (~pillar_filter) & (~range_filter)


def filter_points_slice(
    point_cloud, area_extents, ground_plane, height_min, height_max
):
    """Creates a slice filter to take a slice of the point cloud between
        ground_offset_dist and offset_dist above the ground plane

    Args:
        point_cloud: Point cloud in the shape (N,3) or (N,4)
        area_extents: 3D area extents
        ground_plane: ground plane coefficients
        offset_dist: max distance above the ground
        ground_offset_dist: min distance above the ground plane

    Returns:
        A boolean mask if shape (N,) where
            True indicates the point should be kept
            False indicates the point should be removed
    """

    # Filter points within certain xyz range and offset from ground plane
    top_to_road = filter_points(point_cloud, area_extents, ground_plane, height_max)

    # Filter points within 0.2m of the road plane
    low_to_road = filter_points(point_cloud, area_extents, ground_plane, height_min)

    slice_filter = np.logical_xor(top_to_road, low_to_road)
    return slice_filter


def filter_points_in_box(
    points, box_corners, include_boundary=True, coarse_filters=True, max_range=150
):
    """
    Returns the points that are inside the box
    points = (N,3)
    box_corners = (8,3)

    returns set of indices of point sets that are inside or on boundary of box

    reference to: https://stackoverflow.com/questions/21037241/
    how-to-determine-a-point-is-inside-or-outside-a-cube#:~:
    text=If%20the%20projection%20lies%20inside,the%20point%20P%20is%20V.
    """
    pc = points[:, 0:3]

    # check if we're way off first
    if coarse_filters:
        box_dist = np.linalg.norm(np.mean(box_corners, axis=0))
        if box_dist > max_range:
            return np.zeros((pc.shape[0],), dtype=np.bool)
    # run test
    box_filter = _check_pts_boundary(pc, box_corners.x, include_boundary)
    return box_filter


@jit(nopython=True)
def _check_pts_boundary(pc, box_corners, include_boundary):
    # unpack box corners
    t1, t2, t3, t4, b1, b2, b3, b4 = box_corners
    # get vectors
    dir1 = t1 - b1
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1
    dir2 = b2 - b1
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2
    dir3 = b4 - b1
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    box_center = (b1 + t3) / 2.0
    dir_vec = pc - box_center

    # Run checks
    if include_boundary:
        res1 = (np.absolute(np.dot(dir_vec, dir1)) * 2) <= size1
        if not np.any(res1):
            return res1
        res2 = (np.absolute(np.dot(dir_vec, dir2)) * 2) <= size2
        if not np.any(res2):
            return res2
        res3 = (np.absolute(np.dot(dir_vec, dir3)) * 2) <= size3
        if not np.any(res3):
            return res3
    else:
        res1 = (np.absolute(np.dot(dir_vec, dir1)) * 2) < size1
        if not np.any(res1):
            return res1
        res2 = (np.absolute(np.dot(dir_vec, dir2)) * 2) < size2
        if not np.any(res2):
            return res2
        res3 = (np.absolute(np.dot(dir_vec, dir3)) * 2) < size3
        if not np.any(res3):
            return res3
    box_filter = res1 & res2 & res3
    return box_filter


def filter_points_in_object_bbox(point_cloud, box3d):
    """
    Gets lidar points that fall within a bounding box object
    """
    # Get points in each bounding box
    box3d_pts_3d = box3d.corners
    box3d_pts_3d.change_reference(point_cloud.reference, inplace=True)
    box_filter = filter_points_in_box(point_cloud.data, box3d_pts_3d)
    return box_filter


def filter_points_in_image(points, P):
    """
    Filter which points are in view for an image
    """
    im_size = [2 * P[1, 2], 2 * P[0, 2]]  # size is [h, w]
    points_in_img = tforms.project_to_image(points, P)
    points_in_im_mask = (
        (points_in_img[:, 0] > 0)
        & (points_in_img[:, 0] < im_size[1])
        & (points_in_img[:, 1] > 0)
        & (points_in_img[:, 1] < im_size[0])
        & (points[:, 2] > 0)
    )
    return points_in_im_mask


def box_in_fov(box_3d, camera_calib, d_thresh=None, check_reference=True):
    """Check if a 3d box is in the FOV of the camera

    assumption: camera points along z axis
    """
    if check_reference:
        if box_3d.reference != camera_calib.reference:
            box_3d = box_3d.change_reference(camera_calib.reference, inplace=False)
    if d_thresh is not None:
        if box_3d.t.norm() > d_thresh:
            return False

    # calculate min dot product based on half-angle
    delta = 1.5 * np.pi / 180  # add some small delta for errors...
    fov_half = delta + np.arctan(
        2 * (camera_calib.P[0, 0]) / camera_calib.img_shape[1]
    )  # in radians

    # -- front edge, center, back edge
    center = box_3d.t
    fv = box_3d.q.forward_vector
    lv = box_3d.q.left_vector
    front_edge = center + box_3d.l / 2 * fv
    left_edge = center + box_3d.w / 2 * lv
    back_edge = center - box_3d.l / 2 * fv
    right_edge = center - box_3d.w / 2 * lv

    c1 = (
        np.dot(front_edge.x, np.array([0, 0, 1])) > np.cos(fov_half) * front_edge.norm()
    )
    c2 = np.dot(center.x, np.array([0, 0, 1])) > np.cos(fov_half) * center.norm()
    c3 = np.dot(back_edge.x, np.array([0, 0, 1])) > np.cos(fov_half) * front_edge.norm()
    c4 = np.dot(left_edge.x, np.array([0, 0, 1])) > np.cos(fov_half) * left_edge.norm()
    c5 = (
        np.dot(right_edge.x, np.array([0, 0, 1])) > np.cos(fov_half) * right_edge.norm()
    )

    if any([c1, c2, c3, c4, c5]):
        box_3d = box_3d.project_to_2d_bbox(calib=camera_calib)
        box2d_image = [0, 0, camera_calib.img_shape[1], camera_calib.img_shape[0]]
        if bbox.box_intersection(box_3d.box2d, box2d_image) > 0:
            return True
    return False


def filter_objects_in_frustum(objs_1, objs_2, camera_calib):
    # Loop if lists
    if (type(objs_1) is not list) and (type(objs_1) is not np.ndarray):
        objs_1 = [objs_1]
    if (type(objs_2) is not list) and (type(objs_2) is not np.ndarray):
        objs_2 = [objs_2]

    in_frustum = np.zeros((len(objs_1), len(objs_2)), dtype=bool)

    for i in range(len(objs_1)):
        for j in range(len(objs_2)):
            if _item_in_frustum(objs_1[i], objs_2[j], camera_calib) or _item_in_frustum(
                objs_2[j], objs_1[i], camera_calib
            ):
                in_frustum[i, j] = True

    return in_frustum


def _item_in_frustum(box3d_1, box3d_2, camera_calib):
    box2d_1 = box3d_1.project_to_2d_bbox(calib=camera_calib)
    box2d_2 = box3d_2.project_to_2d_bbox(calib=camera_calib)

    if bbox.box_intersection(box2d_1.box2d, box2d_2.box2d) > 0:
        return True
    else:
        return False
