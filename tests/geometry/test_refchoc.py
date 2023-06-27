import numpy as np
import quaternion
from avstack.geometry.refchoc import ReferenceFrame, GlobalOrigin3D, Vector, Rotation, get_reference_from_line
from avstack.geometry import transform_orientation, q_mult_vec, q_stan_to_cam


def x_rand():
    return np.random.randn(3)


def q_rand():
    return transform_orientation(np.random.rand(3), 'euler', 'quat')


# --------------------------------
# COORDINATE FRAME
# --------------------------------

def test_ref_frame_from_string():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    cf2_string = cf2.format_as_string()
    cf2_recon = get_reference_from_line(cf2_string)
    assert cf2.allclose(cf2_recon)


def test_one_level_coordinate_frame():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    assert not cf1.is_global_origin
    assert cf1.level == 1


def test_two_level_coordinate_frame():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    assert cf1.level == 1
    assert cf2.level == 2


def test_bad_coordinate_frame():
    try:
        _ = ReferenceFrame(x_rand(), q_rand(), None)
    except ValueError:
        pass
    else:
        raise RuntimeError("Should have excepted...")


def test_common_ancestor():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    cf3 = ReferenceFrame(x_rand(), q_rand(), cf1)
    assert cf2.common_ancestor(cf1) == cf1
    assert cf2.common_ancestor(cf3) == cf1


def test_no_common_ancestor():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    cf3 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf4 = ReferenceFrame(x_rand(), q_rand(), cf3)
    assert cf2.common_ancestor(cf4) == GlobalOrigin3D


def test_integrate_1():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf_int = cf1.integrate(start_at=GlobalOrigin3D)
    assert np.allclose(cf1.x, cf_int.x)
    assert quaternion.allclose(cf1.q, cf_int.q)


def test_integrate_2():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    # -- back one level
    cf_int = cf2.integrate(start_at=cf1)
    assert np.allclose(cf2.x, cf_int.x)
    assert quaternion.allclose(cf2.q, cf_int.q)
    # -- back two levels
    cf_int = cf2.integrate(start_at=GlobalOrigin3D)
    x_manual = cf1.x + q_mult_vec(cf1.q.conjugate(), cf2.x)
    q_manual = cf2.q * cf1.q
    assert np.allclose(x_manual, cf_int.x)
    assert np.allclose(q_manual.vec, cf_int.q.vec)


def test_differential_1():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    cf_d = cf1.differential(cf2, in_self=True)
    assert np.allclose(cf2.x, cf_d.x)
    assert np.allclose(cf2.q.vec, cf_d.q.vec)


def test_differential_2():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(x_rand(), q_rand(), cf1)
    cf3 = ReferenceFrame(x_rand(), q_rand(), cf2)
    cf_d = cf1.differential(cf3)
    cf_i = cf3.integrate(start_at=cf1)
    assert np.allclose(cf_d.x, cf_i.x)
    assert quaternion.allclose(cf_d.q, cf_i.q)


def test_differential_moving_frame():
    cf1 = ReferenceFrame(np.array([2,0,0]), np.quaternion(1), GlobalOrigin3D, v=np.array([1,0,0]))
    cf2 = ReferenceFrame(np.array([0,2,0]), cf1.q, GlobalOrigin3D, v=np.array([0,1,0]))
    assert not cf1.allclose(cf2)
    cf_int = cf1.differential(cf2)
    assert np.allclose(cf_int.x, np.array([-2,2,0]))
    assert np.allclose(cf_int.v, np.array([-1,1,0]))
    cf_int_int = cf_int.integrate(start_at=GlobalOrigin3D)
    assert cf_int_int.allclose(cf2)
    assert not cf_int_int.allclose(cf1)


# --------------------------------
# VECTOR
# --------------------------------

def test_change_vector_frame():
    v1 = Vector(x_rand(), GlobalOrigin3D)
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    v1_cf1 = v1.change_reference(cf1, inplace=False)
    v1_cf1_m = q_mult_vec(cf1.q, v1.x - cf1.x)
    assert np.allclose(v1_cf1.x, v1_cf1_m)


def test_add_vectors_same():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    v1 = Vector(x_rand(), cf1)
    v2 = Vector(x_rand(), cf1)
    v_add = v1 + v2
    assert np.allclose(v_add.x, v1.x + v2.x)


def test_add_vectors_same_long():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    cf2 = ReferenceFrame(np.zeros((3,)), np.quaternion(1), cf1)
    v1 = Vector(x_rand(), cf1)
    v2 = Vector(x_rand(), cf2)
    v_add = v1 + v2
    assert np.allclose(v_add.x, v1.x + v2.x)


def test_add_vectors_diff():
    cf2 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    v1 = Vector(x_rand(), GlobalOrigin3D)
    v2 = Vector(x_rand(), cf2)
    v_add = v2 + v1  # always in the frame of the first
    v_manual = v2.x + (q_mult_vec(cf2.q, v1.x - cf2.x))
    assert np.allclose(v_add.x, v_manual)


def test_vector_distance():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    v1 = Vector(x_rand(), cf1)
    v2 = Vector(x_rand(), cf1)
    assert np.isclose(v1.distance(v2), np.linalg.norm(v1.x - v2.x))


def test_vector_known_frame():
    cf1 = ReferenceFrame(np.zeros((3,)), np.quaternion(1), GlobalOrigin3D)
    cf2 = ReferenceFrame(np.zeros((3,)), q_stan_to_cam, GlobalOrigin3D)
    v1 = Vector(np.array([100., 0, 0]), cf1)
    v2 = v1.change_reference(cf2, inplace=False)
    assert np.allclose(v2.x, q_mult_vec(q_stan_to_cam, v1.x))


# --------------------------------
# ROTATION
# --------------------------------

def change_rotation_frame():
    q1 = Rotation(q_rand(), GlobalOrigin3D)
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    q1_cf1 = q1.change_reference(cf1, inplace=False)
    assert np.allclose((cf1.q * q1.q).vec, q1_cf1.vec)


def test_compose_rotations_same():
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    q1 = Rotation(q_rand(), cf1)
    q2 = Rotation(q_rand(), cf1)
    q_comp = q2 * q1
    assert np.allclose(q_comp.q.vec, (q2.q * q1.q).vec)
