"""Property tests for dynamics/rotations.py"""
import numpy as np
from dynamics.rotations import (
    hat, vee, Exp, Log,
    euler_zyx_to_dcm, dcm_to_euler_zyx,
    quat_to_dcm, dcm_to_quat,
)

RNG = np.random.default_rng(0)


def random_rotvec(max_angle=np.pi - 0.01):
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * RNG.uniform(0, max_angle)


# --- EXIT 1: Log(Exp(v)) == v ---
def test_log_exp_roundtrip():
    for _ in range(1000):
        v = random_rotvec()
        assert np.allclose(Log(Exp(v)), v, atol=1e-9)


# --- EXIT 2: R^T R == I to 1e-12 ---
def test_orthonormal():
    for _ in range(1000):
        R = Exp(random_rotvec())
        assert np.max(np.abs(R.T @ R - np.eye(3))) < 1e-12
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


# --- EXIT 3: quat <-> DCM round-trip ---
def test_quat_dcm_roundtrip():
    for _ in range(1000):
        R = Exp(random_rotvec())
        assert np.allclose(quat_to_dcm(dcm_to_quat(R)), R, atol=1e-12)


# --- hat/vee inverse + cross-product property ---
def test_hat_vee():
    for _ in range(1000):
        v = RNG.normal(size=3)
        w = RNG.normal(size=3)
        assert np.allclose(hat(v) @ w, np.cross(v, w))
        assert np.allclose(vee(hat(v)), v)


# --- Euler round-trip (away from gimbal lock) ---
def test_euler_roundtrip():
    for _ in range(1000):
        phi = RNG.uniform(-np.pi, np.pi)
        theta = RNG.uniform(-np.pi/2 + 0.05, np.pi/2 - 0.05)
        psi = RNG.uniform(-np.pi, np.pi)
        R = euler_zyx_to_dcm(phi, theta, psi)
        p, t, s = dcm_to_euler_zyx(R)
        # compare matrices (angles can differ but rotation must match)
        assert np.allclose(euler_zyx_to_dcm(p, t, s), R, atol=1e-12)


# --- EXIT 4: reproduce gimbal lock at pitch = +90 deg ---
def test_gimbal_lock():
    theta = np.pi / 2
    # two different (roll, yaw) pairs that yield the SAME matrix:
    # at +90 pitch the matrix depends only on (psi - phi)
    M1 = euler_zyx_to_dcm(0.3, theta, 0.7)
    M2 = euler_zyx_to_dcm(0.3 + 0.4, theta, 0.7 + 0.4)
    assert np.allclose(M1, M2, atol=1e-12)   # roll & yaw are redundant here

    # but the quaternion is perfectly fine at gimbal lock:
    assert np.allclose(quat_to_dcm(dcm_to_quat(M1)), M1, atol=1e-12)