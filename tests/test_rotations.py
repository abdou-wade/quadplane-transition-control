"""Property tests for dynamics/rotations.py -- the L0.2 EXIT criteria.

These aren't example-based ("does 90 deg work?"). They're PROPERTY tests:
run 1000 random rotations through each identity and assert it holds every time.
That's how you catch the silent sign/convention bugs that a single hand-picked
case would miss.
"""
import numpy as np
from dynamics.rotations import (
    hat, vee, Exp, Log,
    euler_zyx_to_dcm, dcm_to_euler_zyx,
    quat_to_dcm, dcm_to_quat,
)

# fixed seed -> same "random" rotations every run -> reproducible failures
RNG = np.random.default_rng(0)


def random_rotvec(max_angle=np.pi - 0.01):
    # a random rotation vector: random unit axis * random angle in [0, pi).
    # we stop just short of pi because at exactly pi the axis is ambiguous
    # (a known edge case, handled separately in the full version).
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * RNG.uniform(0, max_angle)


# --- EXIT 1: Log(Exp(v)) == v ------------------------------------------
# Exp wraps a rotation vector onto the manifold; Log unwraps it. Doing both
# must return the original vector. This proves Exp and Log are true inverses.
def test_log_exp_roundtrip():
    for _ in range(1000):
        v = random_rotvec()
        assert np.allclose(Log(Exp(v)), v, atol=1e-9)


# --- EXIT 2: R^T R == I to 1e-12 ---------------------------------------
# Every matrix Exp produces must be a valid rotation: orthonormal (R^T R = I)
# and proper (det = +1, i.e. a real rotation, not a reflection).
def test_orthonormal():
    for _ in range(1000):
        R = Exp(random_rotvec())
        assert np.max(np.abs(R.T @ R - np.eye(3))) < 1e-12
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


# --- EXIT 3: quat <-> DCM round-trip -----------------------------------
# Matrix -> quaternion -> matrix must return the original matrix.
# This proves quat_to_dcm and dcm_to_quat agree and are inverses.
def test_quat_dcm_roundtrip():
    for _ in range(1000):
        R = Exp(random_rotvec())
        assert np.allclose(quat_to_dcm(dcm_to_quat(R)), R, atol=1e-12)


# --- hat/vee: inverse pair + the cross-product property ----------------
# hat turns a vector into a skew matrix; vee pulls it back out.
# The defining property: hat(v) @ w == v cross w.
def test_hat_vee():
    for _ in range(1000):
        v = RNG.normal(size=3)
        w = RNG.normal(size=3)
        assert np.allclose(hat(v) @ w, np.cross(v, w))
        assert np.allclose(vee(hat(v)), v)


# --- Euler round-trip (away from gimbal lock) --------------------------
# Build a DCM from (roll, pitch, yaw), recover the angles, rebuild the DCM.
# We compare the MATRICES, not the angles: different angle triples can encode
# the same rotation, but the rotation itself must match exactly.
def test_euler_roundtrip():
    for _ in range(1000):
        phi = RNG.uniform(-np.pi, np.pi)
        theta = RNG.uniform(-np.pi/2 + 0.05, np.pi/2 - 0.05)   # stay off +/-90
        psi = RNG.uniform(-np.pi, np.pi)
        R = euler_zyx_to_dcm(phi, theta, psi)
        p, t, s = dcm_to_euler_zyx(R)
        assert np.allclose(euler_zyx_to_dcm(p, t, s), R, atol=1e-12)


# --- EXIT 4: reproduce gimbal lock at pitch = +90 deg ------------------
# At +90 pitch the roll axis and yaw axis line up, so they become redundant:
# only (yaw - roll) matters, not roll and yaw separately. We prove this by
# showing two DIFFERENT (roll, yaw) inputs produce the SAME matrix -> the
# Euler representation has lost a degree of freedom. THAT is gimbal lock.
# Then we show the quaternion round-trips perfectly at the same point:
# quaternions have no such singularity, which is the whole reason we use them.
def test_gimbal_lock():
    theta = np.pi / 2
    M1 = euler_zyx_to_dcm(0.3, theta, 0.7)
    M2 = euler_zyx_to_dcm(0.3 + 0.4, theta, 0.7 + 0.4)   # shift roll & yaw together
    assert np.allclose(M1, M2, atol=1e-12)               # same matrix -> redundant

    # the quaternion is perfectly well-defined at gimbal lock:
    assert np.allclose(quat_to_dcm(dcm_to_quat(M1)), M1, atol=1e-12)
