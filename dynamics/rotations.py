"""Rotations on SO(3): hat/vee, Exp/Log, quaternions, DCM.

Convention: active rotations, quaternions scalar-first [w,x,y,z], angles in radians.
"""

# Import libraries
import numpy as np
## Hat: turn a 3-vector into a 3x3 skew-symmetric matrix 
def hat(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([
        [ 0 , -z , y ],
        [ z , 0 , -x ],
        [ -y , x , 0 ],
    ])

    ## Vee: inverse of hat
def vee(M):
    x = M[2, 1]
    y = M[0, 2]
    z = M[1, 0]
    return np.array([x, y, z])

    ## Exp: from rotation vector (axis*angle) to rotation matrix (Rodrigues' formula)
def Exp(v):
    v = np.asarray(v, dtype=float)     # needs to be  a float array
    theta = np.linalg.norm(v)          # theta = length of the vector v 

    # no rotation: avoid divide-by-zero, just return identity
    if theta < 1e-12:
        return np.eye(3)

    u = v / theta                      # unit axis
    K = hat(u)                         # let K being the hat 

    # Rodrigues formula 
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R
    ## Log: rotation matrix -> rotation vector (inverse of Exp)
def Log(R):
    R = np.asarray(R, dtype=float)

    # angle from the trace:  
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  
    theta = np.arccos(cos_theta)

    # no rotation: return zero vector
    if theta < 1e-12:
        return np.zeros(3)

    # axis from the antisymmetric part (R - R^T), pulled out with vee
    axis = vee(R - R.T) / (2.0 * np.sin(theta))
    return theta * axis
    ## Rz, Ry, Rx: single-axis rotation matrices (active, radians)
def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])


## Euler ZYX -> DCM:  roll(phi) about x, pitch(theta) about y, yaw(psi) about z
def euler_zyx_to_dcm(phi, theta, psi):
    return Rz(psi) @ Ry(theta) @ Rx(phi)
    ## DCM -> Euler ZYX: recover (roll phi, pitch theta, yaw psi) from a rotation matrix
def dcm_to_euler_zyx(R):
    R = np.asarray(R, dtype=float)
    r20 = np.clip(R[2, 0], -1.0, 1.0)   # this entry is -sin(pitch)

    # gimbal lock check: |R[2,0]| ~ 1  means pitch = +/- 90 deg
    if abs(r20) < 1.0 - 1e-10:
        theta = -np.arcsin(r20)                 # pitch
        phi   = np.arctan2(R[2, 1], R[2, 2])    # roll
        psi   = np.arctan2(R[1, 0], R[0, 0])    # yaw
    else:
        # gimbal lock: pitch is +/-90, roll and yaw are coupled.
        # convention: set yaw = 0 and fold everything into roll.
        psi = 0.0
        if r20 <= -1.0 + 1e-10:                 # pitch = +90
            theta = np.pi / 2
            phi = np.arctan2(R[0, 1], R[0, 2])
        else:                                   # pitch = -90
            theta = -np.pi / 2
            phi = np.arctan2(-R[0, 1], -R[0, 2])
    return phi, theta, psi
## Quaternion helpers (Hamilton, scalar-first [w, x, y, z], unit norm)

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)          # force unit length
    if q[0] < 0:                       # canonical: keep w >= 0 (double-cover fix)
        q = -q
    return q


## Quaternion -> DCM  (Sola eq 138)
def quat_to_dcm(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    ## DCM -> Quaternion  (Shepperd's method: pick largest component for stability)
def dcm_to_quat(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0        # S = 4w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # S = 4x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # S = 4y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # S = 4z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z]))

    ## ------------------------------------------------------------------
## NOTES 
## ------------------------------------------------------------------
## WHAT THIS FILE IS: the one place all attitude math lives. Rotations
## live on a CURVED space (SO(3)); you can't add them like normal numbers.
##
## THE CORE IDEA (Lie): to compute with rotations, go to a FLAT space
##   - Exp(v): rotation vector (axis*angle) -> rotation matrix  (wrap onto curve)
##   - Log(R): rotation matrix -> rotation vector               (unwrap to flat)
##   Exp is Rodrigues' formula. Log is its inverse. Round-trip = identity.
##
## hat/vee: format converters. hat turns a 3-vector into a skew matrix
##   (the [v]x used inside Rodrigues). vee pulls the vector back out.
##   Key fact: hat(v) @ w == cross(v, w).
##
## EULER (ZYX = yaw->pitch->roll): intuitive (how a pilot thinks) but
##   BREAKS at pitch = +/-90 deg = GIMBAL LOCK. At +/-90, the roll axis
##   and yaw axis line up -> you lose a degree of freedom -> yaw undefined.
##   That's the if/else in dcm_to_euler_zyx.
##
## QUATERNIONS: the fix. 4 numbers, unit length, NO gimbal lock ever.
##   Not intuitive but they never break -> what the flight code actually uses.
##   Gotcha: q and -q are the SAME rotation (double cover) -> quat_normalize
##   forces w >= 0 so it's unique.
##   dcm_to_quat uses Shepperd's method (4 branches) to avoid dividing by ~0.
##
## WHY I BUILT THIS: quadplane pitches through ~90 deg during transition,
##   right where Euler dies. So attitude must be quaternion-based end to end.
##
## TESTED (EXIT criteria, see tests/test_rotations.py):
##   Log(Exp(v))=v | R^T R = I to 1e-12 | quat<->DCM round-trip | gimbal lock