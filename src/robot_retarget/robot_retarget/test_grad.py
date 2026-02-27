# test_grad.py — run this to find exactly where gradient dies
import jax
import jax.numpy as jnp
import numpy as np

# copy exact functions here to test in isolation
def skew(v):
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def matrix_exp_se3(S, theta):
    omega = S[:3]
    v = S[3:]
    omega_mat = skew(omega)
    R = jnp.eye(3) + jnp.sin(theta)*omega_mat + (1 - jnp.cos(theta))*(omega_mat @ omega_mat)
    v_mat = (jnp.eye(3)*theta +
             (1 - jnp.cos(theta))*omega_mat +
             (theta - jnp.sin(theta))*(omega_mat @ omega_mat))
    p = v_mat @ v
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(p)
    return T

S_LIST = jnp.array([
    [0.0, 0.0, 1.0,  0.0,     0.0,     0.0],
    [0.0, 1.0, 0.0, -0.10457, 0.0,     0.0],
    [0.0, 1.0, 0.0, -0.30457, 0.0,     0.05],
    [0.0, 1.0, 0.0, -0.30457, 0.0,     0.25],
    [1.0, 0.0, 0.0,  0.0,     0.30457, 0.0]
])
M_HOME = jnp.array([
    [1.0, 0.0, 0.0, 0.408575],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.30457],
    [0.0, 0.0, 0.0, 1.0]
])
M_ELBOW = jnp.array([
    [1.0, 0.0, 0.0, 0.05],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.30457],
    [0.0, 0.0, 0.0, 1.0]
])

def forward_kinematics(joint_angles):
    T = jnp.eye(4)
    T = T @ matrix_exp_se3(S_LIST[0], joint_angles[0])
    T = T @ matrix_exp_se3(S_LIST[1], joint_angles[1])
    T_elbow = T @ M_ELBOW
    elbow_pos = T_elbow[:3, 3]
    T = T @ matrix_exp_se3(S_LIST[2], joint_angles[2])
    T = T @ matrix_exp_se3(S_LIST[3], joint_angles[3])
    T = T @ matrix_exp_se3(S_LIST[4], joint_angles[4])
    T_hand = T @ M_HOME
    hand_pos = T_hand[:3, 3]
    hand_rot = T_hand[:3, :3]
    return elbow_pos, hand_pos, hand_rot

# ── Test 1: does FK produce non-zero jacobian? ──
def fk_hand(q):
    _, hand, _ = forward_kinematics(q)
    return hand

q0 = jnp.zeros(5)
J = jax.jacobian(fk_hand)(q0)
print(f"FK Jacobian at zero config:\n{J}")
print(f"FK Jacobian all zero: {jnp.allclose(J, 0.0)}")

# ── Test 2: does dist_point_to_segment have gradient? ──
def dist_point_to_segment(p, a, b):
    segment_vec = b - a
    point_vec   = p - a
    seg_len_sq  = jnp.dot(segment_vec, segment_vec)
    t = jnp.where(seg_len_sq > 1e-6,
                  jnp.dot(point_vec, segment_vec) / seg_len_sq, 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    closest_point = a + t * segment_vec
    return jnp.linalg.norm(p - closest_point)

p = jnp.array([0.1, 0.0, 0.2])
a = jnp.array([0.0, 0.0, 0.0])
b = jnp.array([0.0, 0.0, 0.3])
grad_dist = jax.grad(dist_point_to_segment)(p, a, b)
print(f"\ndist_point_to_segment grad w.r.t p: {grad_dist}")
print(f"dist grad all zero: {jnp.allclose(grad_dist, 0.0)}")

# ── Test 3: check gradient of full loss w.r.t each joint separately ──
import sys
sys.path.insert(0, '/home/focas/mcdepth_ws/src/robot_retarget/robot_retarget')
import rx200_kinematics as rx_kine

target = jnp.array([0.0, 0.0, 0.3,
                    0.2, 0.0, 0.2,
                    0.35, 0.0, 0.1,
                    0.0, 0.0, 0.0, 1.0])
weights = jnp.array([0.67, 0.33])

# perturb each joint and check if loss changes
print("\nNumerical gradient check:")
eps = 1e-3
for i in range(5):
    q_plus  = q0.at[i].set(eps)
    q_minus = q0.at[i].set(-eps)
    l_plus  = rx_kine.ocra_loss(q_plus,  target, weights)
    l_minus = rx_kine.ocra_loss(q_minus, target, weights)
    num_grad = (l_plus - l_minus) / (2*eps)
    print(f"  joint {i}: numerical grad = {num_grad:.6f}")