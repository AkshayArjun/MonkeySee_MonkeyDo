# test_nan2.py
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/focas/mcdepth_ws/src/robot_retarget/robot_retarget')
import rx200_kinematics as rx_kine

q0     = jnp.zeros(5)
target = jnp.array([0.0, 0.0, 0.3,
                    0.2, 0.0, 0.2,
                    0.35, 0.0, 0.1,
                    0.0, 0.0, 0.0, 1.0])

def just_skel(q):
    ROBOT_BASE  = jnp.zeros(3)
    t_shoulder  = target[:3]
    t_elbow     = target[3:6]
    t_hand      = target[6:9]
    r_elbow, r_hand, _ = rx_kine.forward_kinematics(q)
    robot_chain = jnp.stack([ROBOT_BASE, r_elbow, r_hand])
    human_chain = jnp.stack([t_shoulder, t_elbow, t_hand])
    d_h_elb = rx_kine.get_min_distance_to_chain(t_elbow, robot_chain)
    d_h_hnd = rx_kine.get_min_distance_to_chain(t_hand,  robot_chain)
    d_r_elb = rx_kine.get_min_distance_to_chain(r_elbow, human_chain)
    d_r_hnd = rx_kine.get_min_distance_to_chain(r_hand,  human_chain)
    return d_h_elb + d_h_hnd + d_r_elb + d_r_hnd

def just_theta(q):
    _, _, r_rot = rx_kine.forward_kinematics(q)
    tr  = jnp.trace(r_rot)
    s   = jnp.sqrt(jnp.maximum(1.0 + tr, 1e-6)) * 2.0
    w   = 0.25 * s
    x   = (r_rot[2, 1] - r_rot[1, 2]) / s
    y   = (r_rot[0, 2] - r_rot[2, 0]) / s
    z   = (r_rot[1, 0] - r_rot[0, 1]) / s
    r_quat      = jnp.array([x, y, z, w])
    t_quat      = target[9:13]
    t_quat_conj = jnp.array([-t_quat[0], -t_quat[1], -t_quat[2], t_quat[3]])
    tx, ty, tz, tw = t_quat_conj
    rx, ry, rz, rw = r_quat
    Qd = jnp.array([
        rw*tx + rx*tw + ry*tz - rz*ty,
        rw*ty - rx*tz + ry*tw + rz*tx,
        rw*tz + rx*ty - ry*tx + rz*tw,
        rw*tw - rx*tx - ry*ty - rz*tz
    ])
    # arctan2 instead of arccos — stable at Qd[3]=±1
    Qd_xyz_norm = jnp.sqrt(jnp.maximum(Qd[0]**2 + Qd[1]**2 + Qd[2]**2, 1e-12))
    return 2.0 * jnp.arctan2(Qd_xyz_norm, jnp.abs(Qd[3]))

def just_norm(q):
    _, r_hand, _ = rx_kine.forward_kinematics(q)
    return jnp.linalg.norm(r_hand)

# Test each component
print("=== Scalar values at q0 ===")
print(f"skel:  {just_skel(q0)}")
print(f"theta: {just_theta(q0)}")
print(f"norm:  {just_norm(q0)}")

print("\n=== Gradients ===")
g_skel  = jax.grad(just_skel)(q0)
print(f"grad skel:  {g_skel}")

g_theta = jax.grad(just_theta)(q0)
print(f"grad theta: {g_theta}")

g_norm  = jax.grad(just_norm)(q0)
print(f"grad norm:  {g_norm}")

# Check if NaN comes from linalg.norm at zero vector
print("\n=== NaN check on norm ===")
zero_vec = jnp.zeros(3)
print(f"norm of zeros: {jnp.linalg.norm(zero_vec)}")
print(f"grad of norm at zeros: {jax.grad(jnp.linalg.norm)(zero_vec)}")