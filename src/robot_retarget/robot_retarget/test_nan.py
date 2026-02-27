# test_nan.py
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/focas/mcdepth_ws/src/robot_retarget/robot_retarget')
import rx200_kinematics as rx_kine

q0      = jnp.zeros(5)
target  = jnp.array([0.0, 0.0, 0.3,
                     0.2, 0.0, 0.2,
                     0.35, 0.0, 0.1,
                     0.0, 0.0, 0.0, 1.0])
weights = jnp.array([0.67, 0.33])

# Check forward kinematics output
r_elbow, r_hand, r_rot = rx_kine.forward_kinematics(q0)
print(f"r_elbow: {r_elbow}")
print(f"r_hand:  {r_hand}")
print(f"r_rot:\n{r_rot}")

# Check rotation matrix trace
tr = jnp.trace(r_rot)
print(f"\ntrace: {tr}")
print(f"1 + trace: {1.0 + tr}")

# This is the problematic line
s = jnp.sqrt(jnp.maximum(1.0 + tr, 1e-6)) * 2.0
print(f"s: {s}")

w = 0.25 * s
x = (r_rot[2, 1] - r_rot[1, 2]) / s
y = (r_rot[0, 2] - r_rot[2, 0]) / s
z = (r_rot[1, 0] - r_rot[0, 1]) / s
r_quat = jnp.array([x, y, z, w])
print(f"r_quat: {r_quat}")

# Now check gradient through each step
def just_orient(q):
    _, _, r_rot = rx_kine.forward_kinematics(q)
    tr = jnp.trace(r_rot)
    s  = jnp.sqrt(jnp.maximum(1.0 + tr, 1e-6)) * 2.0
    w  = 0.25 * s
    x  = (r_rot[2, 1] - r_rot[1, 2]) / s
    y  = (r_rot[0, 2] - r_rot[2, 0]) / s
    z  = (r_rot[1, 0] - r_rot[0, 1]) / s
    return jnp.array([x, y, z, w])

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
    tr = jnp.trace(r_rot)
    s  = jnp.sqrt(jnp.maximum(1.0 + tr, 1e-6)) * 2.0
    w  = 0.25 * s
    x  = (r_rot[2, 1] - r_rot[1, 2]) / s
    y  = (r_rot[0, 2] - r_rot[2, 0]) / s
    z  = (r_rot[1, 0] - r_rot[0, 1]) / s
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
    return 2.0 * jnp.arccos(jnp.clip(jnp.abs(Qd[3]), 0.0, 1.0))

g_skel   = jax.grad(just_skel)(q0)
g_orient = jax.grad(just_orient)(q0)
g_theta  = jax.grad(just_theta)(q0)

print(f"\ngrad skel:   {g_skel}")
print(f"grad orient: {g_orient}")
print(f"grad theta:  {g_theta}")