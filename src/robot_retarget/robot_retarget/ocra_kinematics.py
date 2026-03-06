import os
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import pyroki as pk
import yourdfpy
from ament_index_python.packages import get_package_share_directory

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
description_pkg = 'addverb_cobot_description'
URDF_FILE       = 'heal.urdf'  
EE_LINK_NAME    = 'end_effector'

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT LOADING
# ══════════════════════════════════════════════════════════════════════════════
URDF_PATH = os.path.join(
    get_package_share_directory(description_pkg), 'urdf', URDF_FILE
)

print(f"[ocra_kinematics] Loading URDF: {URDF_PATH}")

# load_meshes=True to ensure proper frame identification [cite: 140]
urdf_obj = yourdfpy.URDF.load(URDF_PATH, build_scene_graph=True, load_meshes=True)
robot = pk.Robot.from_urdf(urdf_obj)

# ── Extract kinematic chain (Root to EE) ──────────────────────────────────────
def _get_chain_indices(robot, ee_name):
    """Walks the joint-link hierarchy to extract primary nodes."""
    _all_links = list(robot.links.names)
    chain_indices = []
    
    try:
        current_idx = _all_links.index(ee_name)
    except ValueError:
        raise ValueError(f"End effector '{ee_name}' not found in robot links.")

    while current_idx != -1:
        chain_indices.append(current_idx)
        p_joint_idx = robot.links.parent_joint_indices[current_idx]
        if p_joint_idx == -1:
            break
        current_idx = robot.joints.parent_indices[p_joint_idx]
        
    chain_indices.reverse()
    return jnp.array(chain_indices, dtype=jnp.int32)

CHAIN_INDICES = _get_chain_indices(robot, EE_LINK_NAME)
_chain_names  = [robot.links.names[i] for i in CHAIN_INDICES]

print(f"[ocra_kinematics] Active Chain: {_chain_names}")
N_JOINTS = robot.joints.num_actuated_joints

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@jit
def get_min_distance_to_chain(point, chain_points):
    """Smooth-min distance from a 3D point to a polyline segment."""
    a = chain_points[:-1]
    b = chain_points[1:]

    seg_vec    = b - a
    pt_vec     = point - a
    seg_len_sq = jnp.sum(seg_vec ** 2, axis=-1)

    t = jnp.where(seg_len_sq > 1e-6, jnp.sum(pt_vec * seg_vec, axis=-1) / seg_len_sq, 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    
    seg_pt   = a + t[:, None] * seg_vec
    distance = jnp.linalg.norm(point - seg_pt, axis=-1)

    alpha = 10.0
    return -(1.0 / alpha) * jnp.log(jnp.sum(jnp.exp(-alpha * distance)))

# ══════════════════════════════════════════════════════════════════════════════
# FORWARD KINEMATICS
# ══════════════════════════════════════════════════════════════════════════════

@jit
def forward_kinematics(joint_angles):
    """Returns chain link positions and the EE quaternion (wxyz)."""
    Ts = robot.forward_kinematics(joint_angles)
    chain_positions = Ts[CHAIN_INDICES, 4:]      
    ee_quat_wxyz    = Ts[CHAIN_INDICES[-1], :4]  
    return chain_positions, ee_quat_wxyz

# ══════════════════════════════════════════════════════════════════════════════
# OCRA LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

@jit
def ocra_loss(joint_angles, target_flat, weights):
    """
    OCRA Objective: alpha*eps_s^2 + beta*eps_o^2 + gamma*eps_ee^2 [cite: 29]
    """
    t_shoulder = target_flat[0:3]
    t_elbow    = target_flat[3:6]
    t_hand     = target_flat[6:9]
    t_q_xyzw   = target_flat[9:13] 

    human_chain = jnp.stack([t_shoulder, t_elbow, t_hand])
    human_ell   = jnp.sum(jnp.linalg.norm(jnp.diff(human_chain, axis=0), axis=-1))

    robot_chain, r_q_wxyz = forward_kinematics(joint_angles)
    robot_ell = jnp.sum(jnp.linalg.norm(jnp.diff(robot_chain, axis=0), axis=-1))
    
    ell = human_ell + robot_ell + 1e-8

    # Skeleton Error (eps_s) [cite: 37, 41]
    d_h_elb = get_min_distance_to_chain(t_elbow, robot_chain)
    d_h_hnd = get_min_distance_to_chain(t_hand,  robot_chain)
    d_r_all = jax.vmap(get_min_distance_to_chain, in_axes=(0, None))(robot_chain, human_chain)
    skel_err = (d_h_elb + d_h_hnd + jnp.sum(d_r_all)) / ell

    # Orientation Error (eps_o) [cite: 48, 49, 50]
    t_q_wxyz = jnp.array([t_q_xyzw[3], t_q_xyzw[0], t_q_xyzw[1], t_q_xyzw[2]])
    t_q_conj = jnp.array([t_q_wxyz[0], -t_q_wxyz[1], -t_q_wxyz[2], -t_q_wxyz[3]])
    
    tw, tx, ty, tz = t_q_conj
    rw, rx, ry, rz = r_q_wxyz

    Qd = jnp.array([
        rw*tw - rx*tx - ry*ty - rz*tz,
        rw*tx + rx*tw + ry*tz - rz*ty,
        rw*ty - rx*tz + ry*tw + rz*tx,
        rw*tz + rx*ty - ry*tx + rz*tw 
    ])

    Qd_xyz_norm = jnp.sqrt(jnp.maximum(Qd[1]**2 + Qd[2]**2 + Qd[3]**2, 1e-12))
    theta_d     = 2.0 * jnp.arctan2(Qd_xyz_norm, jnp.abs(Qd[0]))
    orient_err = theta_d / jnp.pi 

    # End-Effector Position Error
    hand_err = jnp.linalg.norm(robot_chain[-1] - t_hand) / ell

    alpha, beta, gamma = weights[0], weights[1], weights[2]
    return alpha * (skel_err**2) + beta * (orient_err**2) + gamma * (hand_err**2)

loss_and_grad = value_and_grad(ocra_loss)