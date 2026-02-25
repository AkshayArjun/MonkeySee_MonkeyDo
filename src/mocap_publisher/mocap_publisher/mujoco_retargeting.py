"""
mujoco_retargeting.py — Direction-Vector Analytical IK
Maps human arm unit-vectors (scaled to robot link lengths) → RX200 5-DOF joints.
No MJINX/JAX dependency.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import socket
import json

# RX200 link lengths (meters, from MJCF)
L1 = 0.200   # shoulder → elbow
L2 = 0.200   # elbow → wrist_link
L3 = 0.065   # wrist_link → gripper

# Joint limits
WAIST_LIM        = (-3.14159, 3.14159)
SHOULDER_LIM     = (-1.88496, 1.97222)
ELBOW_LIM        = (-1.88496, 1.62316)
WRIST_ANGLE_LIM  = (-1.74533, 2.14675)
WRIST_ROTATE_LIM = (-3.14159, 3.14159)


def clamp(v, lo, hi): return max(lo, min(hi, float(v)))


def quat_roll(q):
    qx, qy, qz, qw = q
    return float(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))


def direction_ik(upper_dir, fore_dir, hand_quat):
    """
    Compute RX200 joint angles from arm direction unit vectors.

    upper_dir : unit vector shoulder→elbow  (MuJoCo frame)
    fore_dir  : unit vector elbow→wrist     (MuJoCo frame)

    In MuJoCo:  X=right, Y=forward, Z=up
    The robot reaches primarily in the +Y (forward) direction.
    """
    # Scale directions to robot link lengths
    elbow_pos = upper_dir * L1             # elbow in shoulder frame
    ee_pos    = elbow_pos + fore_dir * (L2 + L3)  # EE in shoulder frame

    ex, ey, ez = ee_pos

    # ── Waist: rotate to face the horizontal projection of the EE ──
    # atan2(x, y) → 0 when arm points straight forward (+Y), +90° when pointing right (+X)
    waist = clamp(np.arctan2(ex, ey), *WAIST_LIM)

    # ── Project into the arm's vertical plane (radial dist + height) ──
    r = np.sqrt(ex**2 + ey**2)   # horizontal distance from base
    z = ez                         # height above shoulder

    # ── Two-link IK (L1, L2+L3) in the (r, z) plane ──
    arm_len = L2 + L3
    d = np.sqrt(r**2 + z**2)
    d = clamp(d, 0.01, L1 + arm_len - 0.005)

    # Elbow angle via law-of-cosines
    cos_e = clamp((L1**2 + arm_len**2 - d**2) / (2*L1*arm_len), -1, 1)
    elbow_inner = np.arccos(cos_e)                   # 0 = straight, π = fully bent
    elbow = clamp(-(np.pi - elbow_inner), *ELBOW_LIM)  # RX200 convention

    # Shoulder angle = elevation angle to EE + internal correction
    alpha = np.arctan2(z, r)                          # elevation toward EE
    cos_b = clamp((L1**2 + d**2 - arm_len**2) / (2*L1*d), -1, 1)
    beta  = np.arccos(cos_b)
    shoulder = clamp(alpha + beta, *SHOULDER_LIM)

    # ── Wrist pitch: keep tool parallel to the fore_dir vector ──
    # Desired wrist pitch = elevation of fore_dir relative to horizontal
    # Here we just compensate for shoulder+elbow to keep gripper horizontal
    wrist_angle = clamp(-(shoulder + elbow), *WRIST_ANGLE_LIM)

    # ── Wrist rotate: from hand quaternion roll ──
    wrist_rotate = clamp(quat_roll(hand_quat), *WRIST_ROTATE_LIM)

    return waist, shoulder, elbow, wrist_angle, wrist_rotate


def smooth(prev, tgt, a=0.3):
    return prev + a * (tgt - prev)


def main():
    mj_model = mujoco.MjModel.from_xml_path("oakd_mocap/scene.xml")
    mj_data  = mujoco.MjData(mj_model)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5005))
    sock.setblocking(False)
    print("Analytical IK (direction-vector) listening on UDP 127.0.0.1:5005 ...")

    calibrated   = False
    q_smooth     = np.zeros(5)

    print("Opening MuJoCo Physics Viewer...")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        last_render = time.time()

        while viewer.is_running():
            try:
                data, _ = sock.recvfrom(4096)
                payload  = json.loads(data.decode())

                if payload.get("state") == "TRACKING":
                    scale_upper = float(payload.get("scale_upper", 1.0))
                    scale_fore  = float(payload.get("scale_fore",  1.0))

                    if not calibrated:
                        calibrated = True
                        print(f"[IK] scale_upper={scale_upper:.3f}  scale_fore={scale_fore:.3f}")

                    h_shoulder  = np.array(payload["h_shoulder"], dtype=float)
                    h_elbow     = np.array(payload["h_elbow"],    dtype=float)
                    h_wrist     = np.array(payload["h_wrist"],    dtype=float)
                    h_hand_quat = payload.get("h_hand_quat", [0., 0., 0., 1.])

                    # ── Raw human direction vectors ──
                    raw_upper = h_elbow - h_shoulder   # shoulder → elbow in MuJoCo space
                    raw_fore  = h_wrist - h_elbow      # elbow    → wrist  in MuJoCo space

                    # Normalise to unit vectors (direction only)
                    ua_len = np.linalg.norm(raw_upper)
                    fa_len = np.linalg.norm(raw_fore)
                    if ua_len < 1e-4 or fa_len < 1e-4:
                        continue

                    upper_dir = raw_upper / ua_len
                    fore_dir  = raw_fore  / fa_len

                    # Compute joint angles from pure arm direction
                    q_target = np.array(direction_ik(upper_dir, fore_dir, h_hand_quat))

                    # Smooth
                    q_smooth = smooth(q_smooth, q_target, a=0.25)

                    # Drive all 5 actuators
                    mj_data.ctrl[0] = q_smooth[0]   # waist
                    mj_data.ctrl[1] = q_smooth[1]   # shoulder
                    mj_data.ctrl[2] = q_smooth[2]   # elbow
                    mj_data.ctrl[3] = q_smooth[3]   # wrist_angle
                    mj_data.ctrl[4] = q_smooth[4]   # wrist_rotate

            except BlockingIOError:
                pass
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"[Error] {e}")

            mujoco.mj_step(mj_model, mj_data)

            if time.time() - last_render > 1.0/60.0:
                viewer.sync()
                last_render = time.time()


if __name__ == '__main__':
    main()
