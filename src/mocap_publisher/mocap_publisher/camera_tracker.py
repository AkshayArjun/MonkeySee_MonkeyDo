import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

ROBOT_UPPER_ARM = 0.20
ROBOT_FOREARM   = 0.265


def init_oakd():
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    # Use 720p for lower processing load while keeping good resolution
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camRgb.setInterleaved(False)
    camRgb.setFps(15)
    camRgb.setPreviewSize(640, 360)  # Smaller preview for faster processing
    camRgb.preview.link(xoutRgb.input)

    return pipeline


def vec3(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def to_mujoco(mp_vec):
    """MediaPipe world → MuJoCo: X=same, Y=-Z_mp, Z=-Y_mp"""
    return np.array([mp_vec[0], -mp_vec[2], -mp_vec[1]], dtype=np.float32)


def rotation_matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    return [float(qx), float(qy), float(qz), float(qw)]


def main():
    print("Initializing OAK-D Pipeline (RGB only, no depth)...")
    pipeline = init_oakd()

    # Faster model: complexity=0 (lite), single image mode off
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,          # Fastest model
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,          # Fastest hand model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_drawing = mp.solutions.drawing_utils

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    state = "CALIBRATION"
    CALIB_FRAMES = 30
    calib_upper = []
    calib_fore  = []
    scale_upper = 1.0
    scale_fore  = 1.0

    last_hand_quat = [0.0, 0.0, 0.0, 1.0]
    frame_count = 0  # For frame skipping

    print("Stand in T-POSE (arms out to both sides) to calibrate.")

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        while True:
            inRgb = qRgb.tryGet()
            if inRgb is None:
                continue

            cv_frame = inRgb.getCvFrame()
            h, w = cv_frame.shape[:2]
            frame_count += 1

            # Run MediaPipe on every frame (already small 640x360)
            frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            pose_results = pose.process(frame_rgb)

            # Only run hand model every 2nd frame to save CPU
            hand_results = None
            if frame_count % 2 == 0:
                hand_results = hands.process(frame_rgb)

            frame_rgb.flags.writeable = True

            # ── Hand Orientation ──
            if hand_results and hand_results.multi_hand_world_landmarks:
                hlm = hand_results.multi_hand_world_landmarks[0].landmark
                wrist_pt  = vec3(hlm[mp_hands.HandLandmark.WRIST])
                index_mcp = vec3(hlm[mp_hands.HandLandmark.INDEX_FINGER_MCP])
                pinky_mcp = vec3(hlm[mp_hands.HandLandmark.PINKY_MCP])
                fwd  = index_mcp - wrist_pt
                side = pinky_mcp - wrist_pt
                y_ax = fwd / (np.linalg.norm(fwd) + 1e-8)
                z_tmp = np.cross(side, fwd)
                z_ax  = z_tmp / (np.linalg.norm(z_tmp) + 1e-8)
                x_ax  = np.cross(y_ax, z_ax)
                R = np.column_stack([x_ax, y_ax, z_ax])
                last_hand_quat = rotation_matrix_to_quat(R)
                mp_drawing.draw_landmarks(
                    cv_frame,
                    hand_results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,200,200), thickness=1)
                )

            # ── Pose Joints ──
            if pose_results.pose_landmarks and pose_results.pose_world_landmarks:
                mp_drawing.draw_landmarks(
                    cv_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(200,80,0), thickness=2)
                )

                wl = pose_results.pose_world_landmarks.landmark
                L  = mp_pose.PoseLandmark

                r_shoulder_mp = vec3(wl[L.RIGHT_SHOULDER.value])
                r_elbow_mp    = vec3(wl[L.RIGHT_ELBOW.value])
                r_wrist_mp    = vec3(wl[L.RIGHT_WRIST.value])

                pl = pose_results.pose_landmarks.landmark
                r_wrist_vis = pl[L.RIGHT_WRIST.value].visibility
                r_elbow_vis = pl[L.RIGHT_ELBOW.value].visibility
                l_wrist_vis = pl[L.LEFT_WRIST.value].visibility

                # ═══ CALIBRATION ═══
                if state == "CALIBRATION":
                    shoulder_y = r_shoulder_mp[1]
                    elbow_y    = r_elbow_mp[1]
                    is_t_pose  = (
                        r_wrist_vis > 0.5 and
                        r_elbow_vis > 0.5 and
                        l_wrist_vis > 0.5 and
                        abs(elbow_y - shoulder_y) < 0.12
                    )

                    progress = int((len(calib_upper) / CALIB_FRAMES) * 100)

                    # UI overlay
                    cv2.rectangle(cv_frame, (0, 0), (w, 120), (0, 0, 0), -1)
                    cv2.putText(cv_frame, "T-POSE  Stretch arms out to both sides",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    bar_w = w - 40
                    cv2.rectangle(cv_frame, (20, 55), (20 + bar_w, 80), (50, 50, 50), -1)
                    fill = int(bar_w * progress / 100)
                    cv2.rectangle(cv_frame, (20, 55), (20 + fill, 80), (0, 220, 0), -1)
                    cv2.putText(cv_frame, f"{progress}%", (20 + bar_w + 4, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
                    if is_t_pose:
                        cv2.putText(cv_frame, "Hold still...", (20, 108),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    if is_t_pose:
                        calib_upper.append(np.linalg.norm(r_elbow_mp - r_shoulder_mp))
                        calib_fore.append(np.linalg.norm(r_wrist_mp  - r_elbow_mp))
                        if len(calib_upper) >= CALIB_FRAMES:
                            ua = float(np.median(calib_upper))
                            fa = float(np.median(calib_fore))
                            scale_upper = ROBOT_UPPER_ARM / ua
                            scale_fore  = ROBOT_FOREARM   / fa
                            state = "TRACKING"
                            print(f"[Calibrated] upper={ua*100:.1f}cm scale={scale_upper:.3f} | fore={fa*100:.1f}cm scale={scale_fore:.3f}")
                    else:
                        if calib_upper:
                            calib_upper.pop()
                        if calib_fore:
                            calib_fore.pop()

                # ═══ TRACKING ═══
                elif state == "TRACKING":
                    cv2.rectangle(cv_frame, (0, 0), (w, 52), (0, 0, 0), -1)
                    cv2.putText(cv_frame, f"TRACKING  |  R=recal  scale_u:{scale_upper:.2f}  scale_f:{scale_fore:.2f}",
                                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    payload = {
                        "state":       "TRACKING",
                        "h_shoulder":  to_mujoco(r_shoulder_mp).tolist(),
                        "h_elbow":     to_mujoco(r_elbow_mp).tolist(),
                        "h_wrist":     to_mujoco(r_wrist_mp).tolist(),
                        "h_hand_quat": last_hand_quat,
                        "scale_upper": scale_upper,
                        "scale_fore":  scale_fore,
                    }
                    sock.sendto(json.dumps(payload).encode(), (UDP_IP, UDP_PORT))

            cv2.imshow("Mocap Camera Tracker", cv_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                state = "CALIBRATION"
                calib_upper.clear()
                calib_fore.clear()
                print("Recalibrating...")


if __name__ == '__main__':
    main()
