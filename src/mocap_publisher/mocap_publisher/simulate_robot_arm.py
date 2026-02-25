import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def init_oakd():
    print("Initializing OAK-D Pro Pipeline...")
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

    xoutRgb.setStreamName("rgb")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    camRgb.setInterleaved(False)
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)
    depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)

    camRgb.video.link(xoutRgb.input)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    
    depth.depth.link(spatialLocationCalculator.inputDepth)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
    
    return pipeline

def get_direction_vector(p1, p2):
    """Calculates the normalized direction vector from point 1 to point 2."""
    b = np.array(p2)
    a = np.array(p1)
    vector = b - a
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector
    return vector / norm

def main():
    try:
        import matplotlib
        matplotlib.use("macosx") 
    except ImportError:
        pass

    # Generic Robot Arm Link Lengths (in millimeters)
    # We will simulate a standard 5-DoF robotic arm proportions
    ROBOT_BASE_Z = 100       # Robot sits 100mm off the ground
    ROBOT_BICEP_LEN = 300    # Length of upper arm
    ROBOT_FOREARM_LEN = 250  # Length of lower arm

    pipeline = init_oakd()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    plt.ion()
    # Create a 1x2 grid: Left for Human, Right for Robot
    fig = plt.figure(figsize=(14, 7))
    ax_human = fig.add_subplot(121, projection='3d')
    ax_robot = fig.add_subplot(122, projection='3d')
    
    print("Connecting to OAK-D Device...")
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        last_world_landmarks = None
        last_root_abs = None

        print("Opening Simulation Graph...")
        while True:
            inRgb = qRgb.tryGet()
            inSpatialData = spatialCalcQueue.tryGet()

            if inRgb is not None:
                cv_frame = inRgb.getCvFrame()
                h, w, _ = cv_frame.shape
                
                frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks and results.pose_world_landmarks:
                    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    
                    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                        cx = int(((left_shoulder.x + right_shoulder.x) / 2) * w)
                        cy = int(((left_shoulder.y + right_shoulder.y) / 2) * h)
                        
                        cx = max(10, min(w - 11, cx))
                        cy = max(10, min(h - 11, cy))
                        
                        cfg = dai.SpatialLocationCalculatorConfig()
                        configData = dai.SpatialLocationCalculatorConfigData()
                        configData.depthThresholds.lowerThreshold = 100
                        configData.depthThresholds.upperThreshold = 10000
                        configData.roi = dai.Rect(dai.Point2f(cx - 10, cy - 10), dai.Point2f(cx + 10, cy + 10))
                        cfg.addROI(configData)
                        spatialCalcConfigInQueue.send(cfg)
                        
                        last_world_landmarks = results.pose_world_landmarks

            if inSpatialData is not None and last_world_landmarks is not None:
                spatialData = inSpatialData.getSpatialLocations()
                if len(spatialData) > 0:
                    root_x = spatialData[0].spatialCoordinates.x
                    root_y = spatialData[0].spatialCoordinates.y
                    root_z = spatialData[0].spatialCoordinates.z
                    
                    if root_z > 0:
                        last_root_abs = (root_x, root_y, root_z)
                        
            # Execute Retargeting & Rendering
            if last_world_landmarks is not None and last_root_abs is not None:
                ax_human.clear()
                ax_robot.clear()
                
                # --- [1] HUMAN KINEMATICS ---
                root_x, root_y, root_z = last_root_abs
                
                mp_left_shoulder = last_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                mp_right_shoulder = last_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                mp_anchor_x = ((mp_left_shoulder.x + mp_right_shoulder.x) / 2) * 1000
                mp_anchor_y = ((mp_left_shoulder.y + mp_right_shoulder.y) / 2) * 1000
                mp_anchor_z = ((mp_left_shoulder.z + mp_right_shoulder.z) / 2) * 1000
                
                shift_x = root_x - mp_anchor_x
                shift_y = root_y - mp_anchor_y
                shift_z = root_z - mp_anchor_z
                
                def get_plot_pt(mp_landmark):
                    # Convert to mm, shift, and map to Matplotlib (X, Depth, UP)
                    lx = (mp_landmark.x * 1000) + shift_x
                    ly = (mp_landmark.z * 1000) + shift_z
                    lz = -((mp_landmark.y * 1000) + shift_y)
                    return [lx, ly, lz]

                # Extract True Absolute Human Right Arm Points
                h_shoulder = get_plot_pt(last_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                h_elbow = get_plot_pt(last_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                h_wrist = get_plot_pt(last_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                
                # Plot Human Arm
                hx = [h_shoulder[0], h_elbow[0], h_wrist[0]]
                hy = [h_shoulder[1], h_elbow[1], h_wrist[1]]
                hz = [h_shoulder[2], h_elbow[2], h_wrist[2]]
                ax_human.plot(hx, hy, hz, color='blue', linewidth=4, marker='o', markersize=8)
                ax_human.scatter(*h_shoulder, color='red', s=100)
                ax_human.scatter(*h_elbow, color='green', s=100)
                ax_human.scatter(*h_wrist, color='yellow', s=100)

                # --- [2] ROBOT RETARGETING ---
                # A. Calculate Direction Vectors of Human Joints
                v_bicep = get_direction_vector(h_shoulder, h_elbow)
                v_forearm = get_direction_vector(h_elbow, h_wrist)

                # B. Build Robot Structure dynamically from Vectors
                r_base = [0, 0, 0]
                r_shoulder = [0, 0, ROBOT_BASE_Z]
                
                # Multiply human vector * robot physical length to find the exact coordinate for the Robot Elbow
                r_elbow = [
                    r_shoulder[0] + (v_bicep[0] * ROBOT_BICEP_LEN),
                    r_shoulder[1] + (v_bicep[1] * ROBOT_BICEP_LEN),
                    r_shoulder[2] + (v_bicep[2] * ROBOT_BICEP_LEN)
                ]

                # Same for Forearm -> Wrist
                r_wrist = [
                    r_elbow[0] + (v_forearm[0] * ROBOT_FOREARM_LEN),
                    r_elbow[1] + (v_forearm[1] * ROBOT_FOREARM_LEN),
                    r_elbow[2] + (v_forearm[2] * ROBOT_FOREARM_LEN)
                ]

                # Plot Robot Arm
                rx = [r_base[0], r_shoulder[0], r_elbow[0], r_wrist[0]]
                ry = [r_base[1], r_shoulder[1], r_elbow[1], r_wrist[1]]
                rz = [r_base[2], r_shoulder[2], r_elbow[2], r_wrist[2]]
                ax_robot.plot(rx, ry, rz, color='black', linewidth=6, marker='s', markersize=10)
                ax_robot.plot([r_base[0], r_shoulder[0]], [r_base[1], r_shoulder[1]], [r_base[2], r_shoulder[2]], color='gray', linewidth=8) # Base Pillar
                ax_robot.scatter(*r_shoulder, color='red', s=150)
                ax_robot.scatter(*r_elbow, color='green', s=150)
                ax_robot.scatter(*r_wrist, color='yellow', s=150)

                # --- [3] Lock Graph Bounds ---
                BOUND = 600
                # Human Bounds
                ax_human.set_xlim([h_shoulder[0] - BOUND, h_shoulder[0] + BOUND])
                ax_human.set_ylim([h_shoulder[1] - BOUND, h_shoulder[1] + BOUND])
                ax_human.set_zlim([h_shoulder[2] - BOUND, h_shoulder[2] + BOUND])
                ax_human.set_title("Source (Human Arm)")
                ax_human.view_init(elev=10, azim=-60)

                # Robot Bounds (Fixed to Origin Base)
                ROBOT_BOUND = 500
                ax_robot.set_xlim([-ROBOT_BOUND, ROBOT_BOUND])
                ax_robot.set_ylim([-ROBOT_BOUND, ROBOT_BOUND])
                ax_robot.set_zlim([0, ROBOT_BOUND + 200]) # Base always at 0
                ax_robot.set_title("Target (Robot Simulation)")
                ax_robot.view_init(elev=10, azim=-60)
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

if __name__ == '__main__':
    main()
