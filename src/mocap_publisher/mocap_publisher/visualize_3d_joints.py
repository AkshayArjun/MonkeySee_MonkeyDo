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

def main():
    try:
        import matplotlib
        matplotlib.use("macosx")  # Use native macOS backend instead of TkAgg to prevent Tcl_Panic threading crash
    except ImportError:
        pass

    pipeline = init_oakd()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Connecting to OAK-D Device...")
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        print("Opening 3D Visualizer...")

        # We will hold on to the last known complete skeleton so the plot doesn't blink
        last_world_landmarks = None
        last_root_abs = None

        while True:
            inRgb = qRgb.tryGet()
            inSpatialData = spatialCalcQueue.tryGet()

            if inRgb is not None:
                cv_frame = inRgb.getCvFrame()
                h, w, _ = cv_frame.shape
                
                frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks and results.pose_world_landmarks:
                    mp_drawing.draw_landmarks(cv_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # We locate the exact midpoint of the shoulders to use as our absolute 3D anchor
                    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    
                    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                        cx = int(((left_shoulder.x + right_shoulder.x) / 2) * w)
                        cy = int(((left_shoulder.y + right_shoulder.y) / 2) * h)
                        
                        # Clamp safely to bounds
                        cx = max(10, min(w - 11, cx))
                        cy = max(10, min(h - 11, cy))
                        
                        # Request the true metric depth of ONLY the core torso
                        cfg = dai.SpatialLocationCalculatorConfig()
                        configData = dai.SpatialLocationCalculatorConfigData()
                        configData.depthThresholds.lowerThreshold = 100
                        configData.depthThresholds.upperThreshold = 10000
                        configData.roi = dai.Rect(dai.Point2f(cx - 10, cy - 10), dai.Point2f(cx + 10, cy + 10))
                        cfg.addROI(configData)
                        spatialCalcConfigInQueue.send(cfg)
                        
                        cv2.circle(cv_frame, (cx, cy), 8, (255, 0, 0), -1)
                        
                        last_world_landmarks = results.pose_world_landmarks

                cv2.imshow("OAK-D RGB AI (Press Q to quit)", cv_frame)
            
            if inSpatialData is not None and last_world_landmarks is not None:
                spatialData = inSpatialData.getSpatialLocations()
                if len(spatialData) > 0:
                    # Our Absolute Root is the midpoint of the shoulders in true millimeters
                    root_x = spatialData[0].spatialCoordinates.x
                    root_y = spatialData[0].spatialCoordinates.y
                    root_z = spatialData[0].spatialCoordinates.z
                    
                    if root_z > 0:
                        last_root_abs = (root_x, root_y, root_z)
                        
            # Actually plot the data if we have it
            if last_world_landmarks is not None and last_root_abs is not None:
                ax.clear()
                
                # Retrieve the true anchor point
                root_x, root_y, root_z = last_root_abs
                
                # Calculate MediaPipe's internal anchor point (Shoulder midpoint in meters, converted to mm)
                mp_left_shoulder = last_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                mp_right_shoulder = last_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                mp_anchor_x = ((mp_left_shoulder.x + mp_right_shoulder.x) / 2) * 1000
                mp_anchor_y = ((mp_left_shoulder.y + mp_right_shoulder.y) / 2) * 1000
                mp_anchor_z = ((mp_left_shoulder.z + mp_right_shoulder.z) / 2) * 1000
                
                # How much we must shift MediaPipe's skeleton to align perfectly with the OAK-D's absolute reality
                # OAK-D Axis: X is right, Y is down, Z is forward
                # MP Axis: X is right, Y is down, Z is forward (depth)
                shift_x = root_x - mp_anchor_x
                shift_y = root_y - mp_anchor_y
                shift_z = root_z - mp_anchor_z
                
                # Render all 33 bones
                connections = mp_pose.POSE_CONNECTIONS
                landmarks = last_world_landmarks.landmark
                
                for connection in connections:
                    p1 = landmarks[connection[0]]
                    p2 = landmarks[connection[1]]
                    
                    # Apply shift and Negate Y for Matplotlib so it points "UP"
                    x1 = (p1.x * 1000) + shift_x
                    y1 = (p1.z * 1000) + shift_z  # Z depth becomes Matplotlib Y axis
                    z1 = -((p1.y * 1000) + shift_y) # Y height becomes Matplotlib Z axis
                    
                    x2 = (p2.x * 1000) + shift_x
                    y2 = (p2.z * 1000) + shift_z
                    z2 = -((p2.y * 1000) + shift_y)
                    
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color='blue', linewidth=2)
                
                rx = [(l.x * 1000) + shift_x for l in landmarks]
                ry = [(l.z * 1000) + shift_z for l in landmarks]
                rz = [-((l.y * 1000) + shift_y) for l in landmarks]
                ax.scatter(rx, ry, rz, color='red', s=15)
                
                # Box boundaries locked to root so it doesn't warp
                ax.set_xlim([root_x - 1000, root_x + 1000])
                ax.set_ylim([root_z - 1000, root_z + 1000])
                ax.set_zlim([-root_y - 1000, -root_y + 1000])
                
                ax.set_xlabel('Left / Right (mm)')
                ax.set_ylabel('Depth from Camera (mm)')
                ax.set_zlabel('Height (mm)')
                ax.set_title('Live 3D Full Skeleton Tracker')
                
                # Make view a bit clearer
                ax.view_init(elev=10, azim=-60)
            
            # Flush GUI events without blocking
            fig.canvas.draw_idle()   # draw_idle is better for non-blocking
            fig.canvas.flush_events()
            plt.pause(0.001)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()
