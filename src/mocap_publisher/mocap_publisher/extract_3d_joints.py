import cv2
import depthai as dai
import mediapipe as mp
import numpy as np

def main():
    print("Initializing OAK-D Pro Pipeline with MediaPipe...")
    
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

    xoutRgb.setStreamName("rgb")
    xoutDepth.setStreamName("depth")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A) # RGB camera is CAM_A on OAK-D Pro
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    camRgb.setInterleaved(False)
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B) # Left infrared
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C) # Right infrared

    # Create a node that will produce the depth map
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)
    depth.setDepthAlign(dai.CameraBoardSocket.CAM_A) # ALIGN point of view to the color camera

    # Config Spatial Location Calculator (Translates 2D pixels to true 3D mm)
    spatialLocationCalculator.inputConfig.setWaitForMessage(False)

    # Linking
    camRgb.video.link(xoutRgb.input)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.disparity.link(xoutDepth.input)
    
    depth.depth.link(spatialLocationCalculator.inputDepth)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    # Initialize MediaPipe Pose 
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    print("Connecting to OAK-D Device...")
    
    try:
        with dai.Device(pipeline) as device:
            print("Connected successfully! Starting streams...")

            # Output queues
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
            spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

            while True:
                inRgb = qRgb.tryGet()
                inDepth = qDepth.tryGet()
                inSpatialData = spatialCalcQueue.tryGet()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                    h, w, _ = frame.shape
                    
                    # Convert to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        landmarks = results.pose_landmarks.landmark
                        
                        # We want Right Arm for mapping to 1 Robotic Arm
                        # MediaPipe IDs: Right Shoulder (12), Right Elbow (14), Right Wrist (16)
                        tracking_points = {
                            "shoulder": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                            "elbow": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                            "wrist": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        }

                        # Find average confidence
                        if all([pt.visibility > 0.6 for pt in tracking_points.values()]):
                            
                            # Create ROIs (Region of Interest) on the depth map to extract true metric distance 
                            cfg = dai.SpatialLocationCalculatorConfig()
                            roi_size = 10 # 10x10 pixel bounding box around the joint to average depth
                            
                            # We need to use the actual depth map dimensions if they differ from RGB
                            # But since we aligned them with setDepthAlign, they should be identical. We'll be safe.
                            depth_w = 1280 if inDepth is None else inDepth.getWidth()
                            depth_h = 800 if inDepth is None else inDepth.getHeight()
                            
                            for name, pt in tracking_points.items():
                                px_x = int(pt.x * depth_w)
                                px_y = int(pt.y * depth_h)
                                
                                # Clamp to depth image bounds safely
                                px_x = max(roi_size, min(depth_w - roi_size - 1, px_x))
                                px_y = max(roi_size, min(depth_h - roi_size - 1, px_y))

                                configData = dai.SpatialLocationCalculatorConfigData()
                                configData.depthThresholds.lowerThreshold = 100
                                configData.depthThresholds.upperThreshold = 10000
                                # Setting ROI
                                configData.roi = dai.Rect(dai.Point2f(px_x - roi_size, px_y - roi_size), 
                                                        dai.Point2f(px_x + roi_size, px_y + roi_size))
                                cfg.addROI(configData)
                                
                                # Draw pixel circle on the visible RGB frame
                                rgb_x = int(pt.x * w)
                                rgb_y = int(pt.y * h)
                                cv2.circle(frame, (rgb_x, rgb_y), 5, (0, 255, 0), -1)

                            # Send config to calculate 3D coordinates
                            spatialCalcConfigInQueue.send(cfg)
                            
                        else:
                            cv2.putText(frame, "Arm not fully visible", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if inSpatialData is not None:
                        spatialData = inSpatialData.getSpatialLocations()
                        # We sent 3 ROIs (Shoulder, Elbow, Wrist) in exactly that order
                        if len(spatialData) == 3:
                            coord_str = ""
                            joint_names = ["Shoulder", "Elbow", "Wrist"]
                            
                            for i, depthData in enumerate(spatialData):
                                coords = depthData.spatialCoordinates
                                # OAK-D spatialCoordinates are in Millimeters (X, Y, Z)
                                # Note: OAK-D coordinate system: X is right, Y is down, Z is forward
                                x_mm, y_mm, z_mm = coords.x, coords.y, coords.z
                                
                                if z_mm > 0.0:  # Z=0 means depth not found
                                    coord_str += f"{joint_names[i]}: ({int(x_mm)}, {int(y_mm)}, {int(z_mm)})  "
                            
                            if coord_str:
                                # Print straight to terminal natively
                                print(f"Raw 3D Coordinates (mm) -> {coord_str}", end='\r')

                    cv2.imshow("OAK-D AI Pose Tracker", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
    except Exception as e:
        print(f"\nPipeline Error: {e}")

if __name__ == '__main__':
    main()
