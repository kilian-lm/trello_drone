import orbslam2
import cv2
import numpy as np
from djitellopy import Tello
import time

# Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()
time.sleep(2)  # Allow the camera to warm up

# Initialize ORB-SLAM2
vocab_path = 'path_to_vocabulary/ORBvoc.txt'
config_path = 'path_to_config_file/config.yaml'
slam_system = orbslam2.System(vocab_path, config_path, orbslam2.Sensor.MONOCULAR)
slam_system.initialize()

try:
    while True:
        # Capture frame from Tello
        frame = tello.get_frame_read().frame
        if frame is None:
            continue

        # Convert frame to grayscale (ORB-SLAM2 expects grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to numpy array with correct type
        image = np.array(gray_frame, dtype=np.uint8)

        # Pass the image to ORB-SLAM2
        pose = slam_system.process_image_mono(image, time.time())

        if pose is not None:
            # Extract pose information
            print("Current Pose:\n", pose)
        else:
            print("Tracking lost...")

        # Display the frame (optional)
        cv2.imshow('Tello Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Shutdown procedures
    slam_system.shutdown()
    tello.streamoff()
    cv2.destroyAllWindows()