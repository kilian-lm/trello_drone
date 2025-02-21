#!/usr/bin/env python3
"""
Autonomous Drone Scanning & Mapping with DJI Tello
- Demonstrates a conceptual integration of:
  1) ORB-SLAM2 for mapping/localization
  2) GCP Vision for object + label + OCR detection
  3) Simple scanning behavior: the drone takes off, rotates slowly to find "products".
"""

import cv2
import time
import logging
import sys

from djitellopy import Tello
from google.cloud import vision
from google.oauth2 import service_account

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "/Users/kilianlehn/Documents/GitHub/trello_drone/image_recognition/enter-universes-fcf7ca441146.json"  # <-- Update
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LOGGING_LEVEL = logging.DEBUG
WAIT_BETWEEN_FRAMES = 0  # small delay after each frame if desired
SHOW_TOP_LABELS = 3       # how many labels to overlay
ROTATION_SPEED = 20       # how fast we rotate in degrees/sec
ROTATION_STEP = 10        # rotate by 10 degrees each iteration
MAX_ROTATION = 360        # total rotation around axis

# If you had actual product keywords you want to match, e.g. "sebamed":
PRODUCT_KEYWORDS = ["sebamed", "nivea", "shampoo", "bottle", "lotion", "cream", "soap"]

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------
# Initialize Vision Client
# --------------------------------------------------------------------
try:
    logging.info("Loading service account credentials...")
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    logging.info("Vision API client initialized.")
except Exception as e:
    logging.error(f"Could not init Vision API client: {e}")
    sys.exit(1)

# --------------------------------------------------------------------
# Placeholder ORB-SLAM2 Integration
# --------------------------------------------------------------------
class ORBSLAMSystem:
    """
    This is a conceptual placeholder for an ORB-SLAM2 integration.
    In reality, you'd:
      - Launch a separate ORB-SLAM2 process in C++ or ROS
      - Feed frames there (via a shared memory or network or python bindings)
      - Retrieve camera pose / map info in real-time
    """
    def __init__(self):
        self.initialized = False

    def start_system(self):
        """
        Start the ORB-SLAM2 process, load vocabulary, etc.
        (Placeholder, real code might be OS-specific)
        """
        logging.info("Starting ORB-SLAM2 system (placeholder).")
        self.initialized = True
        # Real code might do:
        # self.slam = orbslam_py.OrbSlamSystem(VOCAB_PATH, CONFIG_PATH, ...)

    def process_frame(self, frame):
        """
        Send a frame to the SLAM system for tracking.
        Returns True if tracking success, else False
        (Placeholder logic here).
        """
        if not self.initialized:
            return False
        # real code might do:
        # pose = self.slam.TrackMonocular(frame, time_stamp)
        # return (pose is not None)
        return True

    def get_current_pose(self):
        """
        Return the current camera pose from SLAM.
        (Placeholder, real code might return a 4x4 matrix or R,t.)
        """
        if not self.initialized:
            return None
        # example placeholder
        return "POSE_DATA"

orb_slam = ORBSLAMSystem()

# --------------------------------------------------------------------
# Combined detection
# --------------------------------------------------------------------
def run_object_label_text_detection(frame):
    """
    Calls:
      1) Object Localization
      2) Label Detection
      3) Text (OCR)
    Returns dictionary with "objects", "labels", "texts".
    """
    if frame is None:
        logging.debug("Frame is None - skipping detection.")
        return {"objects": [], "labels": [], "texts": []}

    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Failed to encode frame.")
        return {"objects": [], "labels": [], "texts": []}

    image = vision.Image(content=encoded.tobytes())
    height, width = frame.shape[:2]

    results = {"objects": [], "labels": [], "texts": []}

    # Object Localization
    try:
        obj_resp = vision_client.object_localization(image=image)
        if obj_resp.error.message:
            logging.error(f"Object localization error: {obj_resp.error.message}")
        else:
            for obj in obj_resp.localized_object_annotations:
                xs = [v.x * width for v in obj.bounding_poly.normalized_vertices]
                ys = [v.y * height for v in obj.bounding_poly.normalized_vertices]
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))
                results["objects"].append({
                    "name": obj.name,
                    "score": obj.score,
                    "box": (xmin, ymin, xmax, ymax)
                })
    except Exception as ex:
        logging.error(f"Exception calling object_localization: {ex}")

    # Label Detection
    try:
        label_resp = vision_client.label_detection(image=image)
        if label_resp.error.message:
            logging.error(f"Label detection error: {label_resp.error.message}")
        else:
            for lbl in label_resp.label_annotations:
                results["labels"].append({
                    "description": lbl.description,
                    "score": lbl.score
                })
    except Exception as ex:
        logging.error(f"Exception calling label_detection: {ex}")

    # Text (OCR)
    try:
        text_resp = vision_client.text_detection(image=image)
        if text_resp.error.message:
            logging.error(f"Text detection error: {text_resp.error.message}")
        else:
            for i, txt in enumerate(text_resp.text_annotations):
                poly_verts = txt.bounding_poly.vertices
                corners = []
                for v in poly_verts:
                    x_val = int(v.x) if v.x else 0
                    y_val = int(v.y) if v.y else 0
                    corners.append((x_val, y_val))
                results["texts"].append({
                    "text": txt.description,
                    "box": corners
                })
    except Exception as ex:
        logging.error(f"Exception calling text_detection: {ex}")

    return results

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    window_name = "Tello + ORB-SLAM2 + GCP"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    # 1) Start ORB-SLAM2 (placeholder)
    orb_slam.start_system()

    # 2) Connect to Tello
    logging.info("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    logging.info(f"Battery: {tello.get_battery()}%")

    # 3) Take off
    logging.info("Taking off...")
    tello.takeoff()
    time.sleep(2)

    # 4) Start camera stream
    logging.info("Starting Tello camera stream...")
    tello.streamon()
    time.sleep(2)

    frame_read = tello.get_frame_read()
    if not frame_read:
        logging.error("Could not get_frame_read() from Tello.")
        return

    # 5) Autonomous scanning approach
    # We'll do a slow 360° rotation. On each step:
    #  - Capture frame
    #  - Pass to ORB-SLAM2 for mapping
    #  - Send to GCP for detection
    #  - If something looks like a product, check text
    # This is a simple example; you might refine or loop until you find "sebamed" etc.

    total_rotated = 0
    scanning = True

    try:
        while scanning:
            frame = frame_read.frame
            if frame is None:
                logging.debug("No frame from Tello; sleeping 0.1s")
                time.sleep(0.1)
                continue

            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Send frame to ORB-SLAM2 for mapping
            slam_success = orb_slam.process_frame(display_frame)
            if slam_success:
                pose = orb_slam.get_current_pose()
                logging.debug(f"ORB-SLAM2 pose: {pose}")

            # Send frame to GCP Vision
            detection_results = run_object_label_text_detection(display_frame)
            objects = detection_results["objects"]
            labels = detection_results["labels"]
            texts = detection_results["texts"]

            # Draw bounding boxes / overlay info
            for obj in objects:
                (xmin, ymin, xmax, ymax) = obj["box"]
                name = obj["name"]
                score = obj["score"]
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(display_frame, f"{name} {score:.2f}", (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Show top few labels
            offset_y = 20
            for lbl in labels[:SHOW_TOP_LABELS]:
                desc = lbl["description"]
                score = lbl["score"]
                cv2.putText(display_frame, f"{desc} {score:.2f}", (10, offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                offset_y += 25

            # Text bounding boxes in magenta
            for t in texts:
                corners = t["box"]
                recognized_text = t["text"].strip()
                if len(corners) == 4:
                    for j in range(4):
                        pt1 = corners[j]
                        pt2 = corners[(j+1) % 4]
                        cv2.line(display_frame, pt1, pt2, (255,0,255), 2)
                    # put text near top-left
                    cv2.putText(display_frame, recognized_text, (corners[0][0], corners[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            # Show in a window
            cv2.imshow(window_name, display_frame)
            if WAIT_BETWEEN_FRAMES > 0:
                time.sleep(WAIT_BETWEEN_FRAMES)

            # Check if we see "sebamed" or something that might be a product
            # We'll search in both labels + texts for these keywords
            found_product = False
            search_area = [lbl["description"].lower() for lbl in labels]
            search_area += [t["text"].lower() for t in texts]
            if any(any(k in val for k in PRODUCT_KEYWORDS) for val in search_area):
                found_product = True

            if found_product:
                logging.info("Product spotted (matching keywords)! Stopping rotation.")
                # optional: you could command Tello to stop rotating,
                # move closer, or center the bounding box, etc.
                scanning = False
            else:
                # Keep rotating if we haven't completed 360
                if total_rotated < MAX_ROTATION:
                    logging.debug(f"Rotating yaw by {ROTATION_STEP} degrees.")
                    tello.rotate_clockwise(ROTATION_STEP)
                    total_rotated += ROTATION_STEP
                else:
                    logging.info("Finished 360 rotation, no product found.")
                    scanning = False

            # Check for 'q' key to break
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("User pressed 'q'. Exiting scanning.")
                scanning = False

        # After scanning is done, land
        logging.info("Landing now...")
        tello.land()

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt, landing.")
        tello.land()
    finally:
        logging.info("Cleaning up...")
        tello.streamoff()
        cv2.destroyAllWindows()
        logging.info("Done.")

if __name__ == "__main__":
    main()