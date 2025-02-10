"""
Script that connects to a Tello drone, takes off, and scans for *any* recognized objects, labels, 
or logos via the Google Cloud Vision API, using a service account for authentication.

All recognized items are logged with bounding boxes (when available) and confidence scores.
"""

import cv2
import time
import pandas as pd
import logging

from djitellopy import Tello

# Import Google Cloud Vision client library and service account credentials
from google.cloud import vision
from google.oauth2 import service_account

# ----------------------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------------------
# Set level to DEBUG to see all details (including detection details).
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

# ----------------------------------------------------------------------
# Initialize Google Cloud Vision Client with Service Account Credentials
# ----------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "/Users/d0342084/Documents/Git/trello_drone/image_recognition/some_unwanted_creatre.json"  # <-- Update this path!

try:
    logging.info("Loading service account credentials for Google Cloud Vision API...")
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    logging.info("Google Cloud Vision API client initialized with service account credentials.")
except Exception as e:
    logging.error(f"Failed to initialize Cloud Vision API client: {e}")
    exit(1)

# ----------------------------------------------------------------------
# Global Data Storage for Detections
# ----------------------------------------------------------------------
# Each detection entry will include:
# timestamp, altitude (cm), scan direction, label, confidence, bbox.
detections_log = []

def log_detection(timestamp, altitude, direction, label, confidence, bbox):
    """
    Append a detection entry to the global detections_log list.
    bbox is a tuple (xmin, ymin, xmax, ymax); can be None or partially None if not provided by the API.
    """
    entry = {
        "timestamp": timestamp,
        "altitude_cm": altitude,
        "direction": direction,
        "label": label,
        "confidence": confidence,
        "bbox": bbox
    }
    detections_log.append(entry)
    logging.info(f"Detection logged: {entry}")


# ----------------------------------------------------------------------
# Helper: Run All Vision Detections on a Frame (Object + Label + Logo)
# ----------------------------------------------------------------------
def run_detection(frame):
    """
    Calls Object Localization, Label Detection, *and* Logo Detection on the given frame.
    Returns a pandas DataFrame with columns:
      - 'name' (the detected object/label/logo)
      - 'confidence' (float)
      - 'xmin', 'ymin', 'xmax', 'ymax' (bounding box in pixel coords, or None if not applicable)
      - 'type' in ['object', 'label', 'logo']
    """
    if frame is None:
        logging.error("No frame provided to run_detection.")
        return pd.DataFrame()

    try:
        # Encode the frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            logging.error("Failed to encode frame as JPEG.")
            return pd.DataFrame()

        content = buffer.tobytes()
        image = vision.Image(content=content)

        # 1) Object Localization
        obj_response = vision_client.object_localization(image=image)
        if obj_response.error.message:
            logging.error(f"Vision API (object localization) error: {obj_response.error.message}")
            return pd.DataFrame()

        # 2) Label Detection
        label_response = vision_client.label_detection(image=image)
        if label_response.error.message:
            logging.error(f"Vision API (label detection) error: {label_response.error.message}")
            return pd.DataFrame()

        # 3) Logo Detection
        logo_response = vision_client.logo_detection(image=image)
        if logo_response.error.message:
            logging.error(f"Vision API (logo detection) error: {logo_response.error.message}")
            return pd.DataFrame()

        detections_list = []
        height, width = frame.shape[:2]

        # ------------------------------------------------
        # Parse Object Localization results
        # ------------------------------------------------
        logging.debug("=== Object Localization Annotations ===")
        for annotation in obj_response.localized_object_annotations:
            logging.debug(f"Object: {annotation.name}, Score: {annotation.score}")
            # Convert normalized bounding box to pixel coords
            xs = [v.x * width for v in annotation.bounding_poly.normalized_vertices]
            ys = [v.y * height for v in annotation.bounding_poly.normalized_vertices]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            detections_list.append({
                "name": annotation.name,
                "confidence": annotation.score,
                "xmin": int(xmin),
                "ymin": int(ymin),
                "xmax": int(xmax),
                "ymax": int(ymax),
                "type": "object"
            })

        # ------------------------------------------------
        # Parse Label Detection results
        # ------------------------------------------------
        logging.debug("=== Label Detection Annotations ===")
        for lbl in label_response.label_annotations:
            logging.debug(f"Label: {lbl.description}, Score: {lbl.score}")
            # Label detection doesn't provide bounding boxes
            detections_list.append({
                "name": lbl.description,
                "confidence": lbl.score,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None,
                "type": "label"
            })

        # ------------------------------------------------
        # Parse Logo Detection results
        # ------------------------------------------------
        logging.debug("=== Logo Detection Annotations ===")
        for logo in logo_response.logo_annotations:
            logging.debug(f"Logo: {logo.description}, Score: {logo.score}")
            # Logo detection bounding box is in bounding_poly.vertices
            vertices = logo.bounding_poly.vertices
            if len(vertices) == 4:
                xs = [v.x for v in vertices if v.x is not None]
                ys = [v.y for v in vertices if v.y is not None]
                if xs and ys:
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                else:
                    xmin = xmax = ymin = ymax = None
            else:
                xmin = xmax = ymin = ymax = None

            detections_list.append({
                "name": logo.description,
                "confidence": logo.score,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "type": "logo"
            })

        return pd.DataFrame(detections_list)

    except Exception as e:
        logging.error(f"Error in run_detection: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Helper: Scan in a Given Direction
# ----------------------------------------------------------------------
def scan_direction(tello, direction, distance_cm, scan_pause=2, min_conf=0.5):
    """
    Move the drone in the given direction by distance_cm, pause, capture a frame,
    run detection, and log all detections above the confidence threshold.
    """
    try:
        # Move in specified direction
        if direction == "up":
            logging.info(f"Moving up {distance_cm} cm")
            tello.move_up(distance_cm)
        elif direction == "down":
            logging.info(f"Moving down {distance_cm} cm")
            tello.move_down(distance_cm)
        elif direction == "left":
            logging.info(f"Moving left {distance_cm} cm")
            tello.move_left(distance_cm)
        elif direction == "right":
            logging.info(f"Moving right {distance_cm} cm")
            tello.move_right(distance_cm)
        else:
            logging.warning(f"Unknown direction command: {direction}")
            return

        # Pause briefly for stability
        time.sleep(scan_pause)

        # Get a frame from the drone
        frame = tello.get_frame_read().frame
        if frame is None:
            logging.error("No frame captured during scanning.")
            return

        # Run detection (objects, labels, logos)
        detections = run_detection(frame)
        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            # Log everything above the threshold
            if conf >= min_conf:
                current_alt = tello.get_height()
                log_detection(time.time(), current_alt, direction, label, conf, bbox)

    except Exception as e:
        logging.error(f"Error during scan in direction '{direction}': {e}")


# ----------------------------------------------------------------------
# Main Flight & Detection Routine
# ----------------------------------------------------------------------
def main():
    # Initialize and connect to the Tello drone
    tello = Tello()
    logging.info("Connecting to Tello drone...")
    try:
        tello.connect()
        battery = tello.get_battery()
        logging.info(f"Drone battery level: {battery}%")
    except Exception as e:
        logging.error(f"Error connecting to Tello: {e}")
        return

    # Start video stream
    tello.streamon()
    time.sleep(2)  # Allow camera stream to initialize

    # Takeoff
    logging.info("Taking off...")
    try:
        tello.takeoff()
    except Exception as e:
        logging.error(f"Takeoff failed: {e}")
        return

    # Variables to record first detection altitude
    first_article_altitude = None
    scanning_state = "searching"

    # Search for up to 60 seconds for a detection
    search_timeout = 60
    search_start = time.time()
    logging.info("Searching for initial product detection...")

    while time.time() - search_start < search_timeout:
        frame = None
        try:
            frame = tello.get_frame_read().frame
        except Exception as ex:
            logging.error(f"Error obtaining frame from drone: {ex}")
            time.sleep(1)
            continue

        if frame is None:
            logging.warning("Drone did not produce any frame; waiting...")
            time.sleep(1)
            continue

        # Optionally resize frame so it's not huge, but keep it large enough for clarity
        resized_frame = cv2.resize(frame, (640, 480))

        # Run detection on current frame
        detections = run_detection(resized_frame)

        # If we have anything recognized, log it & mark scanning
        if not detections.empty:
            # (You could add a confidence threshold here if desired)
            for _, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                current_alt = tello.get_height()
                log_detection(time.time(), current_alt, "center", label, conf, bbox)

            if first_article_altitude is None:
                first_article_altitude = tello.get_height()
                logging.info(f"First detection found at altitude {first_article_altitude} cm.")
            scanning_state = "scanning"
            break

        # Show debug feed
        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User requested exit during search.")
            break

    if first_article_altitude is None:
        logging.warning("No product or item detected during the search phase.")

    # If something was detected, run a scanning pattern
    if scanning_state == "scanning":
        logging.info("Initiating scanning pattern: up, down, right, left.")
        scan_direction(tello, "up", 30)
        scan_direction(tello, "down", 60)
        scan_direction(tello, "up", 30)
        scan_direction(tello, "right", 30)
        scan_direction(tello, "left", 60)
        scan_direction(tello, "right", 30)
    else:
        logging.info("No scanning pattern performed (nothing detected initially).")

    # Hover a few seconds more to check for additional detections
    logging.info("Monitoring for additional detections before landing...")
    hover_end = time.time() + 5
    while time.time() < hover_end:
        try:
            frame = tello.get_frame_read().frame
        except Exception as ex:
            logging.error(f"Error obtaining frame during hover: {ex}")
            break

        if frame is None:
            logging.warning("No frame produced during hover.")
            time.sleep(1)
            continue

        resized_frame = cv2.resize(frame, (640, 480))
        detections = run_detection(resized_frame)
        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

            # Log everything above some minimal confidence
            if conf >= 0.5:
                current_alt = tello.get_height()
                log_detection(time.time(), current_alt, "hover", label, conf, bbox)

        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write all detections to CSV
    logging.info("Writing detection logs to detections.csv...")
    try:
        df = pd.DataFrame(detections_log)
        df.to_csv("detections.csv", index=False)
        logging.info("Detection log saved to detections.csv")
    except Exception as e:
        logging.error(f"Error writing CSV file: {e}")

    # Land and cleanup
    logging.info("Landing...")
    try:
        tello.land()
    except Exception as e:
        logging.error(f"Error during landing: {e}")

    tello.streamoff()
    cv2.destroyAllWindows()
    logging.info("Mission complete.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Landing and exiting.")
