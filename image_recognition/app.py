#!/usr/bin/env python3
"""
This script connects to a Tello drone, takes off, and scans the environment for retail store products
(such as shower gels or products from German Drogeriemärkte like Müller, DM, Rossmann) using the 
Google Cloud Vision API for object localization with explicit service account authentication.

The drone will:
  - Take off and search for the first product.
  - Record the altitude when the first matching product is detected.
  - Move in a scanning pattern (up, down, right, left) and capture additional detections.
  - Log all detections (timestamp, altitude, scan direction, detected label, confidence score, and bounding box)
    and write them to a CSV file.
  - Land after the mission.

Requirements:
  - djitellopy (pip install djitellopy)
  - opencv-python (pip install opencv-python)
  - pandas (pip install pandas)
  - google-cloud-vision (pip install google-cloud-vision)
  - google-auth (pip install google-auth)

Make sure to update SERVICE_ACCOUNT_FILE to point to your service account JSON key file.
"""

import cv2
import time
import pandas as pd
import numpy as np
import logging

from djitellopy import Tello

# Import the Google Cloud Vision client library and service account credentials.
from google.cloud import vision
from google.oauth2 import service_account

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

# ----------------------------
# Initialize Google Cloud Vision Client with Service Account Credentials
# ----------------------------
SERVICE_ACCOUNT_FILE = '/Users/d0342084/Documents/Git/trello_drone/image_recognition/stovi-infrastructure-73c3e5175-c427735bae33.json'  # <-- Update this path!

try:
    logging.info("Loading service account credentials for Google Cloud Vision API...")
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    logging.info("Google Cloud Vision API client initialized with service account credentials.")
except Exception as e:
    logging.error(f"Failed to initialize Cloud Vision API client with service account: {e}")
    exit(1)

# ----------------------------
# Global Data Storage for Detections
# ----------------------------
# Each detection entry will include: timestamp, altitude (cm), scan direction, label, confidence, bbox.
detections_log = []


def log_detection(timestamp, altitude, direction, label, confidence, bbox):
    """Append a detection entry to the global log."""
    entry = {
        "timestamp": timestamp,
        "altitude_cm": altitude,
        "direction": direction,
        "label": label,
        "confidence": confidence,
        "bbox": bbox  # (xmin, ymin, xmax, ymax)
    }
    detections_log.append(entry)
    logging.info(f"Detection logged: {entry}")


# ----------------------------
# Helper: Run Detection on a Frame via GCP Cloud Vision API
# ----------------------------
def run_detection(frame):
    """
    Send the provided frame to the Cloud Vision API's object localization endpoint and return the detections
    as a pandas DataFrame with columns: 'name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'.

    The bounding boxes are computed from the normalized vertices returned by the API.
    """
    try:
        # Encode the frame as JPEG.
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Failed to encode frame as JPEG.")
            return pd.DataFrame()
        content = buffer.tobytes()

        # Prepare the image for the Vision API.
        image = vision.Image(content=content)

        # Call the object localization method.
        response = vision_client.object_localization(image=image)
        if response.error.message:
            logging.error(f"Vision API Error: {response.error.message}")
            return pd.DataFrame()

        detections_list = []
        height, width = frame.shape[:2]
        for annotation in response.localized_object_annotations:
            # Compute bounding box coordinates from normalized vertices.
            xs = [vertex.x * width for vertex in annotation.bounding_poly.normalized_vertices]
            ys = [vertex.y * height for vertex in annotation.bounding_poly.normalized_vertices]
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
            detections_list.append({
                "name": annotation.name,
                "confidence": annotation.score,
                "xmin": int(xmin),
                "ymin": int(ymin),
                "xmax": int(xmax),
                "ymax": int(ymax)
            })

        df = pd.DataFrame(detections_list)
        return df

    except Exception as e:
        logging.error(f"Error in run_detection: {e}")
        return pd.DataFrame()


# ----------------------------
# Helper: Scan in a Given Direction
# ----------------------------
def scan_direction(tello, direction, distance_cm, scan_pause=2):
    """
    Move the drone in the given direction by the specified distance, pause to capture a frame,
    run object detection on that frame via the Cloud Vision API, and log any detections found.

    Parameters:
      tello      : the Tello drone instance.
      direction  : one of "up", "down", "left", "right".
      distance_cm: distance to move (in centimeters).
      scan_pause : seconds to wait after movement before capturing a frame.
    """
    try:
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
            logging.warning("Unknown direction command.")
            return

        time.sleep(scan_pause)  # Wait for movement to complete and scene stabilization.

        frame = tello.get_frame_read().frame
        if frame is None:
            logging.error("No frame captured during scanning.")
            return

        detections = run_detection(frame)
        # Filter and log detections of interest.
        for idx, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

            # Check if the detection label contains keywords of interest.
            if ("shower" in label.lower() or
                    "müller" in label.lower() or
                    "dm" in label.lower() or
                    "rossmann" in label.lower()):
                current_alt = tello.get_height()
                log_detection(time.time(), current_alt, direction, label, conf, bbox)

    except Exception as e:
        logging.error(f"Error during scan in direction '{direction}': {e}")


# ----------------------------
# Main Flight & Detection Routine
# ----------------------------
def main():
    # Initialize and connect to the Tello drone.
    tello = Tello()
    logging.info("Connecting to Tello drone...")
    try:
        tello.connect()
        battery = tello.get_battery()
        logging.info(f"Drone battery level: {battery}%")
    except Exception as e:
        logging.error(f"Error connecting to Tello: {e}")
        return

    # Start video stream.
    tello.streamon()
    time.sleep(2)  # Allow camera stream to initialize.

    # Takeoff.
    logging.info("Taking off...")
    try:
        tello.takeoff()
    except Exception as e:
        logging.error(f"Takeoff failed: {e}")
        return

    # Variables to record the first detection altitude.
    first_article_altitude = None
    scanning_state = "searching"

    # Run a loop for a set period (e.g., 60 seconds) to search for the first article.
    search_timeout = 60  # seconds.
    search_start = time.time()
    logging.info("Searching for initial product detection...")

    while time.time() - search_start < search_timeout:
        frame = tello.get_frame_read().frame
        if frame is None:
            continue

        # Optionally, resize the frame to speed up processing.
        resized_frame = cv2.resize(frame, (640, 480))
        detections = run_detection(resized_frame)

        # Check each detection – if we see a product of interest.
        if not detections.empty:
            for idx, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                if ("shower" in label.lower() or
                        "müller" in label.lower() or
                        "dm" in label.lower() or
                        "rossmann" in label.lower()):
                    current_alt = tello.get_height()
                    # Record the altitude of the first detected product.
                    if first_article_altitude is None:
                        first_article_altitude = current_alt
                        logging.info(f"First product detected at altitude {first_article_altitude} cm.")
                    # Log the detection (using direction "center" since no movement yet).
                    log_detection(time.time(), current_alt, "center", label, conf, bbox)
                    scanning_state = "scanning"
                    break
        if scanning_state == "scanning":
            break

        # Optionally display the live feed (press 'q' to exit early).
        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User requested exit during search.")
            break

    if first_article_altitude is None:
        logging.warning("No initial product detected during search.")

    # If in scanning state, move in different directions to check for other articles.
    if scanning_state == "scanning":
        logging.info("Initiating scanning pattern: up, down, right, left.")
        # Scan upward.
        scan_direction(tello, "up", 30)
        # Scan downward (from up, so move down more to go below original altitude).
        scan_direction(tello, "down", 60)
        # Return to original altitude.
        scan_direction(tello, "up", 30)
        # Scan right.
        scan_direction(tello, "right", 30)
        # Scan left (to return to center).
        scan_direction(tello, "left", 60)
        # Return to center.
        scan_direction(tello, "right", 30)
    else:
        logging.info("No scanning performed since no product was detected initially.")

    # Optionally continue a brief period of hovering and monitoring (here 5 seconds).
    logging.info("Monitoring for additional detections before landing...")
    hover_end = time.time() + 5
    while time.time() < hover_end:
        frame = tello.get_frame_read().frame
        if frame is not None:
            resized_frame = cv2.resize(frame, (640, 480))
            detections = run_detection(resized_frame)
            for idx, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                if ("shower" in label.lower() or
                        "müller" in label.lower() or
                        "dm" in label.lower() or
                        "rossmann" in label.lower()):
                    log_detection(time.time(), tello.get_height(), "hover", label, conf, bbox)
        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the detection logs to a CSV file.
    logging.info("Writing detection logs to detections.csv...")
    df = pd.DataFrame(detections_log)
    df.to_csv("detections.csv", index=False)
    logging.info("Detection log saved.")

    # Land the drone.
    logging.info("Landing...")
    try:
        tello.land()
    except Exception as e:
        logging.error(f"Error during landing: {e}")

    # Shut down the video stream and close any OpenCV windows.
    tello.streamoff()
    cv2.destroyAllWindows()
    logging.info("Mission complete.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Landing and exiting.")
