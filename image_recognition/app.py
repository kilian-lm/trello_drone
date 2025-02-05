#!/usr/bin/env python3
"""
This script connects to a Tello drone, takes off, and scans the environment for retail store products
using the Google Cloud Vision API for object localization with explicit service account authentication.
It also gracefully handles the case when the drone does not produce video data.
"""

import cv2
import time
import pandas as pd
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
SERVICE_ACCOUNT_FILE = '/Users/d0342084/Documents/Git/trello_drone/image_recognition/some_unwanted_creatre.json'  # <-- Update this path!

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
    """
    try:
        # Ensure frame is valid
        if frame is None:
            logging.error("No frame provided to run_detection.")
            return pd.DataFrame()

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

        return pd.DataFrame(detections_list)

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

        try:
            frame = tello.get_frame_read().frame
        except Exception as ex:
            logging.error(f"Error obtaining frame from drone: {ex}")
            return

        if frame is None:
            logging.error("No frame captured during scanning.")
            return

        detections = run_detection(frame)
        for idx, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            # Check for keywords of interest.
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

    # Run a loop (up to 60 seconds) to search for the first product.
    search_timeout = 60  # seconds.
    search_start = time.time()
    logging.info("Searching for initial product detection...")

    while time.time() - search_start < search_timeout:
        try:
            frame = tello.get_frame_read().frame
        except Exception as ex:
            logging.error(f"Error obtaining frame from drone: {ex}")
            continue

        if frame is None:
            logging.warning("Drone did not produce any frame; waiting...")
            time.sleep(1)
            continue

        # Optionally, resize the frame to speed up processing.
        resized_frame = cv2.resize(frame, (640, 480))
        detections = run_detection(resized_frame)

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
                    if first_article_altitude is None:
                        first_article_altitude = current_alt
                        logging.info(f"First product detected at altitude {first_article_altitude} cm.")
                    log_detection(time.time(), current_alt, "center", label, conf, bbox)
                    scanning_state = "scanning"
                    break
        if scanning_state == "scanning":
            break

        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User requested exit during search.")
            break

    if first_article_altitude is None:
        logging.warning("No initial product detected during search.")

    if scanning_state == "scanning":
        logging.info("Initiating scanning pattern: up, down, right, left.")
        scan_direction(tello, "up", 30)
        scan_direction(tello, "down", 60)
        scan_direction(tello, "up", 30)
        scan_direction(tello, "right", 30)
        scan_direction(tello, "left", 60)
        scan_direction(tello, "right", 30)
    else:
        logging.info("No scanning performed since no product was detected initially.")

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

    logging.info("Writing detection logs to detections.csv...")
    try:
        df = pd.DataFrame(detections_log)
        df.to_csv("detections.csv", index=False)
        logging.info("Detection log saved.")
    except Exception as e:
        logging.error(f"Error writing CSV file: {e}")

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
