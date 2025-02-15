#!/usr/bin/env python3
"""
Example script to connect to a Tello drone, take off, and detect retail store products
using Google Cloud Vision's Label Detection and Logo Detection, with explicit
service account credentials.
"""

import cv2
import time
import pandas as pd
import logging

from djitellopy import Tello

# Google Cloud Vision
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
SERVICE_ACCOUNT_FILE = "/Users/d0342084/Documents/Git/trello_drone/image_recognition/some_unwanted_creatre.json"  # <-- Update this path!

try:
    logging.info("Loading service account credentials for Google Cloud Vision API...")
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    logging.info("Google Cloud Vision API client initialized with service account credentials.")
except Exception as e:
    logging.error(f"Failed to initialize Cloud Vision API client: {e}")
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
        "bbox": bbox  # (xmin, ymin, xmax, ymax) or (None, None, None, None) if not available
    }
    detections_log.append(entry)
    logging.info(f"Detection logged: {entry}")


# ----------------------------
# Helper: Run Label + Logo Detection on a Frame
# ----------------------------
def run_detection(frame):
    """
    Send the provided frame to the Cloud Vision API for both label detection and logo detection.
    Return a pandas DataFrame with columns: 'name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'type'.

    - For label detections, bounding box columns will be None, 'type' will be "label".
    - For logo detections, bounding box is taken from the bounding_poly. 'type' will be "logo".
    """
    if frame is None:
        logging.error("No frame provided to run_detection.")
        return pd.DataFrame()

    try:
        # Encode the frame as JPEG.
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            logging.error("Failed to encode frame as JPEG.")
            return pd.DataFrame()
        content = buffer.tobytes()

        # Prepare the image for the Vision API
        image = vision.Image(content=content)

        # 1) Label Detection
        label_response = vision_client.label_detection(image=image)
        if label_response.error.message:
            logging.error(f"Vision API (label) error: {label_response.error.message}")
            return pd.DataFrame()

        # 2) Logo Detection
        logo_response = vision_client.logo_detection(image=image)
        if logo_response.error.message:
            logging.error(f"Vision API (logo) error: {logo_response.error.message}")
            return pd.DataFrame()

        # Build a combined list of detections
        detections_list = []

        # --- Parse label annotations ---
        for annotation in label_response.label_annotations:
            detections_list.append({
                "name": annotation.description,
                "confidence": annotation.score,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None,
                "type": "label"
            })

        # --- Parse logo annotations (includes bounding polygons) ---
        for logo in logo_response.logo_annotations:
            # Each logo can have a bounding_poly
            vertices = logo.bounding_poly.vertices
            if len(vertices) == 4:
                xs = [v.x for v in vertices if v.x is not None]
                ys = [v.y for v in vertices if v.y is not None]
                if xs and ys:
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                else:
                    xmin = xmax = ymin = ymax = 0
            else:
                # Fallback if bounding_poly is missing or unexpected
                xmin = xmax = ymin = ymax = 0

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


# ----------------------------
# Helper: Scan in a Given Direction
# ----------------------------
def scan_direction(tello, direction, distance_cm, scan_pause=2, min_conf=0.5):
    """
    Move the drone in a given direction by distance_cm, pause, capture a frame,
    run detection, and log *all* detections above min_conf confidence.
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
            logging.warning("Unknown direction command.")
            return

        # Pause briefly for stability
        time.sleep(scan_pause)

        # Get a frame from drone
        frame = tello.get_frame_read().frame
        if frame is None:
            logging.error("No frame captured during scanning.")
            return

        # Run detection (labels + logos)
        detections = run_detection(frame)
        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

            # Log all recognized labels/logos above the confidence threshold
            if conf >= min_conf:
                current_alt = tello.get_height()
                log_detection(time.time(), current_alt, direction, label, conf, bbox)

    except Exception as e:
        logging.error(f"Error during scan in direction '{direction}': {e}")


# ----------------------------
# Main Flight & Detection Routine
# ----------------------------
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

    # Variables to record the first detection altitude
    first_article_altitude = None
    scanning_state = "searching"

    # Search for up to 60 seconds for an initial product detection
    search_timeout = 60
    search_start = time.time()
    logging.info("Searching for initial product detection...")

    while time.time() - search_start < search_timeout:
        # Attempt to retrieve a frame
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

        # Optionally resize the frame (keep at decent size so text/logos are readable)
        resized_frame = cv2.resize(frame, (640, 480))

        # Run detection on the frame
        detections = run_detection(resized_frame)

        if not detections.empty:
            for _, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                # Check if it matches your keywords
                if ("shower" in label.lower() or
                        "shampoo" in label.lower() or
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

        # Show debug stream
        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User requested exit during search.")
            break

    if first_article_altitude is None:
        logging.warning("No initial product detected during the search phase.")

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
        logging.info("No scanning pattern performed (no product initially detected).")

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
            if ("shower" in label.lower() or
                    "shampoo" in label.lower() or
                    "müller" in label.lower() or
                    "dm" in label.lower() or
                    "rossmann" in label.lower()):
                log_detection(time.time(), tello.get_height(), "hover", label, conf, bbox)

        cv2.imshow("Tello Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write detections to CSV
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
