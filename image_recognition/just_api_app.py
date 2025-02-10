#!/usr/bin/env python3

import cv2
import time
import logging

from djitellopy import Tello

# Google Cloud Vision
from google.cloud import vision
from google.oauth2 import service_account

# -----------------------------------------------------
# CONFIG - UPDATE THESE
# -----------------------------------------------------
SERVICE_ACCOUNT_FILE = "/path/to/your_service_account.json"  # <-- REPLACE
LOGGING_LEVEL = logging.INFO  # or logging.DEBUG for more detail

# -----------------------------------------------------
# Setup Logging
# -----------------------------------------------------
logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

# -----------------------------------------------------
# Initialize Google Cloud Vision Client
# -----------------------------------------------------
try:
    logging.info("Loading service account credentials...")
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    logging.info("Vision API client initialized.")
except Exception as e:
    logging.error(f"Failed to initialize Vision API client: {e}")
    exit(1)

def run_object_localization(frame):
    """
    Sends the frame to the Google Cloud Vision API for object localization.
    Returns a list of dicts, each with keys:
      - name: The object name (e.g. 'Bottle')
      - score: Confidence [0..1]
      - box: (xmin, ymin, xmax, ymax) in pixel coordinates
    """
    if frame is None:
        return []

    # 1) Encode as JPEG
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Failed to encode frame to JPEG.")
        return []

    # 2) Prepare Vision image
    content = encoded_image.tobytes()
    image = vision.Image(content=content)

    # 3) Call object localization
    try:
        response = vision_client.object_localization(image=image)
        if response.error.message:
            logging.error(f"Vision API error: {response.error.message}")
            return []
    except Exception as ex:
        logging.error(f"Exception calling Vision API: {ex}")
        return []

    # 4) Parse response
    (height, width) = frame.shape[:2]
    detections = []
    for obj in response.localized_object_annotations:
        # Convert normalized vertices to pixel coords
        xs = [v.x * width for v in obj.bounding_poly.normalized_vertices]
        ys = [v.y * height for v in obj.bounding_poly.normalized_vertices]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        detections.append({
            "name": obj.name,
            "score": obj.score,
            "box": (xmin, ymin, xmax, ymax),
        })

    return detections

def main():
    # 1) Connect to Tello
    logging.info("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    logging.info(f"Battery: {tello.get_battery()}%")

    # 2) Start video stream
    tello.streamon()
    time.sleep(2)  # wait for stream to initialize

    # 3) Main loop
    try:
        while True:
            frame = tello.get_frame_read().frame
            if frame is None:
                logging.warning("No frame received from Tello.")
                time.sleep(0.1)
                continue

            # (Optional) Resize for faster processing or clarity
            display_frame = cv2.resize(frame, (640, 480))

            # Send to GCP
            detections = run_object_localization(display_frame)

            # Draw bounding boxes
            for det in detections:
                label = det["name"]
                score = det["score"]
                (xmin, ymin, xmax, ymax) = det["box"]
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(display_frame, f"{label} {score:.2f}",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("Tello GCP Test", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt, exiting.")
    finally:
        # Cleanup
        tello.streamoff()
        cv2.destroyAllWindows()
        logging.info("Done.")

if __name__ == "__main__":
    main()
