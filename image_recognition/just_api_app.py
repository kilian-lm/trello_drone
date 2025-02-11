#!/usr/bin/env python3
import cv2
import time
import logging
import sys

from djitellopy import Tello
from google.cloud import vision
from google.oauth2 import service_account

# --------------------------------------------------------------------
# USER CONFIG - UPDATE THESE
# --------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "/Users/d0342084/Documents/Git/trello_drone/image_recognition/some_unwanted_creatre.json"  # <-- Update this path!
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LOGGING_LEVEL = logging.DEBUG  # DEBUG prints everything
WAIT_BETWEEN_FRAMES = 0  # Seconds to wait after each frame (0 = no extra delay)

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------
# Setup Vision Client
# --------------------------------------------------------------------
try:
    logging.info("Loading service account credentials...")
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    logging.info("Vision API client initialized.")
except Exception as e:
    logging.error(f"Could not init Vision API: {e}")
    sys.exit(1)

# --------------------------------------------------------------------
# Object Localization Function
# --------------------------------------------------------------------
def run_object_localization(frame):
    """
    Sends the frame to Cloud Vision for object localization, returning
    a list of { 'name': <str>, 'score': <float>, 'box': (xmin, ymin, xmax, ymax) }.
    """
    if frame is None:
        logging.debug("run_object_localization called with None frame!")
        return []

    # Encode frame
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Failed to encode frame to JPEG.")
        return []

    image = vision.Image(content=encoded.tobytes())

    # Call Vision API
    try:
        response = vision_client.object_localization(image=image)
        if response.error.message:
            logging.error(f"Vision API returned an error: {response.error.message}")
            return []
    except Exception as ex:
        logging.error(f"Error calling Vision API: {ex}")
        return []

    # Parse results
    h, w = frame.shape[:2]
    detections = []
    for obj in response.localized_object_annotations:
        xs = [v.x * w for v in obj.bounding_poly.normalized_vertices]
        ys = [v.y * h for v in obj.bounding_poly.normalized_vertices]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))
        detections.append({
            "name": obj.name,
            "score": obj.score,
            "box": (xmin, ymin, xmax, ymax)
        })
    return detections

# --------------------------------------------------------------------
# Main Debug Loop
# --------------------------------------------------------------------
def main():
    # Force creation of an OpenCV window upfront
    # so we can confirm if a GUI can even open
    window_name = "Tello GCP Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # or WINDOW_AUTOSIZE
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    logging.info("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    logging.info(f"Battery: {tello.get_battery()}%")

    logging.info("Starting Tello camera stream...")
    tello.streamon()

    # Give Tello a moment to start streaming
    time.sleep(2)

    frame_read = tello.get_frame_read()
    if not frame_read:
        logging.error("Could not get_frame_read() from Tello.")
        return

    frame_counter = 0

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                logging.debug("No frame from Tello (frame is None). Sleeping 0.1s")
                time.sleep(0.1)
                continue

            frame_counter += 1
            logging.debug(f"Got frame #{frame_counter} from Tello. Shape: {frame.shape}")

            # Resize for our display window
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Send to GCP
            logging.debug("Sending frame to GCP Vision...")
            detections = run_object_localization(display_frame)
            logging.debug(f"Detected {len(detections)} objects this frame.")

            # Draw bounding boxes
            for det in detections:
                (xmin, ymin, xmax, ymax) = det["box"]
                name = det["name"]
                score = det["score"]
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                txt = f"{name} {score:.2f}"
                cv2.putText(display_frame, txt, (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Show the frame in a window
            cv2.imshow(window_name, display_frame)
            logging.debug("cv2.imshow called. Press 'q' to quit.")

            # Wait a little
            if WAIT_BETWEEN_FRAMES > 0:
                time.sleep(WAIT_BETWEEN_FRAMES)

            # Check for 'q' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("User pressed 'q'. Exiting loop.")
                break

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt. Exiting main loop.")
    finally:
        logging.info("Cleaning up...")
        tello.streamoff()
        cv2.destroyAllWindows()
        logging.info("Done.")

if __name__ == "__main__":
    main()
