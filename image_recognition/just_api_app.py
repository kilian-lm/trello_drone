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
SERVICE_ACCOUNT_FILE = "/Users/kilianlehn/Documents/GitHub/trello_drone/image_recognition/enter-universes-fcf7ca441146.json"  # <-- Update this path
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
# Combined Object Localization + Label Detection
# --------------------------------------------------------------------
def run_object_localization_and_label(frame):
    """
    Sends the frame to Cloud Vision for:
      1) Object Localization -> bounding boxes with object names (e.g. "Bottle")
      2) Label Detection -> general labels (e.g. "Beer", "Cosmetics", etc.)

    Returns a dict:
      {
        "objects": [
          {
            "name": str,
            "score": float,
            "box": (xmin, ymin, xmax, ymax)
          },
          ...
        ],
        "labels": [
          {
            "description": str,
            "score": float
          },
          ...
        ]
      }
    """
    if frame is None:
        logging.debug("run_object_localization_and_label called with None frame!")
        return {"objects": [], "labels": []}

    # Encode the frame as JPEG
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Failed to encode frame to JPEG.")
        return {"objects": [], "labels": []}

    image = vision.Image(content=encoded.tobytes())
    results = {"objects": [], "labels": []}

    # 1) Object Localization
    try:
        obj_response = vision_client.object_localization(image=image)
        if obj_response.error.message:
            logging.error(f"Object localization error: {obj_response.error.message}")
        else:
            h, w = frame.shape[:2]
            for obj in obj_response.localized_object_annotations:
                xs = [v.x * w for v in obj.bounding_poly.normalized_vertices]
                ys = [v.y * h for v in obj.bounding_poly.normalized_vertices]
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))
                results["objects"].append({
                    "name": obj.name,
                    "score": obj.score,
                    "box": (xmin, ymin, xmax, ymax)
                })
    except Exception as ex:
        logging.error(f"Exception calling object_localization: {ex}")

    # 2) Label Detection
    try:
        label_response = vision_client.label_detection(image=image)
        if label_response.error.message:
            logging.error(f"Label detection error: {label_response.error.message}")
        else:
            for lbl in label_response.label_annotations:
                results["labels"].append({
                    "description": lbl.description,
                    "score": lbl.score
                })
    except Exception as ex:
        logging.error(f"Exception calling label_detection: {ex}")

    return results

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    # Create an OpenCV window
    window_name = "Tello GCP Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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

            # Resize for display
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Call combined detection
            logging.debug("Sending frame to GCP for object + label detection...")
            detection_results = run_object_localization_and_label(display_frame)

            objects = detection_results["objects"]
            labels = detection_results["labels"]
            logging.debug(f"Object count: {len(objects)}, Label count: {len(labels)}")

            # Draw bounding boxes for objects
            for obj in objects:
                (xmin, ymin, xmax, ymax) = obj["box"]
                name = obj["name"]
                score = obj["score"]
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                text_str = f"{name} {score:.2f}"
                cv2.putText(display_frame, text_str, (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Show top 3 labels in the corner
            offset_y = 20
            for lbl in labels[:3]:
                lbl_str = f"{lbl['description']} {lbl['score']:.2f}"
                cv2.putText(display_frame, lbl_str, (10, offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                offset_y += 25

            cv2.imshow(window_name, display_frame)
            logging.debug("cv2.imshow called. Press 'q' to quit.")

            if WAIT_BETWEEN_FRAMES > 0:
                time.sleep(WAIT_BETWEEN_FRAMES)

            # Press 'q' to exit
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