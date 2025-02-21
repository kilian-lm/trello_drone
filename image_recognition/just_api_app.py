#!/usr/bin/env python3

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
SERVICE_ACCOUNT_FILE = "/Users/kilianlehn/Documents/GitHub/trello_drone/image_recognition/enter-universes-fcf7ca441146.json"  # <-- Update path
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LOGGING_LEVEL = logging.DEBUG
WAIT_BETWEEN_FRAMES = 0  # If you want a delay after each frame
SHOW_TOP_LABELS = 3       # How many labels to overlay on the image

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
    logging.error(f"Could not init Vision API: {e}")
    sys.exit(1)

# --------------------------------------------------------------------
# Combined Detection: Objects, Labels, and Text (OCR)
# --------------------------------------------------------------------
def run_object_label_text_detection(frame):
    """
    Sends the frame to GCP Vision for:
      1) Object Localization
      2) Label Detection
      3) Text Detection (OCR)

    Returns a dict of:
      {
        "objects": [
          {
            "name": str,
            "score": float,
            "box": (xmin, ymin, xmax, ymax)
          }, ...
        ],
        "labels": [
          {
            "description": str,
            "score": float
          }, ...
        ],
        "texts": [
          {
            "text": str,
            "box": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # corners
          }, ...
        ]
      }
    """
    if frame is None:
        logging.debug("Frame is None - skipping detection.")
        return {"objects": [], "labels": [], "texts": []}

    # Encode as JPEG
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Failed to encode frame.")
        return {"objects": [], "labels": [], "texts": []}

    image = vision.Image(content=encoded.tobytes())
    height, width = frame.shape[:2]

    results = {
        "objects": [],
        "labels": [],
        "texts": []
    }

    # ----------------------
    # 1) Object Localization
    # ----------------------
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

    # ----------------------
    # 2) Label Detection
    # ----------------------
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

    # ----------------------
    # 3) Text Detection (OCR)
    # ----------------------
    try:
        text_resp = vision_client.text_detection(image=image)
        if text_resp.error.message:
            logging.error(f"Text detection error: {text_resp.error.message}")
        else:
            # text_annotations[0] is the entire text block,
            # subsequent entries can be individual lines/elements
            for i, txt in enumerate(text_resp.text_annotations):
                # bounding_poly might have 4 vertices for a rectangle
                poly_verts = txt.bounding_poly.vertices
                corners = []
                for v in poly_verts:
                    # If x or y is None, default to 0
                    x_val = int(v.x) if v.x else 0
                    y_val = int(v.y) if v.y else 0
                    corners.append((x_val, y_val))

                # text is in txt.description
                results["texts"].append({
                    "text": txt.description,
                    "box": corners  # list of 4 (x,y) pairs
                })
    except Exception as ex:
        logging.error(f"Exception calling text_detection: {ex}")

    return results

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    window_name = "Tello + GCP (Objects+Labels+OCR)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    logging.info("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    logging.info(f"Battery: {tello.get_battery()}%")

    logging.info("Starting Tello camera stream...")
    tello.streamon()

    # Let Tello warm up
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
                logging.debug("No frame from Tello; sleeping 0.1s")
                time.sleep(0.1)
                continue

            frame_counter += 1
            logging.debug(f"Frame #{frame_counter} from Tello, shape={frame.shape}")
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Call all detections
            logging.debug("Sending frame to GCP for Object+Label+OCR detection...")
            res = run_object_label_text_detection(display_frame)

            objects = res["objects"]
            labels = res["labels"]
            texts = res["texts"]

            logging.debug(f"Objects={len(objects)}, Labels={len(labels)}, Texts={len(texts)}")

            # 1) Draw bounding boxes for objects (GREEN)
            for obj in objects:
                (xmin, ymin, xmax, ymax) = obj["box"]
                name = obj["name"]
                score = obj["score"]
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                txt_str = f"{name} {score:.2f}"
                cv2.putText(display_frame, txt_str, (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 2) Show top labels in top-left corner (CYAN)
            offset_y = 20
            for lbl in labels[:SHOW_TOP_LABELS]:
                desc = lbl["description"]
                score = lbl["score"]
                label_text = f"{desc} {score:.2f}"
                cv2.putText(display_frame, label_text, (10, offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                offset_y += 25

            # 3) Draw text bounding boxes (MAGENTA) + text
            # text_annotations[0] might be the entire block, so we do all
            for i, t in enumerate(texts):
                corners = t["box"]  # list of up to 4 points
                recognized_text = t["text"].strip()
                if not corners or len(corners) < 4:
                    continue

                # Draw polygon
                for j in range(len(corners)):
                    pt1 = corners[j]
                    pt2 = corners[(j+1) % len(corners)]  # next corner, wrap around
                    cv2.line(display_frame, pt1, pt2, (255,0,255), 2)

                # If you want to overlay the text near the top-left of the box:
                (x_text, y_text) = corners[0]
                cv2.putText(display_frame, recognized_text, (x_text, y_text - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            cv2.imshow(window_name, display_frame)
            logging.debug("Press 'q' to quit.")

            if WAIT_BETWEEN_FRAMES > 0:
                time.sleep(WAIT_BETWEEN_FRAMES)

            # Press 'q' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("User pressed 'q'. Exiting loop.")
                break

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt, exiting loop.")
    finally:
        logging.info("Cleaning up...")
        tello.streamoff()
        cv2.destroyAllWindows()
        logging.info("Done.")

if __name__ == "__main__":
    main()