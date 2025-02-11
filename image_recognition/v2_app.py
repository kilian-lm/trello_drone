import cv2
import time
import logging
from djitellopy import Tello
from google.cloud import vision
from google.oauth2 import service_account

logging.basicConfig(level=logging.DEBUG)

SERVICE_ACCOUNT_FILE = "/Users/d0342084/Documents/Git/trello_drone/image_recognition/some_unwanted_creatre.json"  # <-- Update this path!

def main():
    # 1) GCP Credentials
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    logging.info("Initialized GCP Vision Client.")

    # 2) Tello connect
    tello = Tello()
    tello.connect()
    logging.info(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    time.sleep(2)

    frame_read = tello.get_frame_read()

    cv2.namedWindow("Tello + GCP", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tello + GCP", 640, 480)

    last_gcp_time = time.time()
    GCP_INTERVAL = 5  # seconds, to avoid spamming GCP

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                logging.debug("No frame from Tello.")
                time.sleep(0.1)
                continue

            display_frame = cv2.resize(frame, (640, 480))

            # Only call GCP every GCP_INTERVAL seconds
            if time.time() - last_gcp_time > GCP_INTERVAL:
                logging.debug("Calling GCP Vision on current frame...")
                detections = run_object_localization(vision_client, display_frame)
                logging.debug(f"GCP returned {len(detections)} objects.")

                # Draw bounding boxes
                for det in detections:
                    (xmin, ymin, xmax, ymax) = det["box"]
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                    txt = f"{det['name']} {det['score']:.2f}"
                    cv2.putText(display_frame, txt, (xmin, ymin-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                last_gcp_time = time.time()

            # Show on screen
            cv2.imshow("Tello + GCP", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()

def run_object_localization(vision_client, frame):
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Frame encoding failed.")
        return []

    image = vision.Image(content=encoded.tobytes())
    try:
        response = vision_client.object_localization(image=image)
        if response.error.message:
            logging.error(f"Vision API error: {response.error.message}")
            return []
    except Exception as ex:
        logging.error(f"Exception calling GCP: {ex}")
        return []

    height, width = frame.shape[:2]
    detections = []
    for obj in response.localized_object_annotations:
        xs = [v.x * width for v in obj.bounding_poly.normalized_vertices]
        ys = [v.y * height for v in obj.bounding_poly.normalized_vertices]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))
        detections.append({
            "name": obj.name,
            "score": obj.score,
            "box": (xmin, ymin, xmax, ymax)
        })
    return detections

if __name__ == "__main__":
    main()
