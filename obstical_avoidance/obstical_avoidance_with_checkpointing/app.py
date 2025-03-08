#!/usr/bin/env python3

import cv2
import time
import logging
import sys
import uuid  # For generating unique IDs

from djitellopy import Tello
from google.cloud import vision
from google.cloud import storage  # Add storage import
from google.cloud import bigquery  # Add bigquery import
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

# Proximity thresholds (adjust as needed)
OBJECT_SIZE_FAR = 0.05  # Fraction of frame area
OBJECT_SIZE_MEDIUM = 0.15
OBJECT_SIZE_NEAR = 0.30
OBJECT_SIZE_VERY_NEAR = 0.50

# Drone movement speeds (adjust as needed)
SPEED_SLOW = 20
SPEED_NORMAL = 40

# Google Cloud Configuration
GCP_PROJECT_ID = "enter-universes"  # Replace with your project ID
GCS_BUCKET_NAME = "tello_drone_images"  # Replace with your GCS bucket name
BQ_DATASET_ID = "tello_drone_dataset"  # Replace with your BigQuery dataset ID
BQ_METADATA_TABLE_ID = "tello_metadata"
BQ_COORDINATES_TABLE_ID = "tello_coordinates"


logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------
# Initialize Google Cloud Clients
# --------------------------------------------------------------------
try:
    logging.info("Loading service account credentials...")
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    storage_client = storage.Client(project=GCP_PROJECT_ID, credentials=creds)  # Initialize storage client
    bigquery_client = bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)  # Initialize bigquery client


    logging.info("Vision, Storage and BigQuery clients initialized.")
except Exception as e:
    logging.error(f"Could not initialize Google Cloud clients: {e}")
    sys.exit(1)

# --------------------------------------------------------------------
# Create Google Cloud Resources if they don't exist
# --------------------------------------------------------------------
def create_gcs_bucket(bucket_name):
    """Creates a GCS bucket if it does not exist."""
    try:
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name)
            logging.info(f"Bucket {bucket.name} created")
        else:
            logging.info(f"Bucket {bucket.name} already exists")
    except Exception as e:
        logging.error(f"Error creating/checking bucket {bucket_name}: {e}")
        sys.exit(1)

def create_bigquery_dataset(dataset_id):
    """Creates a BigQuery dataset if it does not exist."""
    dataset_ref = bigquery_client.dataset(dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    try:
        if bigquery_client.get_dataset(dataset_ref) is None:
            dataset = bigquery_client.create_dataset(dataset, timeout=30)
            logging.info(f"Dataset {dataset_id} created")
        else:
            logging.info(f"Dataset {dataset_id} already exists")
    except Exception as e:
        logging.error(f"Error creating/checking dataset {dataset_id}: {e}")
        sys.exit(1)


def create_bigquery_table(dataset_id, table_id, schema):
    """Creates a BigQuery table if it does not exist."""
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    try:
        if bigquery_client.get_table(table_ref) is None:
            table = bigquery_client.create_table(table, timeout=30)  # Make an API request.
            logging.info(f"Table {table_id} created")
        else:
             logging.info(f"Table {table_id} already exists")
    except Exception as e:
        logging.error(f"Error creating/checking table {table_id}: {e}")
        sys.exit(1)

# Define schemas for the BigQuery tables
metadata_schema = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("flight_duration", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("coordinates_fk", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("detected_objects_ocr", "STRING", mode="NULLABLE"),  # JSON representation of objects/OCR data
]

coordinates_schema = [
    bigquery.SchemaField("coordinates_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("object_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("object_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("box_xmin", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("box_ymin", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("box_xmax", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("box_ymax", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("label_description", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("label_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("ocr_text", "STRING", mode="NULLABLE"),
]

# Create resources
create_gcs_bucket(GCS_BUCKET_NAME)
create_bigquery_dataset(BQ_DATASET_ID)
create_bigquery_table(BQ_DATASET_ID, BQ_METADATA_TABLE_ID, metadata_schema)
create_bigquery_table(BQ_DATASET_ID, BQ_COORDINATES_TABLE_ID, coordinates_schema)


# --------------------------------------------------------------------
# Image Handling and GCS Upload
# --------------------------------------------------------------------
def upload_image_to_gcs(frame, bucket_name):
    """Uploads a frame as a JPEG image to GCS.
    Returns the GCS URI of the uploaded image.
    """
    try:
        image_id = str(uuid.uuid4())  # Generate a unique image ID
        image_name = f"uncertain_object_{image_id}.jpg"
        _, encoded_image = cv2.imencode(".jpg", frame)
        blob = storage_client.bucket(bucket_name).blob(image_name)
        blob.upload_from_string(encoded_image.tobytes(), content_type="image/jpeg")
        gcs_uri = f"gs://{bucket_name}/{image_name}"
        logging.info(f"Image uploaded to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logging.error(f"Error uploading image to GCS: {e}")
        return None


# --------------------------------------------------------------------
# BigQuery Insertion
# --------------------------------------------------------------------
def insert_metadata_to_bq(run_id, timestamp, flight_duration, coordinates_fk, detected_objects_ocr):
    """Inserts metadata into the BigQuery metadata table."""
    try:
        rows_to_insert = [
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "flight_duration": flight_duration,
                "coordinates_fk": coordinates_fk,
                "detected_objects_ocr": detected_objects_ocr
            }
        ]
        table_ref = bigquery_client.dataset(BQ_DATASET_ID).table(BQ_METADATA_TABLE_ID)
        errors = bigquery_client.insert_rows(table_ref, rows_to_insert)
        if errors:
            logging.error(f"Errors inserting metadata into BigQuery: {errors}")
        else:
            logging.info("Metadata inserted into BigQuery successfully.")
    except Exception as e:
        logging.error(f"Error inserting metadata into BigQuery: {e}")


def insert_coordinates_to_bq(coordinates_data):
    """Inserts coordinate data into the BigQuery coordinates table."""
    try:
        table_ref = bigquery_client.dataset(BQ_DATASET_ID).table(BQ_COORDINATES_TABLE_ID)
        errors = bigquery_client.insert_rows(table_ref, coordinates_data)
        if errors:
            logging.error(f"Errors inserting coordinates into BigQuery: {errors}")
        else:
            logging.info("Coordinates inserted into BigQuery successfully.")
    except Exception as e:
        logging.error(f"Error inserting coordinates into BigQuery: {e}")

# --------------------------------------------------------------------
# Combined Detection: Objects, Labels, and Text (OCR)
# --------------------------------------------------------------------
def run_object_label_text_detection(frame, run_id, tello):
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
        logging.error(f"Exception calling object_localization: ")

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
        logging.error(f"Exception calling label_detection: ")

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
        logging.error(f"Exception calling text_detection: ")

    # ---
    # Handle Uncertain Detections and Upload Image if Needed
    # ---
    # Implement your logic here to determine if the detection is uncertain.
    # For example, check if the object score is below a threshold or if the
    # OCR confidence is low.  For this example, we'll just assume uncertainty
    # if *any* object has a score less than 0.6, or any label has a score < 0.7.
    uncertain = False
    for obj in results["objects"]:
        if obj["score"] < 0.6:
            uncertain = True
            break
    if not uncertain:
        for lbl in results["labels"]:
            if lbl["score"] < 0.7:
                uncertain = True
                break

    if uncertain:
        logging.warning("Uncertain detection! Taking a photo and saving to GCS.")
        gcs_uri = upload_image_to_gcs(frame, GCS_BUCKET_NAME)
        if gcs_uri:
            logging.info(f"Uploaded uncertain image to: {gcs_uri}")


        # ----
        # Prepare Data for BigQuery
        # ----
        coordinates_id = str(uuid.uuid4())  # Generate a unique ID for coordinates

        coordinates_data = []
        for obj in results["objects"]:
            coordinates_data.append({
                "coordinates_id": coordinates_id,
                "run_id": run_id,
                "object_name": obj["name"],
                "object_score": obj["score"],
                "box_xmin": obj["box"][0],
                "box_ymin": obj["box"][1],
                "box_xmax": obj["box"][2],
                "box_ymax": obj["box"][3],
                "label_description": None,  # Fill in if you have label data associated with the object
                "label_score": None,
                "ocr_text": None,  # Fill in if you have OCR data associated with the object
            })

        for lbl in results["labels"]:
             coordinates_data.append({
                "coordinates_id": coordinates_id,
                "run_id": run_id,
                "object_name": None,
                "object_score": None,
                "box_xmin": None,
                "box_ymin": None,
                "box_xmax": None,
                "box_ymax": None,
                "label_description":lbl["description"],
                "label_score": lbl["score"],
                "ocr_text": None,  # Fill in if you have OCR data associated with the object
            })


        for txt in results["texts"]:
            coordinates_data.append({
                "coordinates_id": coordinates_id,
                "run_id": run_id,
                "object_name": None,
                "object_score": None,
                "box_xmin": None,
                "box_ymin": None,
                "box_xmax": None,
                "box_ymax": None,
                "label_description": None,
                "label_score": None,
                "ocr_text": txt["text"],
            })


        # Prepare a string representation of detected objects and OCR data
        detected_objects_ocr_str = str({"objects": results["objects"], "texts": results["texts"]})

        # Insert coordinates data to BigQuery
        insert_coordinates_to_bq(coordinates_data)

        # Insert metadata to BigQuery (use a placeholder for flight duration)
        insert_metadata_to_bq(run_id, time.time(), None, coordinates_id, detected_objects_ocr_str)
    return results

# --------------------------------------------------------------------
# Drone Control Logic
# --------------------------------------------------------------------
def adjust_drone_movement(tello, objects, frame_width, frame_height):
    """
    Analyzes detected objects and adjusts drone movement to avoid collisions.
    """
    if not objects:
        # If no objects detected, move forward at normal speed
        tello.move_forward(SPEED_NORMAL)
        return

    closest_object = None
    closest_object_size = float('inf')  # Initialize with a large value

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["box"]
        object_width = xmax - xmin
        object_height = ymax - ymin
        object_area = object_width * object_height
        frame_area = frame_width * frame_height
        object_size = object_area / frame_area

        # Find the closest object (largest object size)
        if object_size < closest_object_size:  # Corrected comparison
            closest_object_size = object_size
            closest_object = obj

    if closest_object:
        xmin, ymin, xmax, ymax = closest_object["box"]
        object_width = xmax - xmin
        object_height = ymax - ymin
        object_area = object_width * object_height
        frame_area = frame_width * frame_height
        object_size = object_area / frame_area

        logging.debug(f"Closest object size: {object_size:.2f}")

        if object_size > OBJECT_SIZE_VERY_NEAR:
            logging.warning("Object VERY NEAR! Stopping and moving back.")
            tello.send_rc_control(0, -SPEED_NORMAL, 0, 0)  # Move backward
            time.sleep(0.5)  # Short delay
            tello.send_rc_control(0, 0, 0, 0)  # Stop
            # Implement avoidance maneuver (e.g., move up, down, or sideways)
            # Simple example: move up
            tello.move_up(SPEED_SLOW)
            time.sleep(0.3)
        elif object_size > OBJECT_SIZE_NEAR:
            logging.warning("Object NEAR! Slowing down.")
            tello.move_forward(SPEED_SLOW)
        elif object_size > OBJECT_SIZE_MEDIUM:
            logging.info("Object MEDIUM. Approaching slowly.")
            tello.move_forward(SPEED_NORMAL)
        else:
            logging.info("Object FAR. Moving normally.")
            tello.move_forward(SPEED_NORMAL)

        # Basic left/right avoidance
        object_center_x = (xmin + xmax) / 2
        frame_center_x = frame_width / 2
        horizontal_offset = object_center_x - frame_center_x

        if abs(horizontal_offset) > frame_width / 8:  # If object is off-center
            if horizontal_offset > 0:
                logging.info("Object is to the right. Moving left.")
                tello.send_rc_control(-20, 0, 0, 0)  # Move left
                time.sleep(0.1)
                tello.send_rc_control(0, 0, 0, 0)  # Stop
            else:
                logging.info("Object is to the left. Moving right.")
                tello.send_rc_control(20, 0, 0, 0)  # Move right
                time.sleep(0.1)
                tello.send_rc_control(0, 0, 0, 0)  # Stop

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
    run_id = str(uuid.uuid4())  # Generate a unique ID for the run
    start_time = time.time()  # Capture the start time

    try:
        tello.takeoff()
        tello.move_up(50)  # Move up a bit after takeoff

        while True:
            frame = frame_read.frame
            if frame is None:
                logging.debug("No frame from Tello; sleeping 0.1s")
                time.sleep(0.1)
                continue

            frame_counter += 1
            logging.debug(f"Frame # from Tello, shape={frame.shape}")
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Call all detections
            logging.debug("Sending frame to GCP for Object+Label+OCR detection...")
            res = run_object_label_text_detection(display_frame, run_id, tello)

            objects = res["objects"]
            labels = res["labels"]
            texts = res["texts"]

            logging.debug(f"Objects={len(objects)}, Labels={len(labels)}, Texts={len(texts)}")

            # Adjust drone movement based on object detection
            adjust_drone_movement(tello, objects, FRAME_WIDTH, FRAME_HEIGHT)

            # 1) Draw bounding boxes for objects (GREEN)
            for obj in objects:
                (xmin, ymin, xmax, ymax) = obj["box"]
                name = obj["name"]
                score = obj["score"]
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                txt_str = f" {score:.2f}" #added name of object in the picture
                cv2.putText(display_frame, txt_str, (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 2) Show top labels in top-left corner (CYAN)
            offset_y = 20
            for lbl in labels[:SHOW_TOP_LABELS]:
                desc = lbl["description"]
                score = lbl["score"]
                label_text = f" {score:.2f}" #added description of labels in the picture
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
        logging.info("Landing...")
        tello.land()
        logging.info("Cleaning up...")
        tello.streamoff()

        end_time = time.time()  # Capture the end time
        flight_duration = end_time - start_time  # Calculate flight duration
        logging.info(f"Flight duration: {flight_duration:.2f} seconds")

        # Update metadata with flight duration
        # You can update BigQuery data, but it's often more efficient to calculate
        # derived metrics later using SQL in BigQuery itself.  For the sake of
        # example, we'll update.  This requires a read + update using a Merge statement.
        #update_metadata_with_duration(run_id, flight_duration)


        cv2.destroyAllWindows()
        logging.info("Done.")

if __name__ == "__main__":
    main()
