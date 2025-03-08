#!/usr/bin/env python3

import cv2
import time
import logging
import sys
import uuid
import threading
import math

from djitellopy import Tello
from google.cloud import vision
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import pubsub_v1  # Add pubsub import
from google.oauth2 import service_account
import json

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "/Users/kilianlehn/Documents/GitHub/trello_drone/image_recognition/enter-universes-fcf7ca441146.json"  # <-- Update path
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LOGGING_LEVEL = logging.DEBUG
WAIT_BETWEEN_FRAMES = 0
SHOW_TOP_LABELS = 3

OBJECT_SIZE_FAR = 0.05
OBJECT_SIZE_MEDIUM = 0.15
OBJECT_SIZE_NEAR = 0.30
OBJECT_SIZE_VERY_NEAR = 0.50

SPEED_SLOW = 20
SPEED_NORMAL = 40

# Google Cloud Configuration
GCP_PROJECT_ID = "enter-universes"  # Replace with your project ID
GCS_BUCKET_NAME = "tello_drone_images"  # Replace with your GCS bucket name
BQ_DATASET_ID = "tello_drone_dataset"  # Replace with your BigQuery dataset ID
BQ_METADATA_TABLE_ID = "tello_metadata"
BQ_COORDINATES_TABLE_ID = "tello_coordinates"
PUBSUB_TOPIC_ID = "drone-communication"  # Replace with your Pub/Sub topic ID
DRONE_ID = str(uuid.uuid4())  # Generate a unique ID for this drone instance

# Drone Roles
ORCHESTRATOR_DRONE = "orchestrator"
SCANNER_DRONE = "scanner"

DRONE_ROLE = SCANNER_DRONE  # Set the role of this drone. Change to ORCHESTRATOR_DRONE for the orchestrator.

# Initial Drone Relative Positions (Orchestrator Knows)
INITIAL_DRONE_POSITIONS = {
    "drone_1": {"x": 0, "y": 0, "z": 0}, #Orchestrator Location
    "drone_2": {"x": 100, "y": 0, "z": 0},#1 meter appart
    "drone_3": {"x": 0, "y": 100, "z": 0} #1 meter appart

}
if DRONE_ROLE == ORCHESTRATOR_DRONE:
    drone_positions = INITIAL_DRONE_POSITIONS #Orchestrator "knows" initial positions.
else:
    drone_positions = {} #Start empty for scanners.

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
    storage_client = storage.Client(project=GCP_PROJECT_ID, credentials=creds)
    bigquery_client = bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)
    publisher = pubsub_v1.PublisherClient(credentials=creds)  # Initialize Pub/Sub publisher
    subscriber = pubsub_v1.SubscriberClient(credentials=creds)  # Initialize Pub/Sub subscriber

    topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC_ID)

    logging.info("Vision, Storage, BigQuery and Pub/Sub clients initialized.")
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


def create_pubsub_topic(topic_id):
    """Creates a Pub/Sub topic if it does not exist."""
    try:
        topic_path = publisher.topic_path(GCP_PROJECT_ID, topic_id)
        try:
            topic = publisher.get_topic(request={"topic": topic_path})
            logging.info(f"Topic {topic.name} already exists")
        except Exception as e:
            topic = publisher.create_topic(request={"name": topic_path})
            logging.info(f"Created topic {topic.name}")
    except Exception as e:
        logging.error(f"Error creating/checking topic {topic_id}: {e}")
        sys.exit(1)


# Define schemas for the BigQuery tables
metadata_schema = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("flight_duration", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("coordinates_fk", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("detected_objects_ocr", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("drone_id", "STRING", mode="NULLABLE")  # Add drone ID

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
    bigquery.SchemaField("drone_id", "STRING", mode="NULLABLE") # Add drone ID
]

# Create resources
create_gcs_bucket(GCS_BUCKET_NAME)
create_bigquery_dataset(BQ_DATASET_ID)
create_bigquery_table(BQ_DATASET_ID, BQ_METADATA_TABLE_ID, metadata_schema)
create_bigquery_table(BQ_DATASET_ID, BQ_COORDINATES_TABLE_ID, coordinates_schema)
create_pubsub_topic(PUBSUB_TOPIC_ID)

# --------------------------------------------------------------------
# Image Handling and GCS Upload
# --------------------------------------------------------------------
def upload_image_to_gcs(frame, bucket_name):
    """Uploads a frame as a JPEG image to GCS.
    Returns the GCS URI of the uploaded image.
    """
    try:
        image_id = str(uuid.uuid4())
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
def insert_metadata_to_bq(run_id, timestamp, flight_duration, coordinates_fk, detected_objects_ocr, drone_id):
    """Inserts metadata into the BigQuery metadata table."""
    try:
        rows_to_insert = [
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "flight_duration": flight_duration,
                "coordinates_fk": coordinates_fk,
                "detected_objects_ocr": detected_objects_ocr,
                "drone_id": drone_id  # Add drone ID
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
# Pub/Sub Communication
# --------------------------------------------------------------------
def publish_message(message_data):
    """Publishes a message to the Pub/Sub topic."""
    try:
        data = json.dumps(message_data).encode("utf-8")
        future = publisher.publish(topic_path, data=data)
        logging.info(f"Published message: {message_data}")
        future.result()  # Block until the publish succeeds
    except Exception as e:
        logging.error(f"Error publishing message: {e}")

def callback(message: pubsub_v1.subscriber.message.Message):
    """Callback function for processing incoming Pub/Sub messages."""
    try:
        message_data = json.loads(message.data.decode("utf-8"))
        logging.info(f"Received message: {message_data}")

        # Handle different types of messages based on a 'message_type' key
        message_type = message_data.get("message_type")

        if message_type == "new_product_coordinates":
            # Orchestrator: assign the task to scanner drones
            if DRONE_ROLE == ORCHESTRATOR_DRONE:
                assign_task_to_scanner(message_data)
        elif message_type == "product_scanned":
            # Scanner: notify that a product has been scanned
            if DRONE_ROLE == SCANNER_DRONE:
                notify_product_scanned(message_data)
        elif message_type == "drone_position":
            #Scanner: Handle relative position updates from other drones.
            if DRONE_ROLE == SCANNER_DRONE:
                update_relative_position(message_data)
        elif message_type == "relative_position_estimate":
            # Orchestrator:  Process relative position estimates from the drones
            if DRONE_ROLE == ORCHESTRATOR_DRONE:
                process_relative_position_estimate(message_data)


        # Acknowledge the message
        message.ack()
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        message.nack()  # Nack the message to retry


def subscribe_messages():
    """Subscribes to the Pub/Sub topic and listens for messages."""
    subscription_id = f"drone-subscription-{DRONE_ID}"
    subscription_path = subscriber.subscription_path(GCP_PROJECT_ID, subscription_id)

    try:
        # Check if the subscription exists, create if not
        try:
            subscriber.get_subscription(request={"subscription": subscription_path})
            logging.info(f"Subscription {subscription_id} already exists")
        except:  #TODO Specify exception google.api_core.exceptions.NotFound
            topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC_ID)
            subscription = subscriber.create_subscription(
                request={"name": subscription_path, "topic": topic_path}
            )
            logging.info(f"Created subscription {subscription.name}")

        # Start the subscriber
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        logging.info(f"Listening for messages on {subscription_path}...")
        # Keep the main thread alive to continue listening for messages
        try:
            streaming_pull_future.result()
        except Exception as e:
            logging.error(f"Error while listening for messages: {e}")
            streaming_pull_future.cancel()

    except Exception as e:
        logging.error(f"Error subscribing to topic: {e}")
        sys.exit(1)

# --------------------------------------------------------------------
# Drone Agent Logic
# --------------------------------------------------------------------

# Placeholder data structure for unscanned products
unscanned_products = {
    "product_1": {"x": 100, "y": 200, "z": 50},
    "product_2": {"x": 300, "y": 400, "z": 80},
    "product_3": {"x": 500, "y": 600, "z": 30},
}

# drone_positions = {} # Dictionary to store relative positions of drones -Now global (see above)

def assign_task_to_scanner(product_data):
    """Orchestrator: Assigns a scanning task to a scanner drone."""
    # Simple task assignment logic: assign based on proximity (can be improved)
    # This function needs access to the current positions of scanner drones and the product location
    # Positions could be tracked via messages the drones send periodically
    # NOTE: This is a VERY basic example.  Real-world would need sophisticated task assignment.

    closest_drone = None
    min_distance = float('inf')
    product_coords = product_data["product_coords"]

    #Access the global dictionary of drone positions that the scanner drones update.
    global drone_positions

    for drone_id, drone_coords in drone_positions.items():  # Assuming we have drone positions
        distance = ((product_coords["x"] - drone_coords["x"]) ** 2 +
                    (product_coords["y"] - drone_coords["y"]) ** 2 +
                    (product_coords["z"] - drone_coords["z"]) ** 0.5
                    if distance < min_distance:
                    min_distance = distance
                    closest_drone = drone_id

                    if closest_drone:
                    task_assignment = {
                    "message_type": "scan_product",
                    "drone_id": closest_drone,
                    "product_name": product_data["product_name"],
                    "product_coords": product_coords,
                    }
                    publish_message(task_assignment)
        logging.info(f"Orchestrator assigned {product_data['product_name']} to drone {closest_drone}")


def notify_product_scanned(message_data):
    """Scanner: Handles the notification that a product has been scanned."""
    product_name = message_data["product_name"]
    logging.info(f"Drone {DRONE_ID} scanned product {product_name}")
    # Remove the scanned product from the unscanned_products list
    global unscanned_products
    if product_name in unscanned_products:
        del unscanned_products[product_name]
    # Notify the orchestrator that the product has been scanned
    # Send a message with the type "product_scanned" and any relevant details
    message = {
        "message_type": "product_scanned",
        "drone_id": DRONE_ID,
        "product_name": product_name,
    }
    publish_message(message)


def update_relative_position(message_data):
    """Scanner: Updates the relative position of another drone based on estimated data."""
    other_drone_id = message_data["drone_id"]
    relative_position = message_data["relative_position"]

    # Update the drone_positions dictionary with the received relative position
    drone_positions[other_drone_id] = relative_position

    logging.info(f"Drone {DRONE_ID} updated relative position of drone {other_drone_id}: {relative_position}")

def process_relative_position_estimate(message_data):
    """Orchestrator: Process relative position estimates and update drone positions."""
    drone_id = message_data["drone_id"]
    estimated_position = message_data["estimated_position"]

    # Update the drone_positions dictionary with the estimated position
    drone_positions[drone_id] = estimated_position
    logging.info(f"Orchestrator updated position of {drone_id} to {estimated_position}")


def get_next_target():
    """Scanner: Gets the next unscanned product to scan."""
    global unscanned_products
    if unscanned_products:
        product_name, product_coords = unscanned_products.popitem()
        logging.info(f"Drone {DRONE_ID} is targeting product {product_name} at {product_coords}")
        return product_name, product_coords
    else:
        logging.info("No more products to scan.")
        return None, None


def scanner_drone_logic(tello):
    """Scanner Drone: Logic for autonomous scanning of retail market products."""

    try:
        tello.takeoff()
        tello.move_up(50)

        #Store starting position.
        start_x = 0 #tello.get_distance_x() #Start at 0,0,0 for relative movement calculation.
        start_y = 0#tello.get_distance_y()
        start_z = 0 #tello.get_distance_z()
        logging.info(f"Drone start position {start_x},{start_y},{start_z}")

        # Report drone position to orchestrator
        def report_position_to_orchestrator():

            #1. Use tello distance sensor for the current position.
            current_x = tello.get_distance_x()
            current_y = tello.get_distance_y()
            current_z = tello.get_distance_z()
            current_position = {"x": current_x, "y": current_y, "z": current_z}

            #2. Estimate the new drone position relative to initial position.
            estimated_position = {
                "x": INITIAL_DRONE_POSITIONS[DRONE_ID]["x"] + current_x, #Assumes initial position.
                "y": INITIAL_DRONE_POSITIONS[DRONE_ID]["y"] + current_y,
                "z": INITIAL_DRONE_POSITIONS[DRONE_ID]["z"] + current_z,
            }

            position_message = {
                "message_type": "relative_position_estimate",
                "drone_id": DRONE_ID,
                "estimated_position": estimated_position
            }
            publish_message(position_message)

            #If other drones exist, calculate their positions.
            for other_drone_id in INITIAL_DRONE_POSITIONS:
                if other_drone_id != DRONE_ID:
                    #Send the *relative* position so each drone can localize.
                    relative_position = {
                        "x": estimated_position["x"] - drone_positions[other_drone_id]["x"],
                        "y": estimated_position["y"] - drone_positions[other_drone_id]["y"],
                        "z": estimated_position["z"] - drone_positions[other_drone_id]["z"],
                    }

                    relative_message = {
                        "message_type": "drone_position",
                        "drone_id": other_drone_id,
                        "relative_position": relative_position,
                    }
                    publish_message(relative_message)
            time.sleep(5)  #Report every 5 seconds


        position_thread = threading.Thread(target=report_position_to_orchestrator, daemon=True)
        position_thread.start()

        while True:

            #2. Check for tasks.

            #3. If no products remain, land.
            global unscanned_products
            if not unscanned_products:
                logging.info("No more products to scan. Landing.")
                tello.land()
                break

            #4. Get next target product
            product_name, product_coords = get_next_target()

            #Check for none in case products were deleted elsewhere.
            if product_name is None or product_coords is None:
                continue

            #5. Move to the target product location.
            # Replace with actual movement commands using tello.move_forward(), tello.rotate_clockwise(), etc.
            # The product_coords should be relative to the drone's starting position
            # You'll likely need a more sophisticated path planning algorithm for a real-world environment
            logging.info(f"Moving towards {product_name} at {product_coords}")

            #This is all pseudo code as it depends on coordinate system
            relative_x = product_coords["x"] - (INITIAL_DRONE_POSITIONS[DRONE_ID]["x"] + start_x)
            relative_y = product_coords["y"] - (INITIAL_DRONE_POSITIONS[DRONE_ID]["y"] + start_y)
            relative_z = product_coords["z"] - (INITIAL_DRONE_POSITIONS[DRONE_ID]["z"] + start_z)

            #Use this relative to intial positions.
            tello.go_xyz_speed(relative_x, relative_y, relative_z, SPEED_NORMAL)
            time.sleep(5) #Wait to arrive

            #6. Scan the product and upload data.
            frame = tello.get_frame_read().frame
            if frame is not None:
                run_id = str(uuid.uuid4())
                res = run_object_label_text_detection(frame, run_id, tello)

                # 7. After scanning, notify orchestrator that the product has been scanned.
                if res:
                    scan_message = {
                        "message_type": "product_scanned",
                        "drone_id": DRONE_ID,
                        "product_name": product_name,
                        "run_id": run_id
                    }
                    publish_message(scan_message)
                    notify_product_scanned({"product_name": product_name})#Also notify locally.

            else:
                logging.warning("Could not capture frame for scanning.")

            time.sleep(2) #Give time for the scanner to stablilize.

    except Exception as e:
        logging.error(f"Error in scanner drone logic: {e}")
        tello.land()
    finally:
        tello.land()

def orchestrator_drone_logic():
    """Orchestrator Drone: Logic for task assignment and coordination."""
    # In a real-world scenario, the orchestrator drone would:
    # 1. Initialize the list of unscanned products from a database or inventory system.
    # 2. Listen for messages from the scanner drones about their positions and available resources.
    # 3. Assign tasks to the scanner drones based on their proximity to the unscanned products and their available resources.
    # 4. Monitor the progress of the scanning tasks and reassign tasks as needed.
    # 5. Collect the data from the scanner drones and upload it to a central repository.

    # Initialize unscanned products (replace with actual data source)
    global unscanned_products
    unscanned_products = {
        "product_1": {"x": 100, "y": 200, "z": 50},
        "product_2": {"x": 300, "y": 400, "z": 80},
        "product_3": {"x": 500, "y": 600, "z": 30},
        "product_4": {"x": 150, "y": 250, "z": 60},
        "product_5": {"x": 350, "y": 450, "z": 90},
    }

    logging.info("Orchestrator drone logic started.")

    # Send initial tasks to scanner drones based on the current list of unscanned products
    for product_name, product_coords in unscanned_products.items():
        task_data = {
            "message_type": "new_product_coordinates",
            "product_name": product_name,
            "product_coords": product_coords,
        }
        publish_message(task_data)
        time.sleep(1)  # Introduce a delay to avoid overwhelming the scanners
    logging.info("Finished distibuting products.")


# --------------------------------------------------------------------
# Combined Detection: Objects, Labels, and Text (OCR)
# --------------------------------------------------------------------
def run_object_label_text_detection(frame, run_id, tello):
    """
    Sends the frame to GCP Vision for:
      1) Object Localization
      2) Label Detection
      3) Text Detection (OCR)
    """
    if frame is None:
        logging.debug("Frame is None - skipping detection.")
        return None

    # Encode as JPEG
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        logging.error("Failed to encode frame.")
        return None

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
                "drone_id": DRONE_ID
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
                "drone_id": DRONE_ID
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
                "drone_id": DRONE_ID
            })
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
    closest_object_size = float('inf')

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["box"]
        object_width = xmax - xmin
        object_height = ymax - ymin
        object_area = object_width * object_height
        frame_area = frame_width * frame_height
        object_size = object_area / frame_area

        if object_size < closest_object_size:
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
            tello.send_rc_control(0, -SPEED_NORMAL, 0, 0)
            time.sleep(0.5)
            tello.send_rc_control(0, 0, 0, 0)
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

        object_center_x = (xmin + xmax) / 2
        frame_center_x = frame_width / 2
        horizontal_offset = object_center_x - frame_center_x

        if abs(horizontal_offset) > frame_width / 8:
            if horizontal_offset > 0:
                logging.info("Object is to the right. Moving left.")
                tello.send_rc_control(-20, 0, 0, 0)
                time.sleep(0.1)
                tello.send_rc_control(0, 0, 0, 0)
            else:
                logging.info("Object is to the left. Moving right.")
                tello.send_rc_control(20, 0, 0, 0)
                time.sleep(0.1)
                tello.send_rc_control(0, 0, 0, 0)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():

    #1. Initialize the Tello drone
    window_name = f"Tello + GCP (Objects+Labels+OCR) - Drone {DRONE_ID}" #Unique ID in the name.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    logging.info(f"Connecting to Tello as Drone {DRONE_ID}...")
    tello = Tello()
    try:
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

        # Create a thread to listen for Pub/Sub messages
        pubsub_thread = threading.Thread(target=subscribe_messages, daemon=True)
        pubsub_thread.start()

        # Start the drone logic based on its role
        if DRONE_ROLE == SCANNER_DRONE:
            scanner_drone_logic(tello)  # Pass the tello object
        elif DRONE_ROLE == ORCHESTRATOR_DRONE:
            orchestrator_drone_logic()
            #Orchestrator only runs once, so the drone can land.
            tello.land()
        else:
            logging.error("Invalid DRONE_ROLE. Exiting.")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        tello.land()

    finally:
        logging.info("Cleaning up...")
        tello.streamoff()
        cv2.destroyAllWindows()
        logging.info("Done.")

if __name__ == "__main__":
    main()
