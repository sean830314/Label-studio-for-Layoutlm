import argparse
import datetime
import json
import logging
import os
import uuid

import pymongo
import pytesseract
import requests
from PIL import Image

LS_QA_ENDPOINT = os.getenv("LS_QA_ENDPOINT")
LS_QA_TOKEN = os.getenv("LS_QA_TOKEN")
LS_QA_PII_PROJECT_ID = os.getenv("LS_QA_PII_PROJECT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s - %(module)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_and_send_message(messages, date):
    task_type = "LayoutLM V3"
    endpoint = "{}/api/projects/{}".format(LS_QA_ENDPOINT, LS_QA_PII_PROJECT_ID)
    token = LS_QA_TOKEN
    for index, message in enumerate(messages):
        task_id = message["task_id"]
        output = {
            "data": {
                "text": str(message["text"]),
                "meta_info": {
                    "task_id": task_id,
                    "record_time": str(datetime.date.today()),
                },
            },
            "meta": {"task_id": task_id, "import_date": date, "task_type": task_type},
        }
        task_ids = get_taskId(endpoint, token)
        if task_id in task_ids:
            logger.info(f"skip the task({task_id})")
            continue
        rc = import_data(output, endpoint, token)
        if rc != requests.codes.created:
            logger.error(f"Return Code: {rc}")
            logger.error(
                f"Import data failed, (task_id: {task_id}, page: {str(index + 1)})"
            )
        else:
            logger.info(f"Data imported. (task_id: {task_id}, page: {str(index + 1)})")


def import_data(output, endpoint, token):
    ret_code = -1
    headers = {"content-type": "application/json", "Authorization": f"Token {token}"}
    try:
        resp = requests.post(
            "{}/import".format(endpoint), headers=headers, data=json.dumps(output)
        )
        ret_code = resp.status_code
    except Exception as e:
        logger.error(e)

    return ret_code


def get_taskId(endpoint, token):
    headers = {"content-type": "application/json", "Authorization": f"Token {token}"}
    try:
        resp = requests.get("{}/tasks".format(endpoint), headers=headers)
        if resp.status_code == 200:
            task_ids = list()
            for item in resp.json():
                if "task_id" in item["meta"].keys():
                    task_ids.append(item["meta"]["task_id"])
            return task_ids
        else:
            raise Exception("return code not 200")
    except Exception as e:
        logger.error(e)
        return None


def run_pytesseract_ocr(images, file_names):
    """This function ocr images of pytesseract into a list of self-objects

    Args:
       images (str):  The image to use.
       file_names (str): The image name to use

    Returns:
       list of self-objects.

    Raises:
       None

    """
    result = list()
    for image, filename in zip(images, file_names):
        logger.info(f"ocr {filename}")
        ocr_dict = pytesseract.image_to_data(
            image, lang="eng", output_type="dict", config="--oem 3 --dpi 600"
        )
        words, left, top, width, height = (
            ocr_dict["text"],
            ocr_dict["left"],
            ocr_dict["top"],
            ocr_dict["width"],
            ocr_dict["height"],
        )
        # filter empty words and corresponding coordinates
        irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
        words = [
            word for idx, word in enumerate(words) if idx not in irrelevant_indices
        ]
        left = [
            coord for idx, coord in enumerate(left) if idx not in irrelevant_indices
        ]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [
            coord for idx, coord in enumerate(width) if idx not in irrelevant_indices
        ]
        height = [
            coord for idx, coord in enumerate(height) if idx not in irrelevant_indices
        ]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = []
        for x, y, w, h in zip(left, top, width, height):
            actual_box = [x, y, x + w, y + h]
            actual_boxes.append(actual_box)
        assert len(words) == len(
            actual_boxes
        ), "Not as many words as there are bounding boxes"
        text = " ".join([word for word in words])
        task_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, text))
        result.append(
            {
                "token": words,
                "bbox": actual_boxes,
                "filename": filename,
                "text": text,
                "task_id": task_id,
                "project_id": LS_QA_PII_PROJECT_ID,
                "type": "train",
            }
        )
    return {"messages": result}


def insert_message_to_mongodb(db, data):
    image_metadata = [
        {
            "filename": message["filename"],
            "text": message["text"],
            "task_id": message["task_id"],
            "project_id": message["project_id"],
            "type": message["type"],
        }
        for message in data["messages"]
    ]
    image_data = [
        {
            "task_id": message["task_id"],
            "project_id": message["project_id"],
            "token": message["token"],
            "bbox": message["bbox"],
        }
        for message in data["messages"]
    ]
    insert_data_to_collection(db, "image_metadata", image_metadata)
    insert_data_to_collection(db, "image_data", image_data)


def connect_mongo_dbs(host, databases):
    print(os.getenv("MONGODB_USERNAME"))
    print(os.getenv("MONGODB_PASSWORD"))
    print(
        "mongodb://{}:{}@{}/".format(
            os.getenv("MONGODB_USERNAME"), os.getenv("MONGODB_PASSWORD"), host
        )
    )
    myclient = pymongo.MongoClient(
        "mongodb://{}:{}@{}/".format(
            os.getenv("MONGODB_USERNAME"), os.getenv("MONGODB_PASSWORD"), host
        )
    )
    db = myclient[databases]
    return db


def insert_data_to_collection(db, collection, records):
    x = db[collection].insert_many(records)
    print(x)


def load_images_from_folder(folder):
    """This function load images of folder into a list of PIL objects

    Args:
       folder (str):  The folder to use.

    Returns:
       list of PIL objects (PIL object).
       list of file name (str).

    Raises:
       None

    """
    images = []
    file_names = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        logger.info(f"read {image_path}")
        img = Image.open(image_path)
        if img is not None:
            images.append(img)
            file_names.append(image_path)
    assert len(images) == len(file_names)
    return images, file_names


def main(args):
    db = connect_mongo_dbs(args.mongo_host, args.mongo_databases)
    folder = args.folder
    logger.info(f"load image from {folder} folder")
    images, file_names = load_images_from_folder(folder)
    logger.info(f"run pytesseract ocr")
    data = run_pytesseract_ocr(images, file_names)
    if len(data["messages"]) > 0:
        format_and_send_message(data["messages"], args.date)
        insert_message_to_mongodb(db, data)
    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--date", required=True, help="Execution date of parser, ex. 2021-01-01"
    )
    parser.add_argument(
        "-f", "--folder", required=True, help="Execution folder, ex. ./images"
    )
    parser.add_argument(
        "-mongo_host",
        "--mongo_host",
        default="localhost:27017",
        required=True,
        help="connect mongo host, ex. localhost:27017",
    )
    parser.add_argument(
        "-mongo_dbs",
        "--mongo_databases",
        default="mydbs",
        required=False,
        help="connect mongo databases, ex. default_databases",
    )
    args = parser.parse_args()
    main(args)
