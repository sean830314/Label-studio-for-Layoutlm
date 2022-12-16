import argparse
import json
import logging
import os
import re
from operator import itemgetter

import pymongo
import requests

LS_QA_ENDPOINT = os.getenv("LS_QA_ENDPOINT")
LS_QA_TOKEN = os.getenv("LS_QA_TOKEN")
LS_QA_PII_PROJECT_ID = os.getenv("LS_QA_PII_PROJECT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s - %(module)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def export_data(endpoint, project_id, token):
    ret_code = -1
    headers = {"content-type": "application/json", "Authorization": f"Token {token}"}
    try:
        resp = requests.get(
            "{}/api/projects/{}/export?exportType=JSON".format(endpoint, project_id),
            headers=headers,
        )
        ret_code = resp.status_code
        if ret_code == 200:
            return {"data": resp.json()}
        else:
            return None
    except Exception as e:
        logger.error(e)


def connect_mongo_dbs(host, databases):
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


def find_token_bbox_from_collection(db, task_id, project_id):
    for record in db["image_data"].find({"task_id": task_id, "project_id": project_id}):
        return record["token"], record["bbox"]


def find_image_metadata_from_collection(db, task_id, project_id):
    for record in db["image_metadata"].find(
        {"task_id": task_id, "project_id": project_id}
    ):
        return record


def get_annotations(task, text):
    result = list()
    for annotations in task["annotations"]:
        for annotation in annotations["result"]:
            item = {
                "start": annotation["value"]["start"],
                "end": annotation["value"]["end"],
                "label": annotation["value"]["labels"][0],
                "text": annotation["value"]["text"].lstrip().rstrip(),
                "ids": [],
            }
            result.append(item)
    result = sorted(result, key=itemgetter("start"))
    # calculate ids
    for item in result:
        token_len = len([m.start() for m in re.finditer(" ", item["text"])])
        start_index = item["start"]
        if start_index > 0:
            token_id = len([m.start() for m in re.finditer(" ", text[:start_index])])
            for _ in range(token_len + 1):
                item["ids"].append(token_id)
                token_id = token_id + 1
        else:
            token_id = 0
            for _ in range(token_len + 1):
                item["ids"].append(token_id)
                token_id = token_id + 1
    return result


def merge_data(tokens, bboxs, annotations):
    assert len(tokens) == len(bboxs)
    new_hf_token_struct = list()
    hf_token_struct = list()
    # transform struct to huggingface dataset format
    for id, (token, bbox) in enumerate(zip(tokens, bboxs)):
        item = {
            "text": token,
            "label": "OTHERS",
            "box": bbox,
            "words": [{"box": bbox.copy(), "text": token}],
            "id": id,
        }
        hf_token_struct.append(item)
    # merge annotations
    remove_ids = list()
    for annotation in annotations:
        if len(annotation["ids"]) == 1:
            start_token_index = next(
                (
                    index
                    for (index, d) in enumerate(hf_token_struct)
                    if d["id"] == annotation["ids"][0]
                ),
                None,
            )
            hf_token_struct[start_token_index]["label"] = annotation["label"]
        elif len(annotation["ids"]) > 1:
            start_token_index = next(
                (
                    index
                    for (index, d) in enumerate(hf_token_struct)
                    if d["id"] == annotation["ids"][0]
                ),
                None,
            )
            hf_token_struct[start_token_index]["label"] = annotation["label"]
            for i, id in enumerate(annotation["ids"][1:], 1):
                target_token_index = next(
                    (
                        index
                        for (index, d) in enumerate(hf_token_struct)
                        if d["id"] == annotation["ids"][i]
                    ),
                    None,
                )
                hf_token_struct[start_token_index]["text"] += " {}".format(
                    hf_token_struct[target_token_index]["text"]
                )
                hf_token_struct[start_token_index]["words"].extend(
                    hf_token_struct[target_token_index]["words"]
                )
                hf_token_struct[start_token_index]["box"][2] = hf_token_struct[
                    start_token_index
                ]["words"][-1]["box"][2]
                hf_token_struct[start_token_index]["box"][3] = hf_token_struct[
                    start_token_index
                ]["words"][-1]["box"][3]
                remove_ids.append(id)
        else:
            logger.error("merge_data error")
    # skip ids of remove item
    for item in hf_token_struct:
        if item["id"] in remove_ids:
            continue
        else:
            new_hf_token_struct.append(item)
    # renew id
    for index, item in enumerate(new_hf_token_struct):
        item["id"] = index

    return {"form": new_hf_token_struct}


def match_mongodb_token(ls_annotated_result, db):
    for task in ls_annotated_result["data"]:
        task_id = task["meta"]["task_id"]
        task_text = task["data"]["text"]
        annotations = get_annotations(task, task_text)
        tokens, bboxs = find_token_bbox_from_collection(
            db, task_id, LS_QA_PII_PROJECT_ID
        )
        image_metadata = find_image_metadata_from_collection(
            db, task_id, LS_QA_PII_PROJECT_ID
        )
        result = merge_data(tokens, bboxs, annotations)
        with open(
            "annotations/{}.json".format(
                image_metadata["filename"]
                .split("/")[-1]
                .replace(".jpg", "")
                .replace(".png", "")
            ),
            "w",
        ) as fp:
            json.dump(result, fp)


def export_huggingface_dataset_fomat(result, filenames):
    for item, filename in zip(result, filenames):
        with open(
            "annotations/{}.json".format(
                filename["filename"]
                .split("/")[-1]
                .replace(".jpg", "")
                .replace(".png", "")
            ),
            "w",
        ) as fp:
            json.dump(result, fp)


def main(args):
    ls_annotated_result = export_data(LS_QA_ENDPOINT, LS_QA_PII_PROJECT_ID, LS_QA_TOKEN)
    print(ls_annotated_result)
    db = connect_mongo_dbs(args.mongo_host, args.mongo_databases)
    match_mongodb_token(ls_annotated_result, db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
