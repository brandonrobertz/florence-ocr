#!/usr/bin/env python
import os
import json
import sys

from PIL import Image, UnidentifiedImageError
from florence import run_florence


ROOT_DIRECTORY = "/home/brandon/src/openmeasures-research-tools"

INPUT_JSON=sys.argv[1]


records_getter = None
if INPUT_JSON == "--":
    records_getter = (json.loads(line) for line in sys.stdin.readlines())
elif INPUT_JSON.endswith(".jsonl"):
    with open(INPUT_JSON, "r") as f:
        records_getter = (json.loads(line) for line in f.readlines())
else:
    with open(INPUT_JSON, "r") as f:
        records_getter = json.load(f)


for record in records_getter:
    try:
        image_path = record["downloaded_media_path"]
    except KeyError:
        print(json.dumps(record))
        continue

    # see if we already did this within this JSON
    if "media_information" in record:
        print(json.dumps(record))
        continue

    mediamimetype = record.get("mediamimetype")
    if not mediamimetype or not mediamimetype.startswith("image"):
        print(json.dumps(record))
        continue

    joined_image_path = os.path.join(ROOT_DIRECTORY, image_path)

    real_image_path = os.path.realpath(joined_image_path)
    media_path, media_filename = os.path.split(real_image_path)
    media_basename, _ = os.path.splitext(media_filename)
    processed_media_path = os.path.join(media_path, f"{media_basename}-florence.json")
    if os.path.exists(processed_media_path):
        with open(processed_media_path, "r") as f:
            record["media_information"] = json.load(f)
        print(json.dumps(record))
        continue

    try:
        image = Image.open(joined_image_path)
    except UnidentifiedImageError:
        print(json.dumps(record))
        continue

    record["media_information"] = {}

    prompt = "<CAPTION>"
    response = run_florence(image, prompt)
    record["media_information"]["caption"] = response[prompt].strip()

    prompt = "<DENSE_REGION_CAPTION>"
    response = run_florence(image, prompt)
    record["media_information"]["description"] = "; ".join(
        set(response[prompt]["labels"])
    ).strip()

    prompt = "<OCR_WITH_REGION>"
    response = run_florence(image, prompt)
    record["media_information"]["text"] = " ".join(
        response[prompt]["labels"]
    ).strip().replace("</s>", "")

    with open(processed_media_path, "w") as f:
        json.dump(record["media_information"], f)

    print(json.dumps(record))
