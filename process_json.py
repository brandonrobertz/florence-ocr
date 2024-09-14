#!/usr/bin/env python
import os
import json
import sys

from PIL import Image, UnidentifiedImageError
from florence import run_florence


ROOT_DIRECTORY = "/home/brandon/src/openmeasures-research-tools"

INPUT_JSON=sys.argv[1]


def log(msg):
    sys.stderr.write(f"{msg}\n")


def record_getter():
    if INPUT_JSON == "--":
        for line in sys.stdin.readlines():
            yield json.loads(line)
    elif INPUT_JSON.endswith(".jsonl"):
        with open(INPUT_JSON, "r") as f:
            for line in f.readlines():
                yield json.loads(line)
    else:
        with open(INPUT_JSON, "r") as f:
            for record in json.load(f):
                yield record


for record in record_getter():
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

    print(f"Running florence on media mime type {mediamimetype}...",
          file=sys.stderr)

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
