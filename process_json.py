#!/usr/bin/env python
import os
import json
import sys

from PIL import Image, UnidentifiedImageError
from florence import run_florence


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

    joined_image_path = os.path.join("/home/brandon/src/openmeasures-research-tools", image_path)

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
    record["media_information"]["text"] = " ".join(response[prompt]["labels"]).strip()

    print(json.dumps(record))
