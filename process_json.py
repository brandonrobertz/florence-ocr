#!/usr/bin/env python
import json
import sys

from PIL import Image
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
    image_path = record["downloaded_media_path"]
    image = Image.open(image_path)

    record["media_information"] = {}

    prompt = "<CAPTION>"
    response = run_florence(prompt)
    record["media_information"]["caption"] = response[prompt].strip()

    prompt = "<DENSE_REGION_CAPTION>"
    response = run_florence(prompt)
    record["media_information"]["description"] = "; ".join(response[prompt]["labels"])

    prompt = "<OCR_WITH_REGION>"
    response = run_florence(prompt)
    record["media_information"]["text"] = " ".join(response[prompt]["labels"]).strip()

    print(json.dumps(record))
