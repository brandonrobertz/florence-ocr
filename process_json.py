#!/usr/bin/env python
import os
import json
import hashlib
import sys

from PIL import Image, UnidentifiedImageError
import requests

from florence import run_florence


DOWNLOAD_PATH = "/home/brandon/src/openmeasures-research-tools/media_telegram"

INPUT_JSON=sys.argv[1]


def generate_file_hash(content):
    return hashlib.sha256(content).hexdigest()


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
    media_url_or_path = record.get(
        "downloaded_media_path", record.get("media_url")
    )
    if not media_url_or_path:
        print(json.dumps(record))
        continue

    # see if we already did this within this JSON
    if "media_information" in record:
        print(json.dumps(record))
        continue

    print("media_url_or_path:", media_url_or_path, file=sys.stderr)

    mediamimetype = record.get("mediamimetype", record.get("mime_type"))
    if not mediamimetype or not mediamimetype.startswith("image"):
        print(json.dumps(record))
        continue

    # TODO: Parse the download media path, if it's an openmeasures file we have
    # to download it sadly
    if media_url_or_path.startswith("https://"):
        print("Downloading", media_url_or_path, file=sys.stderr)
        response = requests.get(media_url_or_path)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            hash = generate_file_hash(response.content)
            attachment_path = os.path.join(DOWNLOAD_PATH, f"hash-{hash}.pdf")
            if not os.path.exists(attachment_path):
                print("Writing PDF to", attachment_path, file=sys.stderr)
                with open(attachment_path, 'wb') as f:
                    f.write(response.content)
            media_url_or_path = attachment_path

    if not os.path.exists(media_url_or_path):
        print("File doesn't exist!", media_url_or_path, file=sys.stderr)
        print(json.dumps(record))
        continue

    # see if we already did this within this JSON
    if "media_information" in record:
        print(json.dumps(record))
        continue

    print(f"Running florence on media mime type {mediamimetype}...",
          file=sys.stderr)

    try:
        image = Image.open(media_url_or_path).convert("RGB")
    except UnidentifiedImageError as e:
        print(f"Error loading file {media_url_or_path}. Error: {e}",
              file=sys.stderr)
        print(json.dumps(record))
        continue

    record["media_information"] = {}

    prompt = "<CAPTION>"
    response = run_florence(image, prompt)
    caption = response[prompt].strip()
    print("Caption", caption, file=sys.stderr)
    record["media_information"]["caption"] = caption

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

    print(json.dumps(record))
