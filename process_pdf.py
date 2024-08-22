import json
import os
import shutil
import sys

from pdf2image import convert_from_path
from PIL import Image

from florence import run_florence


def pdf_to_images(pdf_path, output_folder, dpi=300):
    # Create output directory if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    # Convert PDF to images
    print(f"Splitting file {pdf_path} at {dpi} DPI to {output_folder}", file=sys.stderr)
    images = convert_from_path(
        pdf_path, dpi=dpi, output_folder=".pdf_tmp",
        thread_count=os.cpu_count()-1,
        paths_only=True,
        fmt="TIFF"
    )

    return sorted(images)


if __name__ == "__main__":
    page_images = pdf_to_images(sys.argv[1], ".pdf_tmp", dpi=72)

    for page_image in page_images:
        page = int(page_image.split("-")[-1].split(".")[0])
        print(f"Loading page {page} {page_image}", file=sys.stderr)
        image = Image.open(page_image)

        prompt = "<OCR_WITH_REGION>"
        response = run_florence(image, prompt)
        text = "\n".join(
            response[prompt]["labels"]
        ).strip().replace("</s>", "")

        page_text = {
            "page": page,
            "text": text,
        }
        print(json.dumps(page_text))
