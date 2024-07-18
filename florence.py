#!/usr/bin/env python
#!pip install 'flash_attn==2.6.1' 'transformers==4.34.0' 'torch==2.0.1'
# unset TRANSFORMERS_CACHE
# export HF_HOME=./.huggingface
import sys

import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def run_florence(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    # print(parsed_answer)
    return parsed_answer


# # Here are the tasks Florence can run
# ## Caption
# prompt = "<CAPTION>"
# run_example(prompt)
#
# ## Detailed Caption
#
# prompt = "<DETAILED_CAPTION>"
# run_example(prompt)
#
# ## More Detailed Caption
#
# prompt = "<MORE_DETAILED_CAPTION>"
# run_example(prompt)
#
# ##Caption to Phrase Grounding
#
# caption to phrase grounding task requires additional text input, i.e. caption.
#
# Caption to phrase grounding results format: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}
#
# task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
# results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")
#
# ## Object Detection
#
# OD results format: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', 'label2', ...]} }
#
# prompt = "<OD>"
# run_example(prompt)
#
# ## Dense Region Caption
#
# Dense region caption results format: {'<DENSE_REGION_CAPTION>' : {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', 'label2', ...]} }
#
# prompt = "<DENSE_REGION_CAPTION>"
# run_example(prompt)
#
# ## Region proposal
#
# Dense region caption results format: {'<REGION_PROPOSAL>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}
#
# prompt = "<REGION_PROPOSAL>"
# run_example(prompt)
#
# ## OCR
#
# prompt = "<OCR>"
# run_example(prompt)
#
# ## OCR with Region
#
# OCR with region output format: {'<OCR_WITH_REGION>': {'quad_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4], ...], 'labels': ['text1', ...]}}
#
# prompt = "<OCR_WITH_REGION>"
# run_example(prompt)

if __name__ == "__main__":
    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    # image = Image.open(requests.get(url, stream=True).raw)

    print(f"Loading image {sys.argv[1]}...", end=" ")
    image = Image.open(sys.argv[1])
    print("Done")

    prompt = "<CAPTION>"
    response = run_florence(image, prompt)
    print("Caption:", response[prompt].strip())

    prompt = "<DENSE_REGION_CAPTION>"
    response = run_florence(image, prompt)
    print("Regions:", "; ".join(response[prompt]["labels"]))

    prompt = "<OCR_WITH_REGION>"
    response = run_florence(image, prompt)
    print("Text:", " ".join(response[prompt]["labels"]).strip())
