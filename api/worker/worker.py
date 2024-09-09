from transformers import (
    BitsAndBytesConfig,
)

from dataclasses import (
    dataclass,
    asdict,
)
import json

import sys
import time
import uuid
import requests
from PIL import Image
import io
import base64
import torch
from flask import Flask, request
import os

sys.path.insert(1, "./src/")
from model import Beni, BeniConfig
from checkpointing import load_model


# TODO: pass this to the llm later
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)
CKPT_DIR = os.getenv("MODEL_CHECKPOINT", "model")

with open(f"{CKPT_DIR}/../model_config.json", "r") as f:
    config_dict = json.loads(f.read())
    MODEL_CONFIG = BeniConfig.from_dict(config_dict)


# defined globally
model = Beni(MODEL_CONFIG, os.getenv("HF_TOKEN"))
model = load_model(model, ckpt_dir=CKPT_DIR, trainable=False)
device = "cuda"
model.to(device)
model.eval()

print(model)

torch.cuda.empty_cache()


"""
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "model-name",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What'\''s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
"""


def apply_chat_template(messages, tokenizer):
    # assume single turn convo (update later)
    msg = [m["content"] for m in messages if m["role"] == "user"][0]

    text = ""
    url = None

    if isinstance(msg, list):
        for d in msg:
            if d["type"] == "text":
                text = d["text"]
            else:
                url = d["image_url"]["url"]
    else:
        text = msg

    text = MODEL_CONFIG.instruction_template.format(instruction=text)

    return text, url


def parse_request(request, tokenizer):
    msg, img_url = apply_chat_template(request["messages"], tokenizer)

    if img_url is not None:
        if img_url.startswith("http"):
            image = Image.open(requests.get(img_url, stream=True).raw)
        elif img_url.startswith("data"):
            dat = img_url.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(dat)))
        else:
            image = Image.open(img_url)

        image = image.convert("RGB")

    else:
        image = None

    return {"prompt": msg, "image": image}


@dataclass
class GenerationArguments:
    min_new_tokens: int = 2
    max_new_tokens: int = 128
    temperature: float = 0.8
    do_sample: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1
    top_k: int = 32


def parse_into_generation_kwargs(req):
    gen_kwargs = asdict(GenerationArguments())
    for k, v in req.items():
        if hasattr(gen_kwargs, k):
            gen_kwargs[k] = v
    return gen_kwargs


def chat_completion(model, prompt, image=None, **generation_kwargs):
    with torch.no_grad():
        inputs = {}
        if prompt is not None:
            print("theres text")
            inputs = model.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)

        if image is not None:
            print("theres image")
            inputs["images"] = [
                image,
            ]

        out = model.generate(
            **inputs,
            **generation_kwargs,
        )

        torch.cuda.empty_cache()
        generated = model.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    # metrics
    usage = {}
    usage["prompt_tokens"] = 42
    usage["completion_tokens"] = 42
    usage["total_tokens"] = 42

    # choices
    choices = [
        {
            "finish_reason": "stop",  # ad hoc (maybe recoup from model generate call)
            "index": 0,
            "message": {
                "content": generated.strip(),  # only field we care about
                "role": "assistant",
            },
            "logprobs": None,
        }
    ]

    # final
    openai_response = {
        "choices": choices,
        "created": int(time.time()),
        "id": str(uuid.uuid4()),
        "model": CKPT_DIR.split("/")[-2],  # stablelm-2-1_6b-chat
        "object": "chat_completion",
        "usage": usage,
    }

    return openai_response


if __name__ == "__main__":
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    def chat():
        data = request.json

        # parse
        parsed = parse_request(data, model.tokenizer)
        generation_kwargs = parse_into_generation_kwargs(data)

        prompt = parsed["prompt"]
        image = parsed["image"]

        print(prompt)

        return chat_completion(model, prompt, image, **generation_kwargs)

    app.run(host="0.0.0.0", port=5001, debug=False)
