from transformers import (
    BitsAndBytesConfig, 
    AutoProcessor,
    GenerationConfig,
    TextStreamer
)
from peft import PeftModel

from dataclasses import (
    dataclass,
    asdict,
)

import sys
import time
import uuid
import requests 
from PIL import Image  
import io 
import base64
import torch 
from flask import Flask, request, jsonify
import gc
import os 

sys.path.insert(1, "../src/")
from model import Beni, BeniConfig
from checkpointing import load_model


# TODO: pass this to the llm later 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)

cfg = BeniConfig(
    vision_name_or_path = "google/siglip-so400m-patch14-384",
    #text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    text_name_or_path = "HuggingFaceTB/SmolLM-360M-Instruct",
    vision_cls = "SiglipVisionModel",
    vision_processor_cls = "SiglipImageProcessor",
    r = 9,
    freeze = True,
    use_cls = True,
    attn_implementation="sdpa",
    img_size = 384,
    feature_select_index = -1,
)

# defined globally
model = Beni(cfg)
model = load_model(model, ckpt_dir=os.getenv("MODEL_CHECKPOINT", None))
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
    msg = [m['content'] for m in messages if m['role'] == 'user'][0]

    text = ''
    url = None 

    if isinstance(msg, list):
        for d in msg:
            if d['type'] == 'text':
                text = d['text']
            else:
                url = d['image_url']['url']
    else:
        text = msg

    if len(text) == 0:
        text = None
    if text is not None:
        #text = text + tokenizer.eos_token + '\n' # custom template i made
        convo = [{'role': 'user', 'content': text}]
        text = tokenizer.apply_chat_template(convo, tokenize=False)  # tokenizer's built-in chat template

    return text, url




def parse_request(request, tokenizer):
  parsed = {}
  msg, img_url = apply_chat_template(request["messages"], tokenizer)
   
  if img_url is not None:
      if img_url.startswith("http"):
          image = Image.open(requests.get(img_url, stream=True).raw)
      elif img_url.startswith("data"):
          dat = img_url.split(',')[1]
          image = Image.open(io.BytesIO(base64.b64decode(dat)))
      else:
          image = Image.open(img_url)

      image = image.convert("RGB")

  else:
      image = None

  return {'prompt': msg, 'image': image} 

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


def chat_completion(model, prompt, image = None, **generation_kwargs):
    with torch.no_grad():
        inputs = {}
        if prompt is not None:
            print('theres text')
            inputs = model.tokenizer(prompt, return_tensors = 'pt').to(device).to(torch.float16)

        if image is not None:
            print('theres image')
            inputs['images'] = [image,]


        out = model.generate(
            **inputs, 
            **generation_kwargs,
        )

        torch.cuda.empty_cache()
        generated = model.tokenizer.batch_decode(out, skip_special_tokens = True)[0]

    # metrics 
    usage = {}
    #if hasattr(inputs, 'input_ids'):
    #    usage['prompt_tokens'] = inputs['input_ids'].shape[1] + model.n_img_prompt_tokens
    #    usage['completion_tokens'] = out.shape[1] - inputs['input_ids'].shape[1]
    #    usage['total_tokens'] = out.shape[1]
    #else:
    #    usage['prompt_tokens'] = model.n_img_prompt_tokens
    #    usage['completion_tokens'] = out.shape[1] 
    #    usage['total_tokens'] = out.shape[1]
    usage['prompt_tokens'] = 42
    usage['completion_tokens'] = 42
    usage['total_tokens'] = 42

    # choices 
    choices = [{
      "finish_reason": "stop", # ad hoc (maybe recoup from model generate call)
      "index": 0,
      "message": {
        "content": generated.strip(), # only field we care about 
        "role": "assistant",
      },
      "logprobs": None,
    }]

    # final 
    openai_response = {
      "choices": choices,
      "created": int(time.time()),
      "id": str(uuid.uuid4()),
      "model": "beni", # model.config._name_or_path.split("/")[-1], #TODO: add this fr 
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

      prompt = parsed['prompt']
      image = parsed['image']
        
      return chat_completion(model, prompt, image, **generation_kwargs)
      
    app.run(host='0.0.0.0', port = '5001', debug=False)
