from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()

print(model.hf_device_map)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://www.akc.org/wp-content/uploads/2018/05/Three-Australian-Shepherd-puppies-sitting-in-a-field.jpg",
            },
            {
                "type": "text",
                "text": "give me the bounding box coordinates for the dogs in the image. coordinates should be min-maxed scaled to the range [0,1] in the format [x1,y1,x2,y2]",
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

# image_inputs = [im.resize((384, 384)) for im in image_inputs]

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)


import re

pattern = r"\(\s*\d\.\d{1,2},\s*\d\.\d{1,2},\s*\d\.\d{1,2},\s*\d\.\d{1,2}\s*\)"
matches = re.findall(pattern, output_text)
coords = [re.findall(r"\d\.\d{1,2}", match) for match in matches]
coords = [[float(num) for num in group] for group in coords]


from PIL import ImageDraw

img = image_inputs[0]
w, h = img.size
coords = [[w * i if e % 2 == 0 else h * i for e, i in enumerate(c)] for c in coords]
draw = ImageDraw.Draw(img)
for c in coords:
    draw.rectangle(c, outline="red", width=3)

img.save("/home/nnethercott/test.jpg")
