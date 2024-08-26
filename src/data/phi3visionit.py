from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
import io

model_id = "microsoft/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager",
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
url = "https://akcdn.detik.net.id/visual/2020/02/28/62a378a9-427e-435d-8f0e-3137c4700804_169.jpeg?w=650"
images.append(Image.open(io.BytesIO(requests.get(url, timeout=3).content)))
placeholder += f"<|image_1|>\n"

prompt = (
    placeholder
    + "describe this image in 1 or 2 sentences. Make sure to use the following words in your description: 'people', 'mask'."
)

messages = [
    {"role": "user", "content": prompt},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
