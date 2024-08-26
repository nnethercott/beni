import spacy 
from concurrent.futures import ThreadPoolExecutor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
import requests
from PIL import Image, ImageDraw
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")





# owl 
processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")

url = "https://d1hw6n3yxknhky.cloudfront.net/054711517_iconm.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

doc = nlp("A sleek modern desk lamp with a metallic finish and a curved neck is positioned on a reflective surface. The lamp's head is angled downwards, casting a shadow on the surface. In front of the lamp, there is a clear glass vase with a delicate, twisted design. The vase is also casting a shadow on the surface, and the scene is set against a plain, light-colored background.")
texts = [[c.text for c in doc.noun_chunks][:3]]

inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])

# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
