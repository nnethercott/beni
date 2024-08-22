from datasets import load_dataset
from PIL import Image
import io

def bytes_to_pil(samples):
    images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in samples['images']]
    samples['pil'] = images
    return samples

dataset = load_dataset("nnethercott/Recap-DataComp-129K", split="train[:1001]")

# load PIL images from bytes 
dataset = dataset.map(bytes_to_pil, batched=True, remove_columns = ["images"]) 
dataset = dataset.rename_columns({'pil': 'images'})

