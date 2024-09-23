from huggingface_hub import HfFileSystem
import torch

# ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", "GEOS(MathV360K)")

# https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system
fs = HfFileSystem()
files = fs.ls("datasets/lmms-lab/LLaVA-OneVision-Data", detail=True)
