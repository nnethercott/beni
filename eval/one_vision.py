from huggingface_hub import HfFileSystem
import torch

# ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", "GEOS(MathV360K)")

# https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system
fs = HfFileSystem()
files = fs.ls("datasets/lmms-lab/LLaVA-OneVision-Data", detail=True)

# TODO: list these


import torch.nn.functional as F

patches = torch.rand(2, 27 * 27, 1152)
patches = patches.view(2, 1152, 27, 27)
new_patches = F.interpolate(patches, size=(6, 6), mode="bilinear", align_corners=False)
new_patches = new_patches.view(2, -1, 1152)
