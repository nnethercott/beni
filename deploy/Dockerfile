# gcp compat image
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310 as base
#FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime as base

WORKDIR /app 

COPY deploy/requirements.txt .
RUN pip3 install -r requirements.txt 

RUN apt-get update && \
    apt-get install -y uuid-runtime net-tools

# gcp vm tries to use XLA runtime instead of CUDA
RUN pip3 uninstall -y torch-xla

COPY src/vlm src/vlm
COPY scripts scripts 

# torch disttributed env vars 
ENV NNODES=1
ENV NPROC_PER_NODE=4

RUN cat <<EOF > /app/probe.py
import torch
import os 
from deepspeed import get_accelerator

if __name__ == "__main__":
    model_dir = os.getenv("AIP_MODEL_DIR", "/tmp")

    gs_prefix = 'gs://'
    gcsfuse_prefix = '/gcs/'
    if model_dir.startswith(gs_prefix):
        model_dir = model_dir.replace(gs_prefix, gcsfuse_prefix)
        model_dir = model_dir.replace("/model", "")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(model_dir)

    device_count = torch.cuda.device_count()

    with open(os.path.join(model_dir, "tmp.txt"), 'w') as f:
        f.write(f"num gpus: {device_count}\n")
        f.write(f"deepspeed fp16 support: {get_accelerator().is_fp16_supported()}\n")

    print(device_count)
EOF


COPY ./deploy/entrypoint.sh . 
RUN chmod +x /app/entrypoint.sh

#CMD ["python3", "/app/probe.py"]
ENTRYPOINT ["/app/entrypoint.sh"]

