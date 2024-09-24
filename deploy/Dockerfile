FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime as base 

WORKDIR /app 

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu124

COPY deploy/requirements.txt .
RUN pip3 install -r requirements.txt 

COPY src/vlm vlm

# torch disttributed env vars 
ENV NNODES=1
ENV NPROC_PER_NODE=4

RUN cat <<EOF > /app/script.py
import torch
if __name__ == "__main__":
    print(torch.cuda.device_count())
EOF

# RUN apt-get update && apt-get -y install locales
# ENV LANG en_US.UTF-8

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["torchrun --nnodes=$NNODES --nproc-per-node=$NPROC_PER_NODE /app/vlm/main.py"]
#CMD ["python", "/app/script.py"]
