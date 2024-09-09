FROM python:3.10-bookworm as builder 

RUN pip install -U pip 
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

WORKDIR /home
COPY ./api/worker/requirements.txt .
RUN pip install -r requirements.txt

COPY api/worker/ .
COPY src ./src

FROM builder as runtime

#CMD ["python", "worker.py"]
CMD ["tail", "-f", "/dev/null"]

