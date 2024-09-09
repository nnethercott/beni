FROM python:3.10-bookworm as builder 

RUN pip install -U pip 

WORKDIR /home
COPY ./chatbot/requirements.txt .
RUN pip install -r requirements.txt

COPY ./chatbot .

FROM builder as runtime

CMD ["python3", "chatbot.py"]
# CMD ["tail", "-f", "/dev/null"]

