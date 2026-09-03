FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt update && apt -y install awscli

RUN pip install -r requirements.txt

CMD python app.py
