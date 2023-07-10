FROM tensorflow/tensorflow:1.15.0-gpu-py3

COPY . ./

RUN pip install -U -r requirements.txt

