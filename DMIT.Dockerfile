FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/AntoineMahe/Deep-Model-Identification-Toolbox

RUN pip install -U -r Deep-Model-Identification-Toolbox/requirements.txt

