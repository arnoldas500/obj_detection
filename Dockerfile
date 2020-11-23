#create image using:
#cd /raid/arnold/clouds_detection/modelsV2/
#docker build -f research/object_detection/dockerfiles/tf1/Dockerfile -t od_tf1:v2 .
#run using
#docker run -it od_tf1:v2

FROM tensorflow/tensorflow:1.15.2-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow . /home/tensorflow/models

# Copy data from host relative path to image abs path
# COPY -r data images training ssd_mobilenet_v1_coco_11_06_2017 /home/arnold/clouds_detection/modelsV2/research/object_detection/legacy

# use ADD instead of COPY. Suppose you want to copy everything in directory src from host to directory dst from container:
# ADD src dst
ADD research/object_detection/legacy /home/tensorflow/models/research/object_detection/legacy

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf1/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN python -m pip install --user .

ENV TF_CPP_MIN_LOG_LEVEL 3
