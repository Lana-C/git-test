
# Base image is a tensorflow image 
# python, pip, jupyter, numpy are already installed in the base image - jupyter is exposed at port 8888, tensorboard port 6006

FROM tensorflow/tensorflow:latest-gpu-py3

# FROM tensorflow/tensorflow:latest-py3  # cpu version

# Install other dependencies
RUN apt-get update && apt-get install -y \
	git \
	wget \
	libsm6 \
	libxext6 \
	libxrender-dev

RUN pip install cython
RUN pip install opencv-python
RUN pip install keras

RUN cd "/" && \
	mkdir mnist && \
	cd "/mnist" && \
	mkdir runs && \
	cd "/mnist/runs" && \
	mkdir ckpt && \
	cd "/"

ADD mnist_full.py /mnist

WORKDIR /mnist 






