
# Base image is a tensorflow image 
# python, pip, jupyter, numpy are already installed in the base image - jupyter is exposed at port 8888

FROM tensorflow/tensorflow:latest-py3  

# Install other dependencies
RUN apt-get update && apt-get install -y \
	git \
	wget \
	libsm6 \
	libxext6 \
	libxrender-dev

RUN pip install cython
RUN pip install opencv-python

# Clone the darkflow git repository and install
# Create additional directories in darkflow directory: bin -> holds the weights for the model, data -> holds input images and annotations

RUN cd "/" && \
	git clone https://github.com/thtrieu/darkflow.git &&\
	cd darkflow && \
	python3 setup.py build_ext --inplace && \
	mkdir bin && \
	mkdir data && \
	cd "/"

# Add configuration, labels and weights folder as well as sample notebook and python script for training to the darkflow folder

ADD cfg/ /darkflow/cfg
ADD lbl/ /darkflow
ADD wgt/ /darkflow/bin
ADD YOLO_training_test.ipynb /darkflow
ADD yolo_training_test.py /darkflow

# Add checkkpoint folder
RUN mkdir /darkflow/ckpt

# Make darkflow the working directory when image is run 

WORKDIR /darkflow 






