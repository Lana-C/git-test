
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
# Create additional directories in darkflow directory: bin -> holds the weights for the model, data -> holds input images to be labeled, results -> holds the model output (labeled images)

RUN cd "/" && \
	git clone https://github.com/thtrieu/darkflow.git &&\
	cd darkflow && \
	python3 setup.py build_ext --inplace && \
	mkdir bin && \
	mkdir data && \
	mkdir results && \
	cd "/"

# Add weights file and the sample notebook to run object detection to the darkflow folder

ADD yolov2.weights /darkflow/bin
ADD YOLO_test.ipynb /darkflow
ADD yolo.py /darkflow

# Make darkflow the working directory when image is run 

WORKDIR /darkflow 






