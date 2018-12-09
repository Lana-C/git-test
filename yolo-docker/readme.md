# YOLO - Docker image

Provided Dockerfile can be used to build a Docker image for object detection on images using [darkflow](https://github.com/thtrieu/darkflow) implementation of the YOLO algorithm. It is based on the tensorflow image (CPU). 

### Build the Docker image
To build the image, add the yolov2.weights file (download [here](https://pjreddie.com/media/files/yolov2.weights)) and YOLO_test.ipynb Jupyter notebook to the folder where Dockerfile is located and run `docker build -t yoloimage .`.

### Run the Docker container
The Docker containter based on the `yoloimage` can be started by running `docker run -it -p 8888:8888 yoloimage` and going to `localhost:8888` in the browser to open the YOLO_test.ipynb Jupyter notebook for testing the algorithm - sample images provided with darkflow implementation are located in `\darkflow\sample_img` folder. Labeled images will be saved in `\darkflow\results` folder. 

### Mounting volumes for data and results 
In order to run object detection on any image, one can bind mount a folder with images to be labeled on the host machine into the Docker container using `docker run -it -v <data-path-on-host-machine>:/darkflow/data -v <results-path-on-host-machine>:/darkflow/results -p 8888:8888  yoloimage`. The image path in YOLO_test.ipynb needs to be changed accordingly. After algorithm is run, labeled images will be saved in the host machine folder mounted into the `/darkflow/results` folder in the Docker containter.     
