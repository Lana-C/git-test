# YOLO Training - Docker image

Provided Dockerfile can be used to build a Docker image for training of YOLO object detection algorithm using [darkflow](https://github.com/thtrieu/darkflow) implementation. It is based on the tensorflow image. 

### Training setup
Setting up training on a new annotated dataset involves several preparation steps (detailed instructions can also be found in the darkflow repository). Here we assume that we are initializing training of the CNN from existing configurations and pre-trained weights available [here](https://pjreddie.com/darknet/yolo/).

1.  If needed, depending on the number of classes that should be trained for, existing configuration files (.cfg) should be adjusted as per instructions in the darkflow repository: make a copy of the original `.cfg` file you choose and change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our example, we are going to use the `yolov2.cfg` to create  `yolov2_new.cfg` and set classes to 3.
    
    ```python
    ...
    [region]
    anchors =  0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
    bias_match=1
    classes=3
    coords=4
    num=5
    softmax=1
    jitter=.3
    rescore=1
    ...
    ```
    Next, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and
    classes are 3 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    ```python
    ...
    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40
    activation=linear    
    ...
    ```
    Save the `.cfg` file in the `cfg/` folder.
    
2.  Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you
set in `.cfg` file). In our case, we will create a custom labels file with 3 labels which can then be specified in the options for training. Save the labels file in the `lbl/` folder. 

3. Images and annotations to be trained on should be saved in the `data/` folder under `data/images` and `data/annotations`. For the example training run we use the images (only 2 (!) so disregard the results) and annotations provided in the darkflow repository under `darkflow/test/training/images/` and `darkflow/test/training/annotations/`.

### Build the Docker image
To build the image, add
* `cfg/` folder with the configuration - using `yolov2_new.cfg` as example (Note: some configurations are already available in `/darkflow/cfg/`)
* `lbl/` folder with the labels `.txt` file
* `wgt/` folder with weights - using `yolov2.weights` as example (download [here](https://pjreddie.com/media/files/yolov2.weights))
* `data/` folder with images and annotations
* `YOLO_training_test.ipynb` sample Jupyter notebook for interactive execution of training
* `yolo_training_test.py` sample Python script for non-interactive execution of training

to the folder where Dockerfile is located and run `docker build -t yolotrainimage .`.

### Run the Docker container

#### Using Jupyter notebook
The Docker containter based on the `yolotrainimage` can be started by running `docker run -it -p 8888:8888 yolotrainimage` and going to `localhost:8888` in the browser to open the `YOLO_training_test.ipynb` Jupyter notebook for testing the algorithm - options need to be set accordingly if differing from the given example. 

#### Using Python script
The Docker containter based on the `yolotrainimage` can be started by running `docker run -it yolotrainimage python yolo_training_test.py ` - options in the script need to be set accordingly if differing from the given example. 

After completed training, results can be found in the `ckpt/` and `built_graph/` directories.     
