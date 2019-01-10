

import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

# yolov2-new was created by modifying yolov2-new.cfg as per instructions in darkflow repo 

options = {"model": "cfg/yolov2-voc2012.cfg", 
           "load": "bin/yolov2.weights",
           "epoch": 2,
           "train": True,
           "annotation": "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/",
           "dataset": "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/",
           "labels": "labels-voc2012.txt"}

tfnet = TFNet(options)

tfnet.train()  # creates automatically a ckpt folder containing the checkpoint

tfnet.savepb()  # creates the built_graph folder 





