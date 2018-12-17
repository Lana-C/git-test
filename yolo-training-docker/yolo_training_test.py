

import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

# yolov2-new was created by modifying yolov2-new.cfg as per instructions in darkflow repo 

options = {"model": "cfg/yolov2-new.cfg", 
           "load": "bin/yolov2.weights",
           "epoch": 5,
           "train": True,
           "annotation": "./test/training/annotations/",
           "dataset": "./test/training/images/",
           "labels": "labels_test.txt"}

tfnet = TFNet(options)

tfnet.train()  # creates automatically a ckpt folder containing the checkpoint

tfnet.savepb()  # creates the built_graph folder 





