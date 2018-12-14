# Import needed libraries
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt


# Set the model options
options = { 
    "model": "cfg/yolo.cfg",
    "load": "bin/yolov2.weights",
    "threshold": 0.3
}

# Instantiate the model
tfnet=TFNet(options)

:
# Read in an image (specify the path manually either from sample_img or data folders) and run labeling 
img = cv2.imread("sample_img/sample_person.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
result

# Create bounding boxes around recognized objects and save the output image into result folder (specify the name of output image)
for r in result:
    tl = (r["topleft"]["x"],r["topleft"]["y"])
    br = (r["bottomright"]["x"],r["bottomright"]["y"])
    label = r["label"]
    conf = round(r["confidence"],2)
    
    im = cv2.rectangle(img, tl, br, (0, 255, 0), 3)
    im = cv2.putText(img, label+str(conf), tl, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 2)

plt.imsave('results/result.jpg', im)
