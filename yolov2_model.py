# Processing an image file using Yolo
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt #%config InlineBackend.figure_format = 'svg'


options = {
 'model': 'cfg/yolov2-voc.cfg',
 'load': 'bin/yolov2-voc.weights',
 'threshold': 0.3
}
tfnet = TFNet(options)

# Loading the image and identifying the objects in the image using Yolo
img = cv2.imread("/home/yashita/Downloads/horse.jpeg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# use YOLO to predict the image
result = tfnet.return_predict(img)
print(result)


# Display the class and the bounding box on the image
# pull out some info from the results
for i in range(0, len(result)):
    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
    label = result[i]['label']# add the box and label and display it
    img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    plt.imshow(img)
plt.show()

# options = {"model": "cfg/yolov2-voc.cfg",
#            "load": "bin/yolov2-voc.weights",
#            "batch": 2,
#            "epoch": 5,
#            "train": True,
#            "annotation": "new_data/annots/",
#            "dataset": "new_data/images/"}
#
# from darkflow.net.build import TFNet
# tfnet = TFNet(options)
# tfnet.train()
#tfnet.savepb()