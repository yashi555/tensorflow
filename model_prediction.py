import cv2
from matplotlib import pyplot as plt

from darkflow.net.build import TFNet

options = {
    'model': 'cfg/yolov2-voc.cfg',
    'load': 100,
    'threshold': 0.3,
    'backup': 'ckpt/'

}
tfnet2 = TFNet(options)


# Load the checkpoint
tfnet2.load_from_ckpt()


#Predicting on an image from the custom dataset
original_img = cv2.imread("/home/yashita/projects/darkflow-master/sample_img/sample_eagle.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
result = tfnet2.return_predict(original_img)
print(result)

for i in range(0, len(result)):
    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
    label = result[i]['label']# add the box and label and display it
    img = cv2.rectangle(original_img, tl, br, (0, 255, 0), 7)
    img = cv2.putText(original_img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    plt.imshow(img)
plt.show()