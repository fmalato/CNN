import cv2
import os
import glob

img_path = "dataSet/images/stop_orig/"
save_path = "dataSet/images/stop/"
num = 0

path = os.path.join(img_path, '*.jpg')
for fl in glob.glob(path):
    num += 1
    image = cv2.imread(fl)
    image = cv2.resize(image, (128, 96))
    cv2.imwrite(('dataSet/images/stop/%d.jpg' % num), image)
