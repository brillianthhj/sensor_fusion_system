import cv2 
import numpy as np
from matplotlib import pyplot as plt 

# 이미지 로드
image = cv2.imread('/home/nvidia/hyejin_ws/src/sensor_fusion_system/image/undistort_img.png')

plt.imshow(image)
plt.show()

