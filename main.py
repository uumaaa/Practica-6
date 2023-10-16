import cv2
import matplotlib.pyplot as plt
from Thresholding import otsu
from Boundaries import boundaries_detection
image = cv2.imread('images/prueba.png',cv2.IMREAD_GRAYSCALE)
threshold = otsu.otsu(image)
image[image < threshold] = 0
image[image >= threshold] = 255
boundary = boundaries_detection.moore_boundary_detection(image)
print(boundary)
plt.imshow(image,cmap="gray")
plt.show()