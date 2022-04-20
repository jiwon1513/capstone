import cv2
import numpy as np
import matplotlib.pyplot as plt

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])

    source = np.float32([[w // 2 - 30, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image


img = cv2.imread('C:/Users/vlxj/Desktop/asdfasdf.png')

img_warp = wrapping(img)

plt.imshow(cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB))
plt.show()