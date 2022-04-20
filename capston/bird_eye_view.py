import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/vlxj/Desktop/road2.jpg') # Read the test img

IMAGE_H, IMAGE_W = img.shape[:2]

src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[IMAGE_W / 4, IMAGE_H], [IMAGE_W * 3 / 4, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping

plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()




# cap = cv2.VideoCapture("C:/Users/vlxj/Desktop/road.mp4")
#
# while (cap.isOpened()):
#     ret, img = cap.read()
#     IMAGE_H, IMAGE_W = img.shape[:2] #이미지 높이, 너비
#
#     src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
#     dst = np.float32([[IMAGE_W / 3, IMAGE_H], [IMAGE_W * 2 / 3, IMAGE_H], [0, 0], [IMAGE_W, 0]])
#     M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
#     Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
#
#
#     img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
#     warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
#
#     plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
#     plt.show()
#
#
# cap.release()
# cv2.destroyAllWindows()