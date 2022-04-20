import numpy as np
import cv2
# BGR에서 HSV값 추출하기
# color = [151, 154, 138]
# pixel = np.uint8([[color]])
#
# hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
# print(hsv, 'shape', hsv.shape)
#
# hsv = hsv[0][0]
#
# print("bgr: ", color)
# print("hsv: ", hsv)

# img_color = cv2.imread("C:/Users/vlxj/Desktop/no.png")
#
# img_color = cv2.resize(img_color, (640, 360))
# height, width = img_color.shape[:2]
#
# img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
#
# lower_white = (0, 0, 0)
# upper_white = (255, 11, 255)
# img_mask1 = cv2.inRange(img_hsv, lower_white, upper_white)
#
# lower_red = (16, 70, 30)
# upper_red = (24, 255, 255)
# img_mask2 = cv2.inRange(img_hsv, lower_red, upper_red)
#
# # lower_dark_white = (79, 17, 117)
# # upper_dark_white = (84, 26, 154)
# # img_mask3 = cv2.inRange(img_hsv, lower_red, upper_red)
#
# img_mask = cv2.add(img_mask2, img_mask1)
# img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)
#
# cv2.imshow('img_result', img_result)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def mark_img_w(img):
    height, width = img.shape[:2]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = (0, 0, 0)
    upper_white = (255, 11, 255)

    img_mask1 = cv2.inRange(img_hsv, lower_white, upper_white)
    img_result = cv2.bitwise_and(img, img, mask=img_mask1)

    return img_result


def mark_img_y(img):
    height, width = img.shape[:2]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = (16, 70, 30)
    upper_yellow = (24, 255, 255)

    img_mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    img_result = cv2.bitwise_and(img, img, mask=img_mask2)

    return img_result


cap = cv2.VideoCapture("C:/Users/vlxj/Desktop/road.mp4")

while (True):
    ret, src = cap.read()
    src = cv2.resize(src, (640, 360))

    img_result = cv2.add(mark_img_w(src), mark_img_y(src))

    cv2.imshow('img_result', img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
