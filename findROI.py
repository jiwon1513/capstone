# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:17:18 2022

@author: 지원
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
path = 'C:/jiwon/please/segmentation/outputs/result_1'
img_name = 'um_road_000006.png'
full_path = path + '/' +img_name
 
img_array = np.fromfile(full_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)



cv2.imshow('first', img)


def findvertex(img):

    hsvLower = np.array([0, 0, 212]) # 추출할 색의 하한 
    hsvUpper = np.array([131, 255, 255]) # 추출할 색의 상한 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환 
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)


    '''cv2.imshow('HSV_test1', hsv_mask)'''


    contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




    '''image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    cv2.imshow('line',image)'''
    return contours



'''contour = contours[0]

leftmost = tuple(contour[contour[:,:,0].argmin()][0])
rightmost = tuple(contour[contour[:,:,0].argmax()][0])
topmost = tuple(contour[contour[:,:,1].argmin()][0])
bottommost = tuple(contour[contour[:,:,1].argmax()][0])

cv2.circle(image,leftmost,5,(0,0,255),-1)
cv2.circle(image,rightmost,5,(0,0,255),-1)
cv2.circle(image,topmost,5,(0,0,255),-1)
cv2.circle(image,bottommost,5,(0,0,255),-1)

cv2.imshow("finddot",image)'''




def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color img인 경우 :
        color = color3
    else:                   # 흑백 img인 경우 :
        color = color1

    cv2.fillPoly(mask, vertices, color)
    # vertices에 정한 점들로 이뤄진 mask 영역(ROI 설정 부분)을 color로 채움
    ROI_image = cv2.bitwise_and(img, mask)
    # 이미지와 color로 채워진 ROI를 합침
    return ROI_image


'''vertices = np.array(
     [[(leftmost[0], leftmost[1]), (leftmost[0], leftmost[1]), (rightmost[0], rightmost[1]), (topmost[0], topmost[1])]],
     dtype=np.int32)
vertices = np.array(
     [[(leftmost[0], leftmost[1]), (bottommost[0], bottommost[1]), (rightmost[0], rightmost[1]), (topmost[0], topmost[1])]],
     dtype=np.int32)    '''

contours = findvertex(img)
roi = region_of_interest(img, contours, (0, 0, 255))

cv2.imshow('roi', roi)
#cv2.fillPoly(mask,contours,1)

#cv2.imshow('re',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
