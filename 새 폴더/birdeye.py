# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:02:01 2022

@author: 지원
"""

import cv2
import numpy as np
import findroidot as fr

def wrapping(img,t1,t2,b1,b2):
    # 좌표점은 좌상->좌하->우상->우하
    source = np.float32([t1,b1,t2,b2])
    destination = np.float32([[0, 0], [0 , h], [w, 0], [w, h]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(img, transform_matrix, (w, h))

    return _image


path = 'C:/jiwon/please/segmentation/outputs/result_1'
img_name = 'um_road_000022.png'
full_path = path + '/' +img_name
 
img_array = np.fromfile(full_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

path1 = 'C:/jiwon/please/segmentation/outputs/result_2'
img_name1 = 'um_road_000022.png'
full_path1 = path1 + '/' +img_name1
 
img_array1 = np.fromfile(full_path1, np.uint8)
img1 = cv2.imdecode(img_array1, cv2.IMREAD_COLOR)


h, w, c = img.shape


cv2.imshow('second', img1)

contours = fr.findvertex(img)                
contour = fr.majorcontour(contours)


roi =fr.region_of_interest(img1,[contour])

t1 = fr.topline(roi, contour, w,50)
b1 = fr.bottomline(roi,contour,w,-20)  
    
    
t1,t2,b1,b2 = fr.finddot(roi,contour,t1,b1,2)

cv2.imshow("roi",roi) 


_image = wrapping(roi, t1, t2, b1, b2)

   
cv2.imshow("birdeye",_image)

cv2.waitKey(0)
cv2.destroyAllWindows()