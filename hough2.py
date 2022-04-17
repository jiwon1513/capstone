# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:46:12 2022

@author: 지원
"""

import matplotlib.image as mpimg
import numpy as np
import os
import io
import cv2
import math
import pickle

from moviepy.editor import VideoFileClip

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)

cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)

font = cv2.FONT_HERSHEY_SIMPLEX
pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
pts = pts.reshape((-1, 1, 2))

l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0))
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)
first_frame = 1



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




def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussoan_blur(img,ksize):
        return cv2.GaussianBlur(img,(ksize,ksize),0)
    
def canny(img,lt,ht):
    return cv2.Canny(img,lt,ht)

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def drawline(img, lines):
   global cache
   global first_frame
   global next_frame

   y_global_min = img.shape[0]
   y_max = img.shape[0]

   l_slope, r_slope = [], []
   l_lane, r_lane = [], []

   det_slope = 0.5
   α = 0.2

   if lines is not None:
       for line in lines:
           for x1,y1,x2,y2 in line:
               slope = get_slope(x1,y1,x2,y2)
               if slope > det_slope:
                   r_slope.append(slope)
                   r_lane.append(line)
               elif slope < -det_slope:
                   l_slope.append(slope)
                   l_lane.append(line)

       y_global_min = min(y1, y2, y_global_min)

   if (len(l_lane) == 0 or len(r_lane) == 0): # 오류 방지
       return 1

   l_slope_mean = np.mean(l_slope, axis =0)
   r_slope_mean = np.mean(r_slope, axis =0)
   l_mean = np.mean(np.array(l_lane), axis=0)
   r_mean = np.mean(np.array(r_lane), axis=0)

   if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
       print('dividing by zero')
       return 1

   # y=mx+b -> b = y -mx
   l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
   r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

   if np.isnan((y_global_min - l_b)/l_slope_mean) or \
   np.isnan((y_max - l_b)/l_slope_mean) or \
   np.isnan((y_global_min - r_b)/r_slope_mean) or \
   np.isnan((y_max - r_b)/r_slope_mean):
       return 1

   l_x1 = int((y_global_min - l_b)/l_slope_mean)
   l_x2 = int((y_max - l_b)/l_slope_mean)
   r_x1 = int((y_global_min - r_b)/r_slope_mean)
   r_x2 = int((y_max - r_b)/r_slope_mean)

   if l_x1 > r_x1: # Left line이 Right Line보다 오른쪽에 있는 경우 (Error)
       l_x1 = ((l_x1 + r_x1)/2)
       r_x1 = l_x1

       l_y1 = ((l_slope_mean * l_x1 ) + l_b)
       r_y1 = ((r_slope_mean * r_x1 ) + r_b)
       l_y2 = ((l_slope_mean * l_x2 ) + l_b)
       r_y2 = ((r_slope_mean * r_x2 ) + r_b)

   else: # l_x1 < r_x1 (Normal)
       l_y1 = y_global_min
       l_y2 = y_max
       r_y1 = y_global_min
       r_y2 = y_max

   current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")

   if first_frame == 1:
       next_frame = current_frame
       first_frame = 0
   else:
       prev_frame = cache
       next_frame = (1-α)*prev_frame+α*current_frame
       
   global pts
   
   pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
   pts = pts.reshape((-1, 1, 2))    

   global l_center
   global r_center
   global lane_center

   div = 2
   l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
   r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
   lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

   global uxhalf, uyhalf, dxhalf, dyhalf
   uxhalf = int((next_frame[2]+next_frame[6])/2)
   uyhalf = int((next_frame[3]+next_frame[7])/2)
   dxhalf = int((next_frame[0]+next_frame[4])/2)
   dyhalf = int((next_frame[1]+next_frame[5])/2)
   
 
   cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), red, 2)
   cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)

   cache = next_frame
            
            
def hough_lines(img, rho, theta, threshold, min_line_len,max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength =  min_line_len,
                            maxLineGap = max_line_gap)
     
    line_img = np.zeros((img.shape[0], img.shape[1],3), dtype = np.uint8)
    drawline(line_img, lines)
    return line_img

def weighted_img(img, inimg, a= 0.8, b=1.0 ,c =0.0):
    return cv2.addWeighted(inimg, a, img, b, c)

def visualize(result):
    height, width = result.shape[:2]
    length = 30
    thickness = 3
    whalf = int(width/2)
    hhalf = int(height/2)
    sl_color = yellow

    # Standard Line
    cv2.line(result, (whalf, lane_center[1]), (whalf, int(height)), sl_color, 2)
    cv2.line(result, (whalf, lane_center[1]), (lane_center[0], lane_center[1]), sl_color, 2)

    # Warning Boundary
    gap = 20
    legth2 = 10
    wb_color = white
    cv2.line(result, (whalf-gap, lane_center[1]-legth2), (whalf-gap, lane_center[1]+legth2), wb_color, 1)
    cv2.line(result, (whalf+gap, lane_center[1]-legth2), (whalf+gap, lane_center[1]+legth2), wb_color, 1)

    # Lane Position
    lp_color = red
    cv2.line(result, (l_center[0], l_center[1]), (l_center[0], l_center[1]-length), lp_color, thickness)
    cv2.line(result, (r_center[0], r_center[1]), (r_center[0], r_center[1]-length), lp_color, thickness)
    cv2.line(result, (lane_center[0], lane_center[1]), (lane_center[0], lane_center[1]-length), lp_color, thickness)

    # cv2.rectangle(result, (0,0), (400, 250), deepgray, -1)
    hei = 30
    font_size = 2
    if lane_center[0] < whalf-gap:
        cv2.putText(result, 'WARNING : ', (10, hei), font, 1, red, font_size)
        cv2.putText(result, 'Turn Right', (190, hei), font, 1, red, font_size)
    elif lane_center[0] > whalf+gap:
        cv2.putText(result, 'WARNING : ', (10, hei), font, 1, red, font_size)
        cv2.putText(result, 'Turn Left', (190, hei), font, 1, red, font_size)
    else :
        pass

    return result

def Region(image):

    
    height, width = image.shape[:2]

    zeros = np.zeros_like(image)
    mask = cv2.fillPoly(zeros,[pts], lime)
    # result = cv2.addWeighted(image, 1, mask, 0.3, 0)

    hhalf = int(height/2)
    if not lane_center[1] < hhalf:
        mask = visualize(mask)
    return mask


                            
cap = cv2.VideoCapture("C:/Users/지원/Desktop/캡디/opencv/test video/video1.mp4")

while (cap.isOpened()):
    ret, image = cap.read()
    height, width = image.shape[:2] #이미지 높이, 너비
    ksize = 5
    gray = grayscale(image) #이미지를 회색화
    gaus = gaussoan_blur(gray,ksize) #라인을 부드럽게 해주는 가우시안 블러
    lt =100
    ht =150
    can= canny(gaus,lt,ht) #선을 검출하는 canny detection
    vertices = np.array(
        [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        dtype=np.int32)
    mask = region_of_interest(can, vertices, (0, 0, 255))  # vertices에 정한 점들 기준으로 ROI 이미지 생성
   
    rho =2
    theta = np.pi/180
    threshold = 100
    min_line_len =50
    max_line_gap = 150
   
    
    lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)
    
    
    result = Region(lines)
    cv2.imshow('s',result)
  
    ii = weighted_img(result,image, a= 0.8, b=1.0,c=0.0)
  

   
    
    cv2.imshow('results', ii)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()