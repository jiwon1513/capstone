# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:52:05 2022

@author: 지원
"""


import cv2
import numpy as np
import time

#외곽선 찾기
def findvertex(img):
    
    

    hsvLower = np.array([0, 0, 213]) # 추출할 색의 하한 
    hsvUpper = np.array([130, 255, 255]) # 추출할 색의 상한 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환 
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)


    '''cv2.imshow('HSV_test1', hsv_mask)'''


    contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)




    #image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    #cv2.imshow('line',image)
    return contours


#외곽선들 중에서 가장 큰 영역 찾기

def majorcontour(contours):
    m = 0;
    for i in range(0,len(contours)):
        if(m < contours[i].size):   
            m = i;
    return contours[m];  

#찾은 외곽선으로 roi 그리기
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


# 찾을려고하는 위쪽 점 2개 위치를 정하는 선 
# offset 값을 이용해서 찾는 점의 위치를 바꿀수있음

def topline(img, contour, w, offset):
    top = tuple(contour[contour[:,:,1].argmin()][0])
    
    #cv2.circle(img,top,5,(0,0,255),-1)   
    #cv2.line(img, (0,top[1] + offset),(w,top[1] + offset), (255,0,0),2)
    return top[1]+offset

# 찾을려고하는 아래에 점 2개 위치를 정하는 선 
# offset 값을 이용해서 찾는 점의 위치를 바꿀수있음

def bottomline(img, contour, w, offset):
    bottom = tuple(contour[contour[:,:,1].argmax()][0])
    #cv2.circle(img,bottom,5,(0,0,255),-1)
    #cv2.line(img, (0,bottom[1] + offset),(w,bottom[1] + offset), (255,0,0),2)
    return bottom[1] + offset

def minmax(c, contour):
    x = c[0].size
   
    minx = contour[c[0][0]][0][0]
    maxx = contour[c[0][-1]][0][0]
    
    
  
    mx = 0
    my = x-1
  
    for i in range (0,x):
        
       
        if(minx > contour[c[0][i]][0][0]):
            minx = contour[c[0][i]][0][0]           
            mx = i
                   
            
        if(maxx < contour[c[0][i]][0][0]):
            maxx = contour[c[0][i]][0][0]    
            my = i
          
           
            
    return mx,my            

#상한, 하한선을 정한 상태에서 점 4개를 찾아줌

   

def finddot(img,contour,yt, yb,e):
    '''a = np.where( (contour[:,:,1] <(yt + e)) & (contour[:,:,1] > (yt - e)) )
    b = np.where( (contour[:,:,1] <(yb + e)) & (contour[:,:,1] > (yb - e)) )'''
    
    a = np.where(contour[:,:,1] == yt )
    b = np.where(contour[:,:,1] == yb )
    
   
    

    amx,amy = minmax(a,contour)
    bmx,bmy = minmax(b,contour)
    
 
    

        
    cv2.circle(img,contour[a[0][amx]][0],5,(0,255,255),-1)
    cv2.circle(img,contour[a[0][amy]][0],5,(0,255,255),-1)

    cv2.circle(img,contour[b[0][bmx]][0],5,(0,255,255),-1)
    cv2.circle(img,contour[b[0][bmy]][0],5,(0,255,255),-1)
    
    t1 = contour[a[0][amx]][0]
    t2 = contour[a[0][amy]][0]
    
    b1 = contour[b[0][bmx]][0]
    b2 = contour[b[0][bmy]][0]
    
    return t1,t2,b1,b2


"""
start = time.time()
path = 'C:/jiwon/please/segmentation/outputs/result_1'
img_name = 'um_road_000003.png'
full_path = path + '/' +img_name
 
img_array = np.fromfile(full_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

h, w, c = img.shape

cv2.imshow('first', img)


contours = findvertex(img)                
contour = majorcontour(contours)

# roi 그리기
roi = region_of_interest(img,[contour], (0, 0, 255))

cv2.imshow('roi', roi)


#최상점, 최하점
#contour[:,:,1]는 contour에서 y값만을 모은 배열




'''cv2.circle(img,leftmost,5,(0,0,255),-1)
cv2.circle(img,rightmost,5,(0,0,255),-1) '''

# 최상점, 최하점 빨간 점으로 표시



#cv2.imshow("dot",img)

#상한, 하한 선 그리기
t1 = topline(roi, contour, w,20)
b1 = bottomline(roi,contour,w,-20)  
    
    

#그린 선을 기준으로 점 찾기
finddot(roi,contour,t1,b1,2)

   
cv2.imshow("finddot",roi)
print("time :", time.time() - start)


cv2.waitKey(0)
cv2.destroyAllWindows()


"""

