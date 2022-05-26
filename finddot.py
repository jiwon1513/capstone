# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:52:05 2022

@author: 지원
"""


import cv2
import numpy as np
import time

def findvertex(img):

    hsvLower = np.array([0, 0, 235]) # 추출할 색의 하한 
    hsvUpper = np.array([131, 255, 255]) # 추출할 색의 상한 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환 
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)


    '''cv2.imshow('HSV_test1', hsv_mask)'''


    contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)




    image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    cv2.imshow('line',image)
    return contours

# 찾을려고하는 위에 점 2개 위치를 정하는 선 
# offset 값을 이용해서 찾는 점의 위치를 바꿀수있음

def topline(img, top, w, offset):
    cv2.line(img, (0,top[1] + offset),(w,top[1] + offset), (255,0,0),2)
    return top[1]+offset

# 찾을려고하는 아래에 점 2개 위치를 정하는 선 
# offset 값을 이용해서 찾는 점의 위치를 바꿀수있음

def bottomline(img, bottom, w, offset):
    cv2.line(img, (0,bottom[1] + offset),(w,bottom[1] + offset), (255,0,0),2)
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

def finddot(img,contour,yt, yb):
    '''a = np.where( (contour[:,:,1] <(yt + e)) & (contour[:,:,1] > (yt - e)) )
    b = np.where( (contour[:,:,1] <(yb + e)) & (contour[:,:,1] > (yb - e)) )'''
    
    a = np.where(contour[:,:,1] == yt )
    b = np.where(contour[:,:,1] == yb )
    
    print(a[0])
    

    amx,amy = minmax(a,contour)
    bmx,bmy = minmax(b,contour)
    

        
    cv2.circle(img,contour[a[0][amx]][0],5,(0,255,255),-1)
    cv2.circle(img,contour[a[0][amy]][0],5,(0,255,255),-1)

    cv2.circle(img,contour[b[0][bmx]][0],5,(0,255,255),-1)
    cv2.circle(img,contour[b[0][bmy]][0],5,(0,255,255),-1)
    
    return 0


start = time.time()
 
path = 'C:/jiwon/please/segmentation/outputs/result_1'
img_name = 'um_road_000004.png'
full_path = path + '/' +img_name
 
img_array = np.fromfile(full_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

h, w, c = img.shape


cv2.imshow('first', img)




contours = findvertex(img)

contour = contours[0]

'''leftmost = tuple(contour[contour[:,:,0].argmin()][0])
rightmost = tuple(contour[contour[:,:,0].argmax()][0])'''

#최상점, 최하점
#contour[:,:,1]는 contour에서 y값만을 모은 배열

topmost = tuple(contour[contour[:,:,1].argmin()][0])
bottommost = tuple(contour[contour[:,:,1].argmax()][0])


'''cv2.circle(img,leftmost,5,(0,0,255),-1)
cv2.circle(img,rightmost,5,(0,0,255),-1) '''

# 최상점, 최하점 빨간 점으로 표시
cv2.circle(img,topmost,5,(0,0,255),-1)
cv2.circle(img,bottommost,5,(0,0,255),-1)


cv2.imshow("dot",img)

#상한, 하한 선 그리기
t1 = topline(img, topmost, w,20)
b1 = bottomline(img,bottommost,w,-20)  
    
    

#그린 선을 기준으로 점 찾기
finddot(img,contour,t1,b1)

   
cv2.imshow("finddot",img)


#코드 시행시간
print("time : ", time.time() - start)


'''ep = 0.009*cv2.arcLength(contour, True)
approx1 = cv2.approxPolyDP(contour, ep, True)

cv2.drawContours(img,[approx1],0,(255,0,0),3)
cv2.imshow('approx',img)'''


cv2.waitKey(0)
cv2.destroyAllWindows()



