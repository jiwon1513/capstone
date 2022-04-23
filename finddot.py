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

def topline(img, top, w, offset):
    cv2.line(img, (0,top[1] + offset),(w,top[1] + offset), (255,0,0),2)
    return top[1]+offset

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

def finddot(img,contour,yt, yb,e):
    a = np.where( (contour[:,:,1] <(yt + e)) & (contour[:,:,1] > (yt - e)) )
    b = np.where( (contour[:,:,1] <(yb + e)) & (contour[:,:,1] > (yb - e)) )
    
 
    

    amx,amy = minmax(a,contour)
    bmx,bmy = minmax(b,contour)
    

        
    cv2.circle(img,contour[a[0][amx]][0],5,(0,255,255),-1)
    cv2.circle(img,contour[a[0][amy]][0],5,(0,255,255),-1)

    cv2.circle(img,contour[b[0][bmx]][0],5,(0,255,255),-1)
    cv2.circle(img,contour[b[0][bmy]][0],5,(0,255,255),-1)
    
    return 0


start = time.time()
 
path = 'C:/jiwon/please/segmentation/outputs/result_1'
img_name = 'um_road_000007.png'
full_path = path + '/' +img_name
 
img_array = np.fromfile(full_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

h, w, c = img.shape


cv2.imshow('first', img)




contours = findvertex(img)

contour = contours[0]

'''leftmost = tuple(contour[contour[:,:,0].argmin()][0])
rightmost = tuple(contour[contour[:,:,0].argmax()][0])'''


topmost = tuple(contour[contour[:,:,1].argmin()][0])
bottommost = tuple(contour[contour[:,:,1].argmax()][0])


'''cv2.circle(img,leftmost,5,(0,0,255),-1)
cv2.circle(img,rightmost,5,(0,0,255),-1) '''


cv2.circle(img,topmost,5,(0,0,255),-1)
cv2.circle(img,bottommost,5,(0,0,255),-1)


cv2.imshow("dot",img)

t1 = topline(img, topmost, w,20)
b1 = bottomline(img,bottommost,w,-20)  
    
    


finddot(img,contour,t1,b1,2)

   
cv2.imshow("finddot",img)

print("time : ", time.time() - start)


'''ep = 0.009*cv2.arcLength(contour, True)
approx1 = cv2.approxPolyDP(contour, ep, True)

cv2.drawContours(img,[approx1],0,(255,0,0),3)
cv2.imshow('approx',img)'''


cv2.waitKey(0)
cv2.destroyAllWindows()



