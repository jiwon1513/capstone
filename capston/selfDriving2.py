import cv2 # opencv 사용
import numpy as np
image = cv2.imread('C:/Users/vlxj/Desktop/DASF.jpg') # 이미지 읽기
image = cv2.resize(image, (640, 360))
mark = np.copy(image)# image 복사


# #  BGR 제한 값 설정
# blue_threshold = 200
# green_threshold = 200
# red_threshold = 200
# bgr_threshold = [blue_threshold, green_threshold, red_threshold]
#
# # BGR 제한 값보다 작으면 검은색으로
# thresholds = (image[:,:,0] < bgr_threshold[0]) \
#             | (image[:,:,1] < bgr_threshold[1]) \
#             | (image[:,:,2] < bgr_threshold[2])
# mark[thresholds] = [0,0,0]

#  BGR 제한 값 설정
blue_threshold = 135
green_threshold = 178
red_threshold = 198
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# BGR 제한 값보다 작으면 검은색으로
thresholds = (image[:,:,0] - 15 <= bgr_threshold[0]) \
            | (image[:,:,1] - 5 <= bgr_threshold[1]) \
            | (image[:,:,2] - 3 <= bgr_threshold[2])
mark[thresholds] = [0,0,0]



cv2.imshow('white',mark) # 흰색 추출 이미지 출력
cv2.imshow('result',image) # 이미지 출력
cv2.waitKey(0)