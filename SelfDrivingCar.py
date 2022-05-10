import cv2
import numpy as np


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color img인 경우 :
        color = color3
    else:  # 흑백 img인 경우 :
        color = color1

    cv2.fillPoly(mask, vertices, color)
    # vertices에 정한 점들로 이뤄진 mask 영역(ROI 설정 부분)을 color로 채움
    ROI_image = cv2.bitwise_and(img, mask)
    # 이미지와 color로 채워진 ROI를 합침
    return ROI_image


def mark_img_w(img, blue_threshold=200, green_threshold=200, red_threshold=200):  # 흰색 차선 찾기
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]  # BGR 제한 값

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (image[:, :, 0] < bgr_threshold[0]) \
                 | (image[:, :, 1] < bgr_threshold[1]) \
                 | (image[:, :, 2] < bgr_threshold[2])
    mark_w[thresholds] = [0, 0, 0]
    return mark_w


def mark_img_y(img, blue_threshold=134, green_threshold=158, red_threshold=182):  # 노란색 차선 찾기
    bgr_threshold1 = [blue_threshold, green_threshold, red_threshold]  # BGR 제한 값

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (image[:, :, 0] < bgr_threshold1[0]) \
                 | (image[:, :, 1] < bgr_threshold1[1]) \
                 | (image[:, :, 2] < bgr_threshold1[2])
    mark_y[thresholds] = [0, 0, 0]
    return mark_y


cap = cv2.VideoCapture("C:/Users/vlxj/Desktop/road.mp4")

while (cap.isOpened()):
    ret, image = cap.read()
    height, width = image.shape[:2]  # 이미지 높이, 너비

    # 사다리꼴 모형의 Points
    vertices = np.array(
        [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        dtype=np.int32)
    roi_img = region_of_interest(image, vertices)  # vertices에 정한 점들 기준으로 ROI 이미지 생성

    mark_w = np.copy(roi_img)  # roi_img 복사
    mark_w = mark_img_w(roi_img)  # 흰색 차선 찾기

    mark_y = np.copy(roi_img)  # roi_img 복사
    mark_y = mark_img_y(roi_img)  # 흰색 차선 찾기

    # # 흰색 차선 검출한 부분을 원본 image에 오버랩 하기
    # color_thresholds = (mark_w[:, :, 0] == 0) & (mark_w[:, :, 1] == 0) & (mark_w[:, :, 2] > 200)

    cv2.imshow('results', mark_w)
    cv2.imshow('resultqs', mark_y)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
