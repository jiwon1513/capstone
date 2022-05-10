import cv2
import numpy as np

# def mark_img_w(img):
#     height, width = img.shape[:2]
#
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     lower_white = (0, 0, 0)
#     upper_white = (255, 11, 255)
#
#     img_mask1 = cv2.inRange(img_hsv, lower_white, upper_white)
#     img_result = cv2.bitwise_and(img, img, mask=img_mask1)
#
#     return img_result
#
#
# def mark_img_y(img):
#     height, width = img.shape[:2]
#
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     lower_yellow = (16, 70, 30)
#     upper_yellow = (24, 255, 255)
#
#     img_mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
#     img_result = cv2.bitwise_and(img, img, mask=img_mask2)
#
#     return img_result

def roi(equ_frame, vertices):
    # blank mask:
    mask = np.zeros_like(equ_frame)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(equ_frame, mask)
    return masked


video = cv2.VideoCapture("C:/Users/vlxj/Desktop/data/road.mp4")
while True:
    ret, orig_frame = video.read()
    # orig_frame = orig_frame[0:590, 0:1300]
    if not ret:
        video = cv2.VideoCapture("C:/Users/vlxj/Desktop/data/road.mp4")
        continue

    # 가우시안 피라미드 다운샘플링 사용 가로세로 1/2 씩 줄어든 이미지로 변함
    lineframe = cv2.pyrDown(orig_frame)  # 라인표시할 프레임
    frame = cv2.pyrDown(orig_frame)

    # 노이즈제거를 위해 가우시안블러사용
    gaus_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 밝기정보추출(히스토그램 평활화)
    gray_frame = cv2.cvtColor(gaus_frame, cv2.COLOR_BGR2GRAY)
    equ_frame = cv2.equalizeHist(gray_frame)

    # 소벨
    """
    sobelX = np.array([[0, 1, 2],
                            [-1, 0, 1],
                            [-2, -1, 0]])
    gx = cv2.filter2D(equ_frame, cv2.CV_32F, sobelX)
    sobelY = np.array([[-2, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 2]])
    gy = cv2.filter2D(equ_frame, cv2.CV_32F, sobelY)
    mag   = cv2.magnitude(gx, gy)
    """
    # edges_frame = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)

    # 캐니
    edges_frame = cv2.Canny(equ_frame, 100, 200)  # canny를 사용하여 에지검출

    # ROI영역설정
    height, width = frame.shape[:2]  # 이미지 높이, 너비
    # vertices = np.array([[(100,height),(width/2-75, height/2+100), (width/2+75, height/2+100), (width-100,height)]], dtype=np.int32)
    # roi_frame = roi(edges_frame, [vertices])
    # 왼쪽 아래, 왼쪽 위, 오른쪽 위, 오른쪽 아래
    vertices = np.array(
        [[(0, 240), (300, 140), (355, 150), (width, 310), (0, 310)]],
        dtype=np.int32)
    # vertices2 = np.array([[(width / 2 + 100, height), (width / 2, height / 2 + 100), (width / 2 + 75, height / 2 + 100),
    #                        (width - 100, height)]], dtype=np.int32)

    roi_frame1 = roi(edges_frame, [vertices])  # edges영상에 roi적용
    # roi_frame2 = roi(edges_frame, [vertices2])
    # roi_frame = cv2.add(roi_frame1, roi_frame2)
    roi_frame = np.uint8(roi_frame1)

    lines = cv2.HoughLinesP(roi_frame, 1, np.pi / 180, 50, maxLineGap=80)

    # 검출된 에지에 확률허프변환을 사용하여 직선을 찾는다.
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(lineframe, (x1, y1), (x2, y2), (51, 104, 255), 2)  # 찾은 직선이 보이게 선을 그린다.
    # frame = roi(frame, [vertices])

    cv2.imshow("lineframe", lineframe)
    cv2.imshow("roi_frame", roi_frame)
    key = cv2.waitKey(24)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()


