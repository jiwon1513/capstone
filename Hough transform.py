import cv2
import math
import cv2 as cv
import numpy as np

cap = cv2.VideoCapture("C:/Users/vlxj/Desktop/road.mp4")

while (True):
    ret, src = cap.read() # 캡처
    src = cv2.resize(src, (640, 360)) # 사이즈 조정
    dst = cv.Canny(src, 50, 200, None, 3)
    # 엣지 검출 , cv.Canny는 엣지 검출함수로 이미지의 엣지만을 되돌려준다
    # cv.Canny(gray_img, threshold1, threshold) 이것이 기본 문법이며 threshold는 엣지인지 아닌지 판단하는 임계값이다.
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR) # 컬러 이미자를 흑백으로.
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 180, None, 0, 0)
    # cv.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]])
    # rho는 r값의 범위로 0~1이다, theta의 범위는 0~180이다.
    # threshold는 만나는 점의 기준값으로 숫자 크기가 낮을수록 많은 선이 검출되고 숫자 크기가 높을수록 정확도가 높아진다.

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
            # cv.line(img, pt1, pt2, color [, thickness[, lineType[, shift]]])

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    # cv.HoughLinesP(img, rho, theta, threshold, minLineLength, maxLineGap)
    # minLineLength 는 선의 최소 길이, maxLineGap 는 선과 선 사이의 최대 허용 가격

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic  Line Transform", cdstP)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
