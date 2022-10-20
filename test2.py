import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

# from predict import *
from time import time
import cv2
import tensorflow as tf
import numpy as np
from scipy import interpolate
from tensorflow.python.keras.models import load_model

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#
main_path = 'E:/dataset/'
image_path = main_path + "test/RGB/"
file_path = 'E:/dataset/'

# num = 0
# model, height, width, name = loadModel(num)


def loadModel(num):
    i = num
    if i == 0:
        height, width = 256, 256
        name = "UNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 1:
        height, width = 192, 192
        name = "PSPNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 2:
        height, width = 224, 224
        name = "ResNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 3:
        height, width = 224, 224
        name = "ResNet2"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 4:
        height, width = 192, 192
        name = "ResNet_PSPNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')

    print('load images')
    return model, height, width, name


def predict(model, height, width, image_list):
    resize = (height, width)
    image = tf.io.read_file(image_list)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.resize(image, resize, method='nearest')
    preds = model(np.expand_dims(image, 0), training=False)
    B = tf.math.argmax(preds[0], 2)
    C = tf.keras.backend.eval(B)
    return preds, image


def wrapping(img, t1, t2, b1, b2):
    h = int(len(img))
    w = int(len(img[0]))

    # 좌표점은 좌상->좌하->우상->우하
    source = np.float32([t1, b1, t2, b2])
    destination = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    # minv = cv2.getPerspectiveTransform(destination, source)
    image = cv2.warpPerspective(img, transform_matrix, (w, h))

    return image


def writePredict(eval, filename):
    f = open(f"filename.txt", 'w')
    for i in range(len(eval)):
        for j in range(len(eval[0])):
            f.write(str(eval[i][j]) + ' ')
        f.write('\n')
    f.close()


def make_interpolate(f1, f2, line):
    # line = line.numpy()  # x값 기준 중복값 처리 및 보간법 적용을 위한 전처리
    # mid_point = 128
    old = 0     # x 값 0부터 시작
    left = []    # x 값 중 첫번째 값만 배열 저장
    right = []
    right_temp = []
    # print(len(line))
    # print(line)
    for i in range(len(line)):
        now = line[i][0]    # label값 기준으로 필터링 된 포인트[x, y] 중 x 좌표값, 이를 기준으로 가장 좌측값과 우측값을 출력
        if old == now:
            right_temp.append(line[i])
        else:
            old = now
            left.append(line[i])
            if len(right_temp) > 0:
                right.append(right_temp[-1])
            right_temp = []
    # print(left)
    # print(right)

    left = list(filter(lambda x: x[1] < 128, left))
    left = np.transpose(left)
    if len(left) >= 2:
        x_left = left[0]
        y_left = left[1]
        # print(x_left)
        # print(y_left)
        if len(x_left) > 2:
            # f1 = np.polyfit([x_left[0], x_left[-1]], [y_left[0], y_left[-1]], 3)
            f1 = np.polyfit(x_left, y_left, 1)
            f1 = np.poly1d(f1)
            # f1 = interpolate.InterpolatedUnivariateSpline([x_left[0], x_left[-1]], [y_left[0], y_left[-1]], k=1)

    right = list(filter(lambda x: x[1] > 128, right))
    right = np.transpose(right)
    if len(right) >= 2:
        x_right = right[0]
        y_right = right[1]
        if len(x_right) > 2:
            # f2 = np.polyfit([x_right[0], x_right[-1]], [y_right[0], y_right[-1]], 3)
            f2 = np.polyfit(x_right, y_right, 1)
            f2 = np.poly1d(f2)
            # f2 = interpolate.InterpolatedUnivariateSpline([x_right[0], x_right[-1]], [y_right[0], y_right[-1]], k=1)

    # if len(line) >= 2:
    #     # x = line[0]
    #     # y = line[1]
    #     # print(np.where(line[1] <= 128)[0])
    #     x_left = line[0][np.where(line[1] <= mid_point)[0]]
    #     y_left = line[1][np.where(line[1] <= mid_point)[0]]
    #     x_right = line[0][np.where(line[1] > mid_point)[0]]
    #     y_right = line[1][np.where(line[1] > mid_point)[0]]
    #     if len(x_left) > 2:
    #         # midpoint = int(len(x_left)/2)
    #         # f1 = interpolate.InterpolatedUnivariateSpline(x_left[4:-4], y_left[4:-4], k=1)    # 보간법, 2차함수, 점 3개 이상 있으면 update
    #         # f1 = interpolate.InterpolatedUnivariateSpline([x_left[0], x_left[midpoint], x_left[-1]], [y_left[0], y_left[midpoint], y_left[-1]], k=1)
    #         f1 = interpolate.InterpolatedUnivariateSpline([x_left[0], x_left[-1]], [y_left[0], y_left[-1]], k=1)
    #     if len(x_right) > 2:
    #         # midpoint = int(len(x_right)/2)
    #         # f2 = interpolate.InterpolatedUnivariateSpline(x_right, y_right, k=1)    # 보간법, 2차함수, 점 3개 이상 있으면 update
    #         # f2 = interpolate.InterpolatedUnivariateSpline([x[0], x[midpoint], x[-1]], [y[0], y[midpoint], y[-1]], k=1)
    #         f2 = interpolate.InterpolatedUnivariateSpline([x_right[0], x_right[-1]], [y_right[0], y_right[-1]], k=1)
    return f1, f2


def plothistogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase


def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array(
        [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)],[int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image