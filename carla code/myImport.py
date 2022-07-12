import os

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

from time import time
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import load_model
from scipy import interpolate

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#
main_path = 'E:/dataset/'
image_path = main_path + "test/RGB/"

# num = 0
# model, height, width, name = loadModel(num)


# bird eye view
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


# txt 파일로 출력하여 결과 확인용
def writePredict(eval, filename):
    f = open(f"filename.txt", 'w')
    for i in range(len(eval)):
        for j in range(len(eval[0])):
            f.write(str(eval[i][j]) + ' ')
        f.write('\n')
    f.close()


# 각 모델과 해당 크기와 이름 데이터 출력
def loadModel(num):
    i = num
    if i == 0:
        height, width = 256, 256
        name = "UNet"
        model = load_model(main_path + 'carla-image-segmentation-' + name + '.h5')
    elif i == 1:
        height, width = 192, 192
        name = "PSPNet"
        model = load_model(main_path + 'carla-image-segmentation-' + name + '.h5')
    elif i == 2:
        height, width = 224, 224
        name = "ResNet"
        model = load_model(main_path + 'carla-image-segmentation-' + name + '.h5')
    else:
        height, width = 192, 192
        name = "ResNet_PSPNet"
        model = load_model(main_path + 'carla-image-segmentation-' + name + '.h5')

    print('load images')
    return model, height, width, name


# 입력된 라벨 이미지에서 가장 좌측값과 가장 우측값들에 보간법을 각각 적용하여 직선 함수를 생성
def make_interpolate(f1, f2, line):
    # line = line.numpy()  # x값 기준 중복값 처리 및 보간법 적용을 위한 전처리
    mid_point = 128
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
    left = np.transpose(left)
    right = np.transpose(right)

    if len(left) >= 2:
        x_left = left[0]
        y_left = left[1]
        if len(x_left) > 2:
            f1 = interpolate.InterpolatedUnivariateSpline([x_left[0], x_left[-1]], [y_left[0], y_left[-1]], k=1)
    if len(right) >= 2:
        x_right = right[0]
        y_right = right[1]
        if len(x_right) > 2:
            f2 = interpolate.InterpolatedUnivariateSpline([x_right[0], x_right[-1]], [y_right[0], y_right[-1]], k=1)
    return f1, f2


def init_interpolate(x=[10, 60, 110], y=[240, 195, 150]):
    return interpolate.InterpolatedUnivariateSpline(x, y, k=2)


# 모델에 이미지를 넣어 여러 필터링을 거침
def predict(model, height, width, image):
    resize = (height, width)
    # image = tf.io.read_file(image_list)
    # image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)   # tensorflow에서 제공하는 image 형태로 convert
    image = tf.image.resize(image, resize, method='nearest')    # resize
    preds = model(np.expand_dims(image, 0), training=False)     # prediction
    return preds, image


def preprocessing_interpolate(preds, height, width):
    pred_num = preds.numpy()
    t1, t2, b1, b2 = [width - 154, height - 106], [width - 102, height - 106], [0, height - 16], [width, height - 16]
    birdeyeview = wrapping(pred_num[0], t1, t2, b1, b2)
    image_bev = tf.math.argmax(birdeyeview, 2)    # layer-2에서 argmax 적용 : 가장 큰 원소 인덱스로 각 리스트 치환
    image_bev = tf.keras.backend.eval(image_bev)    # value만 list 형식으로 출력

    line_bev = tf.where(tf.equal(image_bev, 2)).numpy()    # 값이 2(차선)인 위치만 출력
    line_load_bev = tf.where(tf.equal(image_bev, 1)).numpy()    # 값이 1(도로외각선)인 위치만 출력
    return line_bev, line_load_bev