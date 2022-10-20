import os

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

from time import time
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import load_model

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#
main_path = 'E:/dataset/'
image_path = main_path + "test/RGB/"

# num = 0
# model, height, width, name = loadModel(num)


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