import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

from predict import *
from time import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

main_path = 'E:/dataset/'
image_path = main_path + "test/RGB/"

num = 0
model, height, width, name = loadModel(num)
model.summary()


n = 0
image_list = []
image_temp = os.listdir(image_path)
image_list.extend([image_path+i for i in image_temp])

start = time()
resize = (height, width)
image = tf.io.read_file(image_list[n])
# print(f'read time : {time() - start}')
# start = time()
image = tf.image.decode_png(image, channels=3)
# print(f'decode time : {time() - start}')
# start = time()
# image = tf.image.convert_image_dtype(image, np.uint8)
image = tf.image.convert_image_dtype(image, tf.uint8)
# print(f'convert time : {time() - start}')
# start = time()
image = tf.image.resize(image, resize, method='nearest')
# print(f'resize time : {time() - start}')
# start = time()

# preds = model.predict(np.expand_dims(image, 0))
# print(f'predict time 1: {time() - start}')
# plt.subplot(2, 1, 1)
# plt.imshow(image)
# plt.subplot(2, 1, 2)
# plt.imshow(preds[0])
# # plt.show()
# start = time()
# # print(preds)
# # temp = np.zeros([len(preds[0]), len(preds[0][0])])
# # for i in range(len(preds[0])):
# #     for j in range(len(preds[0][0])):
# #         temp[i][j] = np.argmax(preds[0][i][j])
# A = tf.math.argmax(preds[0])
# print(f'argmax time : {time() - start}')
# # print(A)
# start = time()

preds = model(np.expand_dims(image, 0), training=False)
# print(f'predict time 2: {time() - start}')
# plt.subplot(2, 1, 1)
# plt.imshow(image)
# plt.subplot(2, 1, 2)
# plt.imshow(preds[0])
# plt.show()
# start = time()
# print(preds)
# temp = np.zeros([len(preds[0]), len(preds[0][0])])
# for i in range(len(preds[0])):
#     for j in range(len(preds[0][0])):
#         temp[i][j] = np.argmax(preds[0][i][j])
B = tf.math.argmax(preds[0], 2)
print(f'argmax time : {time() - start}')
# print(preds[0])
# print(B)
# start = time()
# plt.subplot(2, 1, 1)
# plt.imshow(image)
# plt.subplot(2, 1, 2)
# plt.imshow(preds[0])
# plt.show()
#
# print(preds[0])