import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

# file_path = 'E:/dataset/'
# image_path = file_path + 'town5/RGB/'
# mask_path = file_path + 'town5/seg/'

main_path = "E:/dataset/"
# image_path = [main_path + "town" + i + "/RGB/" for i in ['0', '5', '6', '7', '10']]
# mask_path = [main_path + "town" + i + "/seg/" for i in ['0', '5', '6', '7', '10']]
image_path = main_path + "test/RGB/"
mask_path = main_path + "test/seg/"
image_list = []
mask_list = []

image_temp = os.listdir(image_path)
mask_temp = os.listdir(mask_path)
image_list.extend([image_path+i for i in image_temp])
mask_list.extend([mask_path+i for i in mask_temp])

import natsort
image_list = natsort.natsorted(image_list)
mask_list = natsort.natsorted(mask_list)
import sys
for i in range(10):
    if image_list[i].split('/')[-1] != mask_list[i].split('/')[-1]:
        print("input list sort error!")
        sys.exit(0)
print("Complete sorting input lists!")

# height, width = 224, 224
# height, width = 256, 256
height, width = 192, 192
resize = (height, width)
test_images = np.zeros((len(image_list), height, width, 3), dtype=np.uint8)
test_masks = np.zeros((len(mask_list), height, width, 1), dtype=np.uint8)

# # Label of the classes
# CLASSES = {0:'Unlabeled',
#            1:'Building',
#            2:'Fence',
#            3:'Other',
#            4:'People',
#            5:'Posts',
#            6:'Road Marking',
#            7:'Street',
#            8:'Sidewalk',
#            9:'Vegatation',
#            10:'Vehicle',
#            11:'Wall',
#            12:'Traffic Sign'}

for n in tqdm(range(len(image_list))):
    image = tf.io.read_file(image_list[n])
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.uint8)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(mask_list[n])
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')
    mask_road = np.zeros((height, width, 1), dtype=np.uint8)
    mask_road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 1
    mask_road[np.where(mask == 6)[0], np.where(mask == 6)[1]] = 2
    test_images[n] = image
    test_masks[n] = mask_road

plt.subplots(figsize=(15, 5))
for i in range(5):
    N = random.randint(0, int(len(test_images)))
    plt.subplot(5, 3, i * 3 + 1)
    plt.imshow(test_images[N])
    plt.subplot(5, 3, i * 3 + 2)
    plt.imshow(test_masks[N])
    plt.subplot(5, 3, i * 3 + 3)
    plt.imshow(test_images[N])
    plt.imshow(test_masks[N], alpha=0.5)
plt.savefig(f'{main_path}test_{height}_{width}.png')
plt.show()

np.save(main_path+f'test_test_images_{height}_{width}.npy', test_images)
np.save(main_path+f'test_test_masks_{height}_{width}.npy', test_masks)
print("Complete saving data")

