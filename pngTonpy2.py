import random
import natsort as natsort
from numpy.random import seed
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

# file_path = 'E:/dataset/'
# image_path = file_path + 'town5/RGB/'
# mask_path = file_path + 'town5/seg/'

main_path = 'E:/dataset/'
image_path = [main_path + "town" + i + "/RGB/" for i in ['0', '5', '6', '7', '10']]
mask_path = [main_path + "town" + i + "/seg/" for i in ['0', '5', '6', '7', '10']]
# image_path = main_path + "test/RGB/"
# mask_path = main_path + "test/seg/"
image_list = []
mask_list = []

for image_path, mask_path in zip(image_path, mask_path):
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

train_images_list, val_images_list, train_masks_list, val_masks_list = train_test_split(image_list, mask_list, test_size=0.2, shuffle=True, random_state=10)
test_images_list, val_images_list, test_masks_list, val_masks_list = train_test_split(val_images_list, val_masks_list, test_size=0.5, shuffle=True, random_state=10)

height, width = 192, 192
resize = (height, width)
train_images = np.zeros((len(train_images_list), height, width, 3), dtype=np.uint8)
train_masks = np.zeros((len(train_images_list), height, width, 1), dtype=np.uint8)
val_images = np.zeros((len(val_images_list), height, width, 3), dtype=np.uint8)
val_masks = np.zeros((len(val_images_list), height, width, 1), dtype=np.uint8)
test_images = np.zeros((len(test_images_list), height, width, 3), dtype=np.uint8)
test_masks = np.zeros((len(test_images_list), height, width, 1), dtype=np.uint8)

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

for n in tqdm(range(len(train_images_list))):
    image = tf.io.read_file(train_images_list[n])
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.uint8)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(train_masks_list[n])
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')
    mask_road = np.zeros((height, width, 1), dtype=np.uint8)
    mask_road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 1
    mask_road[np.where(mask == 6)[0], np.where(mask == 6)[1]] = 2
    train_images[n] = image
    train_masks[n] = mask_road
for n in tqdm(range(len(val_images_list))):
    image = tf.io.read_file(val_images_list[n])
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.uint8)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(val_masks_list[n])
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')
    mask_road = np.zeros((height, width, 1), dtype=np.uint8)
    mask_road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 1
    mask_road[np.where(mask == 6)[0], np.where(mask == 6)[1]] = 2
    val_images[n] = image
    val_masks[n] = mask_road
for n in tqdm(range(len(test_images_list))):
    image = tf.io.read_file(test_images_list[n])
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.uint8)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(test_masks_list[n])
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')
    mask_road = np.zeros((height, width, 1), dtype=np.uint8)
    mask_road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 1
    mask_road[np.where(mask == 6)[0], np.where(mask == 6)[1]] = 2
    test_images[n] = image
    test_masks[n] = mask_road

# plt.subplot(2,2,1)
# plt.imshow(images[0])
# plt.subplot(2,2,2)
# plt.imshow(masks[0])
# test = images[0]
# plt.subplot(2,2,3)
# plt.imshow(test)
# plt.subplot(2,2,4)
# plt.imshow(train_masks[0])
# plt.show()

# del images, masks, image_list, image_path, mask_list, mask_path

plt.subplots(figsize=(15, 5))
for i in range(5):
    N = random.randint(0, int(len(train_images)))
    plt.subplot(5, 3, i * 3 + 1)
    plt.imshow(train_images[N])
    plt.subplot(5, 3, i * 3 + 2)
    plt.imshow(train_masks[N])
    plt.subplot(5, 3, i * 3 + 3)
    plt.imshow(train_images[N])
    plt.imshow(train_masks[N], alpha=0.5)
plt.savefig(f'{main_path}train_{height}_{width}.png')
plt.show()

plt.subplots(figsize=(15, 5))
for i in range(5):
    N = random.randint(0, int(len(val_images)))
    plt.subplot(5, 3, i * 3 + 1)
    plt.imshow(val_images[N])
    plt.subplot(5, 3, i * 3 + 2)
    plt.imshow(val_masks[N])
    plt.subplot(5, 3, i * 3 + 3)
    plt.imshow(val_images[N])
    plt.imshow(val_masks[N], alpha=0.5)
plt.savefig(f'{main_path}val_{height}_{width}.png')
plt.show()

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

np.save(main_path+f'train_images_{height}_{width}.npy', train_images)
np.save(main_path+f'train_masks_{height}_{width}.npy', train_masks)
np.save(main_path+f'val_images_{height}_{width}.npy', val_images)
np.save(main_path+f'val_masks_{height}_{width}.npy', val_masks)
np.save(main_path+f'test_images_{height}_{width}.npy', test_images)
np.save(main_path+f'test_masks_{height}_{width}.npy', test_masks)
print("Complete saving data")

