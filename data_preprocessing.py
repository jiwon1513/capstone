from numpy.random import seed
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

file_path = 'D:/download/dataB/'
image_path = file_path + 'CameraRGB/'
mask_path = file_path + 'CameraSeg/'
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)
image_list = [image_path+i for i in image_list]
mask_list = [mask_path+i for i in mask_list]

import natsort
image_list = natsort.natsorted(image_list)
mask_list = natsort.natsorted(mask_list)

import sys
for i in range(10):
    if image_list[i].split('/')[-1] != mask_list[i].split('/')[-1]:
        print("input list sort error!")
        sys.exit(0)
print("Complete sorting input lists!")

# # show test image
# N = 0 # image index
# img = imageio.imread(image_list[N])
# mask = imageio.imread(mask_list[N])
# mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

height, width = 600, 800
images = np.zeros((len(image_list), height, width, 3), dtype=np.uint8)
masks = np.zeros((len(image_list), height, width, 1), dtype=np.uint8)

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
    img = imageio.imread(image_list[n])

    mask = imageio.imread(mask_list[n])
    mask_road = np.zeros((600, 800, 1), dtype=np.int8)
    mask_road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 1

    images[n] = resize(img, (height, width, 3))
    masks[n] = resize(mask_road, (height, width, 1))
    # images[n] = img
    # masks[n] = mask_road

np.random.seed(123)
shuffle_ids = np.array([i for i in range(len(masks))])
np.random.shuffle(shuffle_ids)
train_ids = shuffle_ids[:int(len(masks)*0.8)]
val_ids = shuffle_ids[int(len(masks)*0.8):int(len(masks)*0.8+100)]
test_ids = shuffle_ids[int(len(masks)*0.8+100):]

train_images, train_masks = images[train_ids], masks[train_ids]
val_images, val_masks = images[val_ids], masks[val_ids]
test_images, test_masks = images[test_ids], masks[test_ids]

del images, masks, mask
print(train_images.shape, val_images.shape, test_images.shape)
plt.imshow(train_images[1].reshape(height, width, 3))
plt.show()

np.save(file_path+'train_images.npy', train_images)
np.save(file_path+'train_masks.npy', train_masks)
np.save(file_path+'val_images.npy', val_images)
np.save(file_path+'val_masks.npy', val_masks)
np.save(file_path+'test_images.npy', test_images)
np.save(file_path+'test_masks.npy', test_masks)
print("Complete saving data")

