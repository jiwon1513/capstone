import os

import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

file_path = 'D:/download/dataB/'
image_path = file_path + 'CameraRGB/'
mask_path = file_path + 'CameraSeg/'
save_image = file_path + 'resizeRGB/'
save_mask = file_path + 'resizeSeg/'

image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)

image_list = [image_path+i for i in image_list]
mask_list = [mask_path+i for i in mask_list]

masks = np.zeros((len(image_list), 600, 800, 1), dtype=np.uint8)

import natsort
image_list = natsort.natsorted(image_list)
mask_list = natsort.natsorted(mask_list)

for n in tqdm(range(3)):
    image = Image.open(image_list[n])
    resize_image = image.resize((768, 576))
    resize_image.save(save_image + str(n) + '.png', "png", quality=100)

    mask = imageio.imread(mask_list[n])
    mask_road = np.zeros((600, 800, 1), dtype=np.int8)
    mask_road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 1
    np.save(save_mask + str(n) + '.npy', mask_road)

img = imageio.imread(save_image + "0.png")
mask = np.load(save_mask + "0.npy")
mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

print(np.asarray(img).shape)
print(np.asarray(mask).shape)

N = 0 # image index
fig, arr = plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img)
arr[0].set_title('Image')
arr[1].imshow(mask, cmap='Paired')
arr[1].set_title('Segmentation')
plt.show()