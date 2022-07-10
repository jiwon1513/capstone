from numpy.random import seed
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

file_path = 'E:/'
image_path = file_path + 'carla_test/'
mask_path = file_path + 'seg/'
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)
image_list = [image_path+i for i in image_list]
mask_list = [mask_path+i for i in mask_list]

save_path = file_path + 'dataset/dataB/'
save_image = save_path + 'cameraRGB/'
save_mask = save_path + 'cameraSeg/'

import natsort
image_list = natsort.natsorted(image_list)
mask_list = natsort.natsorted(mask_list)
i, j, k = 0, 0, 0

while(True):
    if int(image_list[i].split('/')[-1].split('.')[0]) > int(mask_list[j].split('/')[-1].split('.')[0]):
        j += 1
    elif int(image_list[i].split('/')[-1].split('.')[0]) < int(mask_list[j].split('/')[-1].split('.')[0]):
        i += 1
    else:
        os.replace(image_list[i], save_image + str(k) + '.png')
        os.replace(mask_list[j], save_mask + str(k) + '.png')
        i += 1
        j += 1
        k += 1