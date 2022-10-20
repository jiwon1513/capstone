import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
from utils.utils import select_device, lane_line_mask, LoadImages

weights = 'data/weights/yolopv2.pt'
# source = 'data/example.jpg'
device = '0'

model = torch.jit.load(weights)
device = select_device(device)
model = model.to(device)
model.half()
model.eval()

stride = 32
imgsz = 640

main_path = 'E:/dataset/'
image_path = main_path + "test/RGB/"
image_list = []
image_temp = os.listdir(image_path)
image_list.extend([image_path + i for i in image_temp])

figure, ax = plt.subplots(figsize=(9, 4))

for source in image_list:
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        start = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        _, _, ll = model(img)
        ll_seg_mask = lane_line_mask(ll)

        img0 = cv2.imread(path)
        img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
        plt.imshow(img0)
        plt.imshow(ll_seg_mask, alpha=0.5)
        # np.save("test.txt", ll_seg_mask)
        # plt.show()

        plt.draw()
        plt.pause(0.01)
        figure.clear()

        # f = open("text.txt", 'w')
        # for i in range(len(ll_seg_mask)):
        #     for j in range(len(ll_seg_mask[0])):
        #         f.write(str(ll_seg_mask[i][j]) + ' ')
        #     f.write('\n')
        # f.close()
        print(start - time.time())