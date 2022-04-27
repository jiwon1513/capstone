import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tqdm import tqdm
from PIL import Image


file_path = 'D:/download/dataB/'

train_images=np.load(file_path+'train_images2.npy')
train_masks=np.load(file_path+'train_masks2.npy')

# print(len(train_images)) # 800
# print(train_images.shape)   # (800, 600, 800, 3)

# height, width = 600, 800
# images = np.zeros((len(image_list), height, width, 3), dtype=np.int16)
# image = train_images[0]
# masks = train_masks[0]
aug_images = []
aug_masks = []

for n in tqdm(range(1)):    # https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html
    image, masks = train_images[n], train_masks[n]
    segmap = SegmentationMapsOnImage(masks, shape=image.shape)

    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.10]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=True)

    # images_aug = []
    # segmaps_aug = []

    # aug_images.append(image)
    # aug_masks.append(masks)

    images_aug, segmaps_aug = seq(image=image, segmentation_maps=segmap)
    aug_images.append(images_aug)
    label = Image.fromarray(segmaps_aug.draw(size=images_aug.shape[:2])[0]).convert('P')
    aug_masks.append(label)
    imageio.imwrite("sample_label_1.png", masks)
    imageio.imwrite("sample_label_2.png", segmaps_aug.draw(size=images_aug.shape[:2])[0])
    imageio.imwrite("sample_label_3.png", label)
    print(masks, "\n**********************\n")
    print(segmaps_aug.draw(size=images_aug.shape[:2])[0], "\n**********************\n")
    print(label, "\n**********************\n")


print("Success Augmentation")
print(train_masks.shape)
print(np.array(label).shape)
# np.save(file_path + 'train_images_aug.npy', aug_images)
# np.save(file_path + 'train_masks_aug.npy', aug_masks)
print("Saved Data")



# # 원본 코드의 이미지 append 이후 파일 출력
# cells=[]
# for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
#     cells.append(image)                                         # column 1
#     cells.append(segmap.draw(size=image_aug.shape[:2])[0])                # column 2
#     cells.append(image_aug)                                     # column 3
#     cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])       # column 4
#
# grid_image = ia.draw_grid(cells, cols=4)
# imageio.imwrite("example_segmaps2.jpg", grid_image)


# # 원본코드, 처리된 이미지 저장
# aug_images = []
# aug_masks = []
#
# for image, masks in zip(train_images, train_masks):
#
#     segmap = SegmentationMapsOnImage(masks, shape=image.shape)
#
#     seq = iaa.Sequential([
#         iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
#         iaa.Sharpen((0.0, 1.0)),       # sharpen the image
#         iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
#         iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
#     ], random_order=True)
#
#     images_aug = []
#     segmaps_aug = []
#     for _ in range(5):
#         images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
#         images_aug.append(images_aug_i)
#         segmaps_aug.append(segmaps_aug_i)
#
#     for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
#         imageio.imwrite("test1.jpg", image)
#         imageio.imwrite("test2.jpg", segmap_aug.draw(size=image_aug.shape[:2])[0])