import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def list_image_paths(directory_paths):
    image_paths = []
    for directory in range(len(directory_paths)):
        image_filenames = os.listdir(directory_paths[directory])
        for image_filename in image_filenames:
            image_paths.append(directory_paths[directory] + image_filename)
    return image_paths


def read_image_UNet(image_path, mask_path):
    resize = (256, 256) # UNet
    # resize = (224, 224) # ResNet
    # resize = (384, 576) # PSPNet

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')

    return image, mask


def read_image_ResNet(image_path, mask_path):
    # resize = (256, 256) # UNet
    resize = (224, 224) # ResNet
    # resize = (384, 576) # PSPNet

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')

    return image, mask


def read_image_PSPNet(image_path, mask_path):
    # resize = (256, 256) # UNet
    # resize = (224, 224) # ResNet
    resize = (384, 576) # PSPNet

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, resize, method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, resize, method='nearest')

    return image, mask


def data_generator(image_paths, mask_paths, buffer_size, batch_size, name):
    image_list = tf.constant(image_paths)
    mask_list = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    if name == 'UNet':
        dataset = dataset.map(read_image_UNet, num_parallel_calls=tf.data.AUTOTUNE)
    elif name == 'ResNet':
        dataset = dataset.map(read_image_ResNet, num_parallel_calls=tf.data.AUTOTUNE)
    elif name == 'PSPNet':
        dataset = dataset.map(read_image_PSPNet, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(read_image_UNet, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)

    return dataset


def dataset():
    # main_path = 'D:/carla/'
    # image_path = [main_path + "data" + i +"/" +"data" + i +"/CameraRGB/" for i in ['A', 'B', 'C', 'D', 'E']]
    # mask_path = [main_path + "data" + i +"/" +"data" + i +"/CameraSeg/" for i in ['A', 'B', 'C', 'D', 'E']]
    main_path = 'E:/dataset/'
    image_path = [main_path + "town" + i + "/RGB/" for i in ['5', '6', '7', '10']]
    mask_path = [main_path + "town" + i + "/seg/" for i in ['5', '6', '7', '10']]

    image_paths = list_image_paths(image_path)
    mask_paths = list_image_paths(mask_path)
    number_of_images, number_of_masks = len(image_paths), len(mask_paths)
    print(f"1. There are {number_of_images} images and {number_of_masks} masks in our dataset")
    print(f"2. An example of an image path is: \n {image_paths[0]}")
    print(f"3. An example of a mask path is: \n {mask_paths[0]}")

    number_of_samples = len(image_paths)

    for i in range(1):
        # N = random.randint(0, number_of_samples - 1)
        N = 1

        img = imageio.imread(image_paths[N])
        # mask = imageio.imread(mask_paths[N])
        # mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

        mask = tf.io.read_file(mask_paths[N])
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        mask = tf.image.resize(mask, (600, 800), method='nearest')

        fig, arr = plt.subplots(1, 3, figsize=(20, 8))
        arr[0].imshow(img)
        arr[0].set_title('Image')
        arr[0].axis("off")
        arr[1].imshow(mask)
        arr[1].set_title('Segmentation')
        arr[1].axis("off")
        arr[2].imshow(mask, cmap='Paired')
        arr[2].set_title('Image Overlay')
        arr[2].axis("off")
        plt.show()

    # First split the image paths into training and validation sets
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, train_size=0.8, random_state=0)

    # Keep part of the validation set as test set
    validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths = train_test_split(val_image_paths, val_mask_paths, train_size=0.8, random_state=0)

    print(f'There are {len(train_image_paths)} images in the Training Set')
    print(f'There are {len(validation_image_paths)} images in the Validation Set')
    print(f'There are {len(test_image_paths)} images in the Test Set')

    return train_image_paths, val_image_paths, train_mask_paths, val_mask_paths, validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths