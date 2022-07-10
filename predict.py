import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model
import image_preprocessing
from image_preprocessing import data_generator, dataset
import numpy as np




file_path = 'E:/dataset/'


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)

    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(dataset, num):
    sample_image, sample_mask = image_preprocessing.read_image('sample.png', 'sample.png')
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


def loadModel(num):
    i = num
    if i == 0:
        height, width = 256, 256
        name = "UNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 1:
        height, width = 192, 192
        name = "PSPNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 2:
        height, width = 224, 224
        name = "ResNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 3:
        height, width = 224, 224
        name = "ResNet2"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')
    elif i == 4:
        height, width = 192, 192
        name = "ResNet_PSPNet"
        model = load_model(file_path + 'models/carla-image-segmentation-' + name + '.h5')

    print('load images')
    return model, height, width, name

# show_predictions(train_dataset, 6)
# show_predictions(validation_dataset, 6)
# show_predictions(test_dataset, 6)

# train_loss, train_accuracy = model.evaluate(train_image, train_mask, batch_size=32)
# validation_loss, validation_accuracy = model.evaluate(val_image, val_mask, batch_size=32)
# test_loss, test_accuracy = model.evaluate(test_image, test_mask, batch_size=32)

# print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
# print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
# print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')


def main():
    num = 3
    model, height, width, name = loadModel(num)
    # train_image = np.load(file_path + f'train_images_{height}_{width}.npy')
    # train_mask = np.load(file_path + f'train_masks_{height}_{width}.npy')
    # val_image = np.load(file_path + f'val_images_{height}_{width}.npy')
    # val_mask = np.load(file_path + f'val_masks_{height}_{width}.npy')
    test_image = np.load(file_path + f'test_test_images_{height}_{width}.npy')
    test_mask = np.load(file_path + f'test_test_masks_{height}_{width}.npy')

    plt.subplots(figsize=(15, 5))
    for i in range(5):
        N = i * 100
        my_preds = model.predict(np.expand_dims(test_image[N], 0))
        # my_preds = my_preds.flatten()
        # my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])
        plt.subplot(5, 3, i * 3 + 1)
        plt.imshow(test_image[N])
        plt.subplot(5, 3, i * 3 + 2)
        plt.imshow(test_mask[N])
        plt.subplot(5, 3, i * 3 + 3)
        plt.imshow(my_preds.reshape(height, width, 3))
    # plt.savefig(f'{main_path}test_{height}_{width}.png')
    plt.show()

if __name__ == "__main__":
    main()