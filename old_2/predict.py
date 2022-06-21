import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model

import image_preprocessing
from image_preprocessing import data_generator, dataset

train_image_paths, val_image_paths, train_mask_paths, val_mask_paths, validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths = dataset()
batch_size = 16
buffer_size = 500
train_dataset = data_generator(train_image_paths, train_mask_paths, buffer_size, batch_size)
validation_dataset = data_generator(validation_image_paths, validation_mask_paths, buffer_size, batch_size)
test_dataset = data_generator(test_image_paths, test_mask_paths, buffer_size, batch_size)


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


model = load_model('carla-image-segmentation-model.h5')
show_predictions(train_dataset, 6)
show_predictions(validation_dataset, 6)
show_predictions(test_dataset, 6)


show_predictions(False, 0)