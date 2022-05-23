import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
from tensorflow.python.client import device_lib
import tensorflow as tf

from PSPNet import pspnet
from UNet import unet_model
from _UNet import ResNet_UNet, ResNet_UNet_3
from image_preprocessing import dataset, data_generator

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
device_lib.list_local_devices()
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

train_image_paths, val_image_paths, train_mask_paths, val_mask_paths, validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths = dataset()
warnings.filterwarnings('ignore')

batch_size = 8
buffer_size = 200

train_dataset = data_generator(train_image_paths, train_mask_paths, buffer_size, batch_size)
validation_dataset = data_generator(validation_image_paths, validation_mask_paths, buffer_size, batch_size)
test_dataset = data_generator(test_image_paths, test_mask_paths, buffer_size, batch_size)

# img_height = 224
# img_width = 224
# num_channels = 3
# filters = 32
# n_classes = 13

# model = unet_model((img_height, img_width, num_channels), filters=32, n_classes=13)
# model = ResNet_UNet(n_classes=n_classes)
# model = pspnet(13, 384, 576, 3)
model = ResNet_UNet_3(13, 224, 224, 3)
model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
epochs = 30

history = model.fit(train_dataset,
                    validation_data = validation_dataset,
                    epochs = epochs,
                    # verbose=1,
                    callbacks = [callback, reduce_lr],
                    batch_size = batch_size,
                    shuffle = True)

acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# model.save('carla-image-segmentation-model.h5')

train_loss, train_accuracy = model.evaluate(train_dataset, batch_size = 32)
validation_loss, validation_accuracy = model.evaluate(validation_dataset, batch_size = 32)
test_loss, test_accuracy = model.evaluate(test_dataset, batch_size = 32)

print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')