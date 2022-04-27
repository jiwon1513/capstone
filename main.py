import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import PSPNet
import ResNet
import UNet
import _UNet

file_path = 'D:/download/dataB/'

# load images
# train_images=np.load(file_path+'train_images.npy')
# train_masks=np.load(file_path+'train_masks.npy')
# val_images=np.load(file_path+'val_images.npy')
# val_masks=np.load(file_path+'val_masks.npy')
# test_images=np.load(file_path+'test_images.npy')
# test_masks=np.load(file_path+'test_masks.npy')

train_images=np.load(file_path+'train_images2.npy')
train_masks=np.load(file_path+'train_masks2.npy')
val_images=np.load(file_path+'val_images2.npy')
val_masks=np.load(file_path+'val_masks2.npy')
test_images=np.load(file_path+'test_images2.npy')
test_masks=np.load(file_path+'test_masks2.npy')

# model data
model_name_list = ['UNet', 'UNet_mini', 'ResNet_UNet', 'PSPNet_UNet']
height, width = 600, 800

# Setting GPU
device_lib.list_local_devices()
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.compat.v1.Session(config=config)

# model = UNet.UNet(1)
# model = PSPNet.pspnet(1, height, width, 3)
model = ResNet.ResNet_UNet_2(1, 600 , 800, 3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

MODEL_SAVE_FOLDER_PATH=file_path+'temp/'
model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'

callbacks = [
    EarlyStopping(patience=12, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
]

results = model.fit(train_images,
                    train_masks,
                    batch_size=16,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=(val_images, val_masks)
                    )

model_name = "UNet"
model.save(file_path + 'results/' + model_name + '.h5')

# save plot
# accuracy plot
plt.subplot(2, 1, 1)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title(model_name + ' model with data accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# loss plot
plt.subplot(2, 1, 2)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title(model_name + ' model with data loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig(file_path + 'results/' + model_name + '_plot.png')
plt.show()