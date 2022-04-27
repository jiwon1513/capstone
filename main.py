import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.models import load_model

import PSPNet
import ResNet
import UNet
import _UNet

file_path = 'D:/download/dataB/'

# load images
train_images=np.load(file_path+'train_images.npy')
train_masks=np.load(file_path+'train_masks.npy')
val_images=np.load(file_path+'val_images.npy')
val_masks=np.load(file_path+'val_masks.npy')
test_images=np.load(file_path+'test_images.npy')
test_masks=np.load(file_path+'test_masks.npy')

# train_images=np.load(file_path+'train_images2.npy')
# train_masks=np.load(file_path+'train_masks2.npy')
# val_images=np.load(file_path+'val_images2.npy')
# val_masks=np.load(file_path+'val_masks2.npy')
# test_images=np.load(file_path+'test_images2.npy')
# test_masks=np.load(file_path+'test_masks2.npy')

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


# model = PSPNet.pspnet(1, height, width, 3)
# model = ResNet.ResNet_UNet_2(1, 600 , 800, 3)

optimizer = [ # tf.keras.optimizers.Adam(lr=0.005, decay=0.0),
              # tf.keras.optimizers.Adagrad(lr=0.01, decay=0.0),
             tf.keras.optimizers.Adadelta(lr=1, decay=0.0)]
earlystop = [EarlyStopping(patience=12, verbose=1),
             EarlyStopping(patience=9, verbose=1)]

for op, ea in zip(optimizer, earlystop):
    model = UNet.UNet(1, height, width, 3)

    model.compile(optimizer=op, loss='binary_crossentropy', metrics=['accuracy'])

    MODEL_SAVE_FOLDER_PATH=file_path+'temp/'
    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'

    callbacks = [
        ea,
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
    model.save(file_path + 'results/' + model_name + '_'
               + (str(op).split('.')[-1]).split(' ')[0] + '_' + str(ea.patience) + '.h5')

    # save plot
    # accuracy plot
    plt.subplot(2, 1, 1)
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title(model_name + ' accuracy')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')

    # loss plot
    plt.subplot(2, 1, 2)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title(model_name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig(file_path + 'results/' + model_name + '_'
                + (str(op).split('.')[-1]).split(' ')[0] + '_' + str(ea.patience)
                + '_plot.png')
    # plt.show()

    del model, results

for op, ea in zip(optimizer, earlystop):
    model_name = "UNet"
    loaded_model = load_model(file_path + 'results/' + model_name + '_'
               + (str(op).split('.')[-1]).split(' ')[0] + '_' + str(ea.patience) + '.h5')
    NUMBER = 1
    my_preds = loaded_model.predict(np.expand_dims(test_images[NUMBER], 0))
    my_preds = my_preds.flatten()
    my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])

    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(my_preds.reshape(600, 800))
    ax1.set_title('prediction')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(test_masks[NUMBER].reshape(600, 800))
    ax2.set_title('real')
    ax2.axis("off")

    plt.savefig(file_path + 'results/' + model_name + '_'
                + (str(op).split('.')[-1]).split(' ')[0] + '_' + str(ea.patience)
                + '_plot.png')
    # plt.show()