from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def unet_mini(n_classes, input_height=600, input_width=800, channels=3):
    img_input = Input(shape=(input_height, input_width, channels))

    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2])
    conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1])
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same' , name="seg_feats")(conv5)

    o = Conv2D(n_classes, (1, 1), padding='same')(conv5)
    o = (Activation('sigmoid'))(o)

    model = Model(inputs=[img_input], outputs=[o])

    return model