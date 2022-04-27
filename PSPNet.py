import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


def pool_block(feats, pool_factor):

    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = K.resize_images(x,
                        height_factor=strides[0],
                        width_factor=strides[1],
                        data_format="channels_last",
                        interpolation='bilinear')

    return x


def _pspnet(n_classes, encoder,  input_height=384, input_width=576, channels=3):

    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    o = Conv2D(512, (1, 1), use_bias=False, name="seg_feats" )(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), padding='same')(o)
    o = resize_image(o, (8, 8))

    o = (Activation('sigmoid'))(o)
    model = Model(img_input, o)

    return model


def resnet50_pspnet(n_classes,  input_height=384, input_width=576, channels=3):

    model = _pspnet(n_classes, get_resnet50_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_pspnet"
    return model


def pspnet(n_classes,  input_height=384, input_width=576, channels=3):

    model = _pspnet(n_classes, vanilla_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "pspnet"
    return model


def vanilla_encoder(input_height=224,  input_width=224, channels=3):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, channels))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(filter_size, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(128, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad)))(x)
        x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size)))(x)
        levels.append(x)

    return img_input, levels