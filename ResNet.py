from tensorflow import keras
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import layers



pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                 "releases/download/v0.2/" \
                 "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


def one_side_pad(x):
    x = ZeroPadding2D((1, 1))(x)
    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


def identity_block(input_tensor, kernel_size, filters):

    filters1, filters2, filters3 = filters

    bn_axis = 3

    x = Conv2D(filters1, (1, 1))(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    bn_axis = 3

    x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization(axis=bn_axis)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def get_resnet50_encoder(input_height=224,  input_width=224,
                         pretrained='imagenet', channels=3):

    assert input_height % 32 == 0
    assert input_width % 32 == 0


    img_input = Input(shape=(input_height, input_width, channels))
    bn_axis = 3


    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    f5 = x

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # f6 = x

    if pretrained == 'imagenet':
        weights_path = keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(weights_path, by_name=True, skip_mismatch=True)

    return [f1, f2, f3, f4, f5]

def ResNet_UNet_2(c=1, h=600, w=800, n=3):
    input = Input(shape=(h, w, n))

    resnet = ResNet50(weights='imagenet', include_top=False)

    x = resnet(input)
    # x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # x = Dense(1024, activation='sigmoid')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # output
    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    c8 = Conv2D(8, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(8, (3, 3), activation='relu', padding='same')(c8)

    output = Conv2D(c, (1, 1), activation='sigmoid')(c8)

    # model
    model = Model(input, output)

    return model