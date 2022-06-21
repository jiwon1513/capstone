from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, concatenate, \
    Activation
from tensorflow.python.keras.applications.resnet import ResNet50
from types import MethodType
import ResNet


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=224,
          input_width=224, channels=3):

    img_input = Input(shape=(input_height, input_width, channels))
    levels = encoder(input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid')(o)

    model = Model(img_input, o)
    # model.output_width = input_width
    # model.output_height = input_height
    # model.n_classes = n_classes
    # model.input_height = input_height
    # model.input_width = input_width

    return model


def ResNet_UNet(n_classes,  input_height=224, input_width=224, channels=3):

    model = _unet(n_classes, ResNet.get_resnet50_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_unet"
    return model


def ResNet_UNet_3(n_classes=13, h=224, w=224, n=3):
    input = Input(shape=(h, w, n))

    resnet = ResNet50(weights='imagenet', include_top=False)

    o = resnet(input)

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((4, 4)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((4, 4)))(o)

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid')(o)

    model = Model(input, o)

    return model