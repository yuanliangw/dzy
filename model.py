# model.py
from keras.models import Model
from keras import layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf

class MaskFormerHead(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(MaskFormerHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # 将特征图展平并进行分类
        x = self.flatten(inputs)
        return self.dense(x)

def VGG16_unet_model(input_size=(288, 384, 3), use_batchnorm=False, if_transfer=False, if_local=True, use_maskformer=False):
    axis = 3
    kernel_initializer = 'he_normal'
    origin_filters = 32
    weights = None
    model_path = 'models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if if_transfer:
        if if_local:
            weights = model_path
        else:
            weights = 'imagenet'
    vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_size)
    for layer in vgg16.layers:
        layer.trainable = False

    output = vgg16.layers[17].output
    up6 = layers.Conv2D(origin_filters * 8, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(output))
    merge6 = layers.concatenate([vgg16.layers[13].output, up6], axis=axis)
    conv6 = layers.Conv2D(origin_filters * 8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = layers.Conv2D(origin_filters * 8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = layers.Conv2D(origin_filters * 4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([vgg16.layers[9].output, up7], axis=axis)
    conv7 = layers.Conv2D(origin_filters * 4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = layers.Conv2D(origin_filters * 4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = layers.Conv2D(origin_filters * 2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([vgg16.layers[5].output, up8], axis=axis)
    conv8 = layers.Conv2D(origin_filters * 2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = layers.Conv2D(origin_filters * 2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = layers.Conv2D(origin_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([vgg16.layers[2].output, up9], axis=axis)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    if use_maskformer:
        # 使用 MaskFormer 分割头
        conv_out = layers.Conv2D(origin_filters, 1, padding='same')(conv9)
        mask_head = MaskFormerHead(num_classes=3)(conv_out)  # 假设有3个类别
    else:
        # 使用卷积分割头
        conv10 = layers.Conv2D(3, 1, activation='sigmoid')(conv9)
        mask_head = conv10

    model = Model(inputs=vgg16.input, outputs=mask_head)
    return model

# 示例调用
model = VGG16_unet_model(use_maskformer=True)  # 使用 MaskFormer
print(model.summary())
