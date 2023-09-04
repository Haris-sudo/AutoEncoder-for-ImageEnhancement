import tensorflow as tf
from tensorflow.keras import models, layers


#modular conv block (used for encoder)
def conv_block(x, filters, kernel_size, strides, padding='same'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

#modular deconv block (used for decoder)
def deconv_block(x, filters, kernel_size, strides, padding='same'):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape) # determining the input shape

    # Encoder
    conv1 = conv_block(inputs, 32, 3, 1)
    conv2 = conv_block(conv1, 64, 3, 2)
    # conv3 = conv_block(conv2, 256, 3, 2)
    # conv4 = conv_block(conv3, 512, 3, 2, padding='valid')

    # Decoder
    deconv1 = deconv_block(conv2, 64, 3, 2)
    deconv2 = deconv_block(deconv1, 32, 3, 1)
    # deconv3 = deconv_block(deconv2, 128, 3, 2)
    # deconv4 = deconv_block(deconv3, 64, 3, 1)

    outputs = layers.Conv2D(3, 1, activation='sigmoid')(deconv2) #activation was sigmoid 
    return models.Model(inputs=inputs, outputs=outputs)

# input_shape = (256, 256, 3)
# autoencoder = build_autoencoder(input_shape)
#autoencoder.summary()
