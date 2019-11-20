from keras import layers, models
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence

def CNN01(input_shape, n_class):
    x = layers.Input(shape=input_shape)
    block1 = layers.Conv2D(filters=128, kernel_size=(4, 7), padding='valid', activation='sigmoid')(x)
    block1 = layers.MaxPooling2D(pool_size=(1, 2))(block1)
    flat = layers.Flatten()(block1)
    outputs = layers.Dense(n_class, activation='sigmoid')(flat)
    model = models.Model(inputs=[x], outputs=[outputs])
    return model



# def CNN01():
#     from keras import layers, models
#     from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
#     from keras.preprocessing import sequence
#     """
#     A Capsule Network on MNIST.
#     :param input_shape: data shape, 4d, [None, width, height, channels]
#     :param num_routing: number of routing iterations
#     :return: A Keras Model with 2 inputs and 2 outputs
#     """
#     x = layers.Input(shape=(4, maxlen, 1))
#
#     # Block 1
#
#     # 1x1 down-sample conv
#     conv_x = layers.BatchNormalization()(x)
#     conv_x = layers.Activation('relu')(conv_x)
#     conv_x = layers.Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 2))(conv_x)
#     # LxL conv
#     conv_y = layers.BatchNormalization()(conv_x)
#     conv_y = layers.Activation('relu')(conv_y)
#     conv_y = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(conv_y)
#     # 1x1 up-sample conv
#     conv_z = layers.BatchNormalization()(conv_y)
#     conv_z = layers.Activation('relu')(conv_z)
#     conv_z = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(conv_z)
#     # Identity branch - expand channels for the sum
#     shortcut_y = layers.Conv2D(filters=128, kernel_size=(4, 1), strides=(1, 2))(x)
#     shortcut_y = layers.BatchNormalization()(shortcut_y)
#
#     output_block_1 = layers.add([shortcut_y, conv_z])
#     output_block_1 = layers.Activation('relu')(output_block_1)
#
#     # Block 2
#     # 1x1 down-sample conv
#     conv_x = layers.BatchNormalization()(output_block_1)
#     conv_x = layers.Activation('relu')(conv_x)
#     conv_x = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 2))(conv_x)
#     # LxL conv
#     conv_y = layers.BatchNormalization()(conv_x)
#     conv_y = layers.Activation('relu')(conv_y)
#     conv_y = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(conv_y)
#     # 1x1 up-sample conv
#     conv_z = layers.BatchNormalization()(conv_y)
#     conv_z = layers.Activation('relu')(conv_z)
#     conv_z = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(conv_z)
#     # Identity branch - expand channels for the sum
#     shortcut_y = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 2))(output_block_1)
#     shortcut_y = layers.BatchNormalization()(shortcut_y)
#
#     output_block_2 = layers.add([shortcut_y, conv_z])
#     output_block_2 = layers.Activation('relu')(output_block_2)
#
#     # Block 3
#     # 1x1 down-sample conv
#     conv_x = layers.BatchNormalization()(output_block_2)
#     conv_x = layers.Activation('relu')(conv_x)
#     conv_x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 2))(conv_x)
#     # LxL conv
#     conv_y = layers.BatchNormalization()(conv_x)
#     conv_y = layers.Activation('relu')(conv_y)
#     conv_y = layers.Conv2D(filters=64, kernel_size=(1, 3), padding='same')(conv_y)
#     # 1x1 up-sample conv
#     conv_z = layers.BatchNormalization()(conv_y)
#     conv_z = layers.Activation('relu')(conv_z)
#     conv_z = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same')(conv_z)
#     # Identity branch - expand channels for the sum
#     shortcut_y = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 2))(output_block_2)
#     shortcut_y = layers.BatchNormalization()(shortcut_y)
#
#     output_block_3 = layers.add([shortcut_y, conv_z])
#     output_block_3 = layers.Activation('relu')(output_block_3)
#
#     # Block 4
#     # 1x1 down-sample conv
#     conv_x = layers.BatchNormalization()(output_block_3)
#     conv_x = layers.Activation('relu')(conv_x)
#     conv_x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 2))(conv_x)
#     # LxL conv
#     conv_y = layers.BatchNormalization()(conv_x)
#     conv_y = layers.Activation('relu')(conv_y)
#     conv_y = layers.Conv2D(filters=128, kernel_size=(1, 3), padding='same')(conv_y)
#     # 1x1 up-sample conv
#     conv_z = layers.BatchNormalization()(conv_y)
#     conv_z = layers.Activation('relu')(conv_z)
#     conv_z = layers.Conv2D(filters=1024, kernel_size=(1, 1), padding='same')(conv_z)
#     # Identity branch - expand channels for the sum
#     shortcut_y = layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 2))(output_block_3)
#     shortcut_y = layers.BatchNormalization()(shortcut_y)
#
#     output_block_4 = layers.add([shortcut_y, conv_z])
#     output_block_4 = layers.Activation('relu')(output_block_4)
#
#     drop = layers.Dropout(0.2)(output_block_4)
#
#     gap_layer = layers.GlobalAveragePooling2D()(drop)
#
#     # flat1 = layers.Flatten()(pool2)
#
#     # dense1 = layers.Dense(128, activation='relu')(gap_layer)
#     # drop2 = layers.Dropout(0.1)(dense1)
#     # outputs = layers.Dense(1, activation='sigmoid')(drop2)
#     outputs = layers.Dense(1, activation='sigmoid')(gap_layer)
#
#     return models.Model(inputs=[x], outputs=outputs)