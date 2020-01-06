from keras import layers, models
from keras.initializers import glorot_normal, glorot_uniform
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
from keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, AveragePooling1D, Dense, Dropout, \
    Flatten, GlobalAveragePooling2D, LSTM, Embedding, GlobalAveragePooling1D, Conv1D, MaxPooling1D, TimeDistributed
from keras import regularizers


def HOTCNN01(input_shape, n_class):
    x = layers.Input(shape=input_shape)
    block1 = Conv2D(filters=128, kernel_size=(4, 15), padding='valid', activation='sigmoid')(x)
    block1 = MaxPooling2D(pool_size=(1, 2))(block1)
    flat = Flatten()(block1)
    outputs = Dense(n_class, activation='sigmoid')(flat)
    model = models.Model(inputs=[x], outputs=[outputs])
    return model


# python onehot_nets.py -o Bacillus --model HOT_CNN_BACILLUS_01 --coding onehot --epochs 300 --patience 0 --cv 5 --seeds 3 --n_class 1
def HOT_CNN_BACILLUS_01(input_shape, n_class):
    # Input layer
    x = layers.Input(shape=input_shape)

    # ConvPool block 2
    block1 = layers.Conv2D(
        filters=100,
        kernel_size=(4, 15),
        padding='valid',
        activation='relu')(x)
    block1 = layers.MaxPooling2D(
        pool_size=(1, 2),
        strides=(1, 2))(block1)

    # ConvPool block 2
    block2 = layers.Conv2D(
        filters=250,
        kernel_size=(1, 17),
        strides=(1, 2),
        padding='valid',
        activation='relu')(block1)
    block2 = layers.MaxPooling2D(
        pool_size=(1, 2),
        strides=(1, 2))(block2)

    # Flat tensors
    flat = layers.Flatten()(block2)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flat)

    # Classification layer
    outputs = layers.Dense(n_class, activation='sigmoid')(dense1)

    # Create model object
    model = models.Model(inputs=[x], outputs=[outputs])

    return model

# python onehot_nets.py -o Bacillus --model EMB_CNN_BACILLUS_01 --coding embedding --epochs 300 --patience 0 --cv 5 --seeds 3 --n_class 1
def EMB_CNN_BACILLUS_01(input_shape, n_class):
    # Input layer
    x = layers.Input(shape=input_shape)

    emb = Embedding(4, 9, input_length=input_shape[0])(x)

    # ConvPool block 2
    block1 = layers.Conv1D(
        filters=100,
        kernel_size=15,
        padding='valid',
        activation='relu')(emb)
    block1 = layers.MaxPooling1D(
        pool_size=2,
        strides=2)(block1)

    # ConvPool block 2
    block2 = layers.Conv1D(
        filters=250,
        kernel_size=17,
        strides=2,
        padding='valid',
        activation='relu')(block1)
    block2 = layers.MaxPooling1D(
        pool_size=2,
        strides=2)(block2)

    # Flat tensors
    flat = layers.Flatten()(block2)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flat)

    # Classification layer
    outputs = layers.Dense(n_class, activation='sigmoid')(dense1)

    # Create model object
    model = models.Model(inputs=[x], outputs=[outputs])

    return model

def HOT_RES_BACILLUS_01(input_shape, n_class):
    n_filters = 128
    # Input layer
    x = layers.Input(shape=input_shape)

    conv_x = Conv2D(filters=128, kernel_size=(4, 15), padding='same')(x)
    # conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    # conv_x = MaxPooling2D((1, 3), strides=(1, 1), padding='same')(conv_x)

    conv_y = Conv2D(filters=128, kernel_size=(1, 17), padding='same')(conv_x)
    # conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
    # conv_y = MaxPooling2D((1, 2), strides=(1, 1), padding='same')(conv_y)

    # conv_z = Conv2D(filters=128, kernel_size=(1, 21), padding='same')(conv_y)
    # conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv2D(filters=128, kernel_size=(4, 1), padding='same')(x)
    # shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = Add()([shortcut_y, conv_y])
    output_block_1 = Activation('relu')(output_block_1)

    gap_layer = GlobalAveragePooling2D()(output_block_1)

    # Fully connected layers
    # fully1 = Dense(128, activation='relu', name='fc1' + str(n_class))(gap_layer)
    fully1 = Dropout(.3)(gap_layer)
    outputs = Dense(n_class, activation='sigmoid', name='fc' + str(n_class))(fully1)

    # Create model object
    model = models.Model(inputs=[x], outputs=[outputs])

    return model

# python onehot_nets.py -o Bacillus --model EMB_RES_BACILLUS_01 --coding embedding --epochs 300 --patience 0 --cv 5 --seeds 3 --n_clas
def EMB_RES_BACILLUS_01(input_shape, n_class):
    n_filters = 128
    # Input layer
    x = layers.Input(shape=input_shape)

    emb = Embedding(4, 9, input_length=input_shape[0])(x)

    conv_x = Conv1D(filters=128, kernel_size=15, padding='same')(emb)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = MaxPooling1D(2, strides=1, padding='same')(conv_x)

    conv_y = Conv1D(filters=128, kernel_size=17, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
    conv_y = MaxPooling1D(2, strides=1, padding='same')(conv_y)

    conv_z = Conv1D(filters=128, kernel_size=21, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=128, kernel_size=1, padding='same')(emb)
    # shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = Add()([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)



    gap_layer = GlobalAveragePooling1D()(output_block_1)
    gap_layer = Activation('relu')(gap_layer)
    # gap_layer = BatchNormalization()(gap_layer)


    # Fully connected layers
    fully1 = Dropout(.2)(gap_layer)
    # fully1 = Dense(128, activation='relu', name='fully1' + str(n_class))(fully1)
    # fully1 = Dropout(.1)(fully1)
    outputs = Dense(n_class, activation='sigmoid', name='fc' + str(n_class))(fully1)

    # Create model object
    model = models.Model(inputs=[x], outputs=[outputs])

    return model

# python onehot_nets.py -o Bacillus --model EMB_RES_BACILLUS_02 --coding embedding --epochs 300 --patience 0 --cv 5 --seeds 3 --n_class 1
def EMB_RES_BACILLUS_02(input_shape, n_class):
    n_filters = 128
    emb_dim = 9

    pool_types = ('avg', 'max')
    pool_type = 0
    pool_size = 3

    # INPUT
    x = layers.Input(shape=input_shape)

    # EMBEDDING
    emb = Embedding(4, emb_dim, input_length=input_shape[0])(x)

    # =========================
    # RESIDUAL BLOCK 01 - START
    # =========================

    # CONV X
    conv_x = Conv1D(filters=n_filters, kernel_size=15, padding='same')(emb)
    conv_x = Activation('relu')(conv_x)
    conv_x = BatchNormalization()(conv_x)
    # POOLING X
    if pool_types[pool_type] == 'max':
        conv_x = MaxPooling1D(pool_size, strides=1, padding='same')(conv_x)
    elif pool_types[pool_type] == 'avg':
        conv_x = AveragePooling1D(pool_size, strides=1, padding='same')(conv_x)

    # CONV Y
    conv_y = Conv1D(filters=n_filters, kernel_size=17, padding='same')(conv_x)
    conv_y = Activation('relu')(conv_y)
    conv_y = BatchNormalization()(conv_y)
    # POOLING Y
    if pool_types[pool_type] == 'max':
        conv_y = MaxPooling1D(pool_size, strides=1, padding='same')(conv_y)
    elif pool_types[pool_type] == 'avg':
        conv_y = AveragePooling1D(pool_size, strides=1, padding='same')(conv_y)

    # CONV Z
    conv_z = Conv1D(filters=n_filters, kernel_size=21, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # SHORTCUT
    shortcut_y = Conv1D(filters=n_filters, kernel_size=1, padding='same')(emb)
    shortcut_y = BatchNormalization()(shortcut_y)

    # =========================
    # RESIDUAL BLOCK 01 - END
    # =========================
    output_block_1 = Add()([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # POOLING
    gap_layer = GlobalAveragePooling1D()(output_block_1)
    gap_layer = Activation('relu')(gap_layer)
    # gap_layer = BatchNormalization()(gap_layer)

    # FULLY CONNECTED
    fully1 = Dropout(.2)(gap_layer)
    # fully1 = Dense(128, activation='relu', name='fully1' + str(n_class))(fully1)
    # fully1 = Dropout(.1)(fully1)
    outputs = Dense(n_class, activation='sigmoid', name='fc' + str(n_class))(fully1)

    # CREATE MODEL
    model = models.Model(inputs=[x], outputs=[outputs])

    return model

# python onehot_nets.py -o Bacillus --model EMB_RES_BACILLUS_03 --coding embedding --epochs 300 --patience 0 --cv 5 --seeds 3 --n_class 1
def EMB_RES_BACILLUS_03(input_shape, n_class):
    n_filters = 128
    emb_dim = 9

    pool_types = ('avg', 'max', False)
    pool_type = 1
    pool_size = 2

    # INPUT
    x = layers.Input(shape=input_shape)

    # EMBEDDING
    emb = Embedding(4, emb_dim, input_length=input_shape[0])(x)

    # =========================
    # RESIDUAL BLOCK 01 - START
    # =========================

    # CONV X
    conv_x = Conv1D(filters=n_filters, kernel_size=15, padding='same')(emb)
    conv_x = Activation('relu')(conv_x)
    conv_x = BatchNormalization()(conv_x)
    # POOLING X
    if pool_types[pool_type] == 'max':
        conv_x = MaxPooling1D(pool_size, strides=1, padding='same')(conv_x)
    elif pool_types[pool_type] == 'avg':
        conv_x = AveragePooling1D(pool_size, strides=1, padding='same')(conv_x)

    # CONV Y
    conv_y = Conv1D(filters=n_filters, kernel_size=17, padding='same')(conv_x)
    conv_y = Activation('relu')(conv_y)
    conv_y = BatchNormalization()(conv_y)
    # POOLING Y
    if pool_types[pool_type] == 'max':
        conv_y = MaxPooling1D(pool_size, strides=1, padding='same')(conv_y)
    elif pool_types[pool_type] == 'avg':
        conv_y = AveragePooling1D(pool_size, strides=1, padding='same')(conv_y)

    # CONV Z
    conv_z = Conv1D(filters=n_filters, kernel_size=21, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # SHORTCUT
    shortcut_y = Conv1D(filters=n_filters, kernel_size=1, padding='same')(emb)
    shortcut_y = BatchNormalization()(shortcut_y)

    # =========================
    # RESIDUAL BLOCK 01 - END
    # =========================
    output_block_1 = Add()([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # POOLING
    gap_layer = GlobalAveragePooling1D()(output_block_1)
    gap_layer = Activation('relu')(gap_layer)
    # gap_layer = BatchNormalization()(gap_layer)

    # FULLY CONNECTED
    fully1 = Dropout(.2)(gap_layer)
    # fully1 = Dense(128, activation='relu', name='fully1' + str(n_class))(fully1)
    # fully1 = Dropout(.1)(fully1)
    outputs = Dense(n_class, activation='sigmoid', name='fc' + str(n_class))(fully1)

    # CREATE MODEL
    model = models.Model(inputs=[x], outputs=[outputs])

    return model

def HOT_RES_BACILLUS_03(input_shape, n_class):
    n_filters = 128
    # Input layer
    x = layers.Input(shape=input_shape)

    conv_x = Conv2D(
        filters=100,
        kernel_size=(4, 15),
        padding='same'
    )(x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_filters, kernel_size=(1, 5), padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_filters, kernel_size=(1, 3), padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv2D(filters=n_filters, kernel_size=(4, 1), padding='same')(x)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = Add()([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = Conv2D(filters=n_filters * 2, kernel_size=(1, 8), padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_filters * 2, kernel_size=(1, 5), padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_filters * 2, kernel_size=(1, 3), padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv2D(filters=n_filters * 2, kernel_size=(1, 1), padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = Add()([shortcut_y, conv_z])
    output_block_2 = Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = Conv2D(filters=n_filters * 2, kernel_size=(1, 8), padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_filters * 2, kernel_size=(1, 5), padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_filters * 2, kernel_size=(1, 3), padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = Add()([shortcut_y, conv_z])
    output_block_3 = Activation('relu')(output_block_3)

    gap_layer = GlobalAveragePooling2D()(output_block_3)
    # X = AveragePooling2D((1, 2), name="avg_pool")(X)

    # Fully connected layers
    # X = Flatten()(X)
    X = Dropout(.2)(gap_layer)
    outputs = Dense(n_class, activation='sigmoid', name='fc' + str(n_class), kernel_initializer=glorot_uniform(seed=0))(
        X)

    # Create model object
    model = models.Model(inputs=[x], outputs=[outputs])

    return model
