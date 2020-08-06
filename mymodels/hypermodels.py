import numpy as np
from tensorflow import keras
from tensorflow import random as tf_random
from kerastuner import HyperModel
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    MaxPooling2D,
    Input,
    Activation,
    Add,
    Concatenate,
    Embedding,
    BatchNormalization,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import (Adam, Nadam, )


class HotCNNHyperModel(HyperModel):
    def __init__(self, data, num_classes):
        super().__init__()
        input_shapes = list()
        input_types = list()
        for d in data:
            input_shapes.append(d.shape()[1:])
            input_types.append(d.get_encode())
        self.input_shapes = input_shapes
        self.input_types = input_types
        self.num_classes = num_classes

    def define_search_space(self):
        self.num_filters = [32, 64, 128]
        self.default_num_filters = 32
        self.kernel_sizes = [3, 7, 15, 21]
        self.default_kernel_sizes = 7
        self.pool_sizes = [2, 3, 5]
        self.default_pool_sizes = 2
        self.neurons_units = [0, 32, 64, 128, 256]
        self.default_neurons_units = 128

    def setup_input(self):
        inputs = list()
        for i, t in enumerate(self.input_types):
            # Setup input for this branch
            input_shape = self.input_shapes[i]
            # print('input_shape', input_shape)
            x = Input(shape=input_shape)
            if self.input_types[i] == 'categorical':
                n_words = self.k ** 4
                emb_size = (n_words * 2) + 1
                x = Embedding(n_words, emb_size, input_length=input_shape[0])(x)
            inputs.append(x)
        return inputs

    def build(self, hp):

        self.define_search_space()

        # Verify number branches based on the number of inputs
        num_branches = len(self.input_shapes)
        branches = list()
        inputs = self.setup_input()

        # Construct each branch
        for branch in range(num_branches):

            x = inputs[branch]

            # Verify the number of dimensions of this branch input - used to determine the type of used filters
            input_shape_length = len(self.input_shapes[branch])

            # Define number of stacked blocks
            n_blocks = hp.Int('branch-{}_num_blocks'.format(branch), 1, 3)
            for _block in range(n_blocks):

                # Choose number of filters
                block_name = 'branch-{}_block-{}-{}'.format(branch, _block, n_blocks)
                name = block_name + '_n_filters'
                n_filters = hp.Choice(name, values=self.num_filters, default=self.default_num_filters)
                # Choose the size of kernels
                name = block_name + '_k_size'
                k_size = hp.Choice(name, values=self.kernel_sizes, default=self.default_kernel_sizes)
                # Choose the size of pooling filters
                name = block_name + '_p_size'
                p_size = hp.Choice(name, values=self.pool_sizes, default=self.default_pool_sizes)

                # Setup the block
                if input_shape_length == 2:
                    # 1D - One dimensional filters
                    x = Conv1D(filters=n_filters, kernel_size=k_size, padding='same')(x)
                    x = Activation('relu')(x)
                    x = MaxPooling1D(pool_size=p_size, padding='same')(x)
                elif input_shape_length == 3:
                    # 2D - Two dimensional filters
                    height = 1 if _block != 0 else self.input_shapes[branch][2]
                    x = Conv2D(filters=n_filters, kernel_size=(k_size, height), padding='same')(x)
                    x = Activation('relu')(x)
                    x = MaxPooling2D(pool_size=(p_size, 1), padding='same')(x)
                else:
                    print('ERROR: Invalid dimension ({}).'.format(input_shape_length))
                # Flatten the last block of 2D layers
                if (_block + 1) == n_blocks:
                    x = Flatten()(x)

            # Store this branch on the list to further concatenation
            branches.append(x)

        # Concatenate all branches
        if num_branches > 1:
            x = Concatenate()(branches)

        # Apply Dropout regularizer
        drop = hp.Float('dropout', min_value=.0, max_value=.6, step=.1, default=.2)
        x = Dropout(rate=drop)(x)

        # Define the fully connected layer
        n_neurons = hp.Choice('neurons_units', values=self.neurons_units, default=self.default_neurons_units)
        if n_neurons > 0:
            x = Dense(units=n_neurons, activation='relu')(x)

        # Define the output layer
        x = Dense(units=self.num_classes, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x)

        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        model.compile(optimizer=Adam(lr=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy']
                      )

        return model


class ResNetHyperModel(HyperModel):
    def __init__(self, data, num_classes):
        super().__init__()
        input_shapes = list()
        input_types = list()
        for d in data:
            input_shapes.append(d.shape()[1:])
            input_types.append(d.get_encode())
        self.input_shapes = input_shapes
        self.input_types = input_types
        self.num_classes = num_classes

    def define_search_space(self):
        self.num_filters = [32, 64, 128]
        self.default_num_filters = 32
        self.kernel_sizes = [3, 7, 15, 21]
        self.default_kernel_sizes = 7
        self.pool_sizes = [0, 2, 3, 5]
        self.default_pool_sizes = 2
        self.neurons_units = [0, 32, 64, 128, 256]
        self.default_neurons_units = 128

    def setup_input(self):
        inputs = list()
        for i, t in enumerate(self.input_types):
            # Setup input for this branch
            input_shape = self.input_shapes[i]
            print('input_shape', input_shape)
            x = Input(shape=input_shape)
            if self.input_types[i] == 'categorical':
                n_words = self.k ** 4
                emb_size = (n_words * 2) + 1
                x = Embedding(n_words, emb_size, input_length=input_shape[0])(x)
            inputs.append(x)
        return inputs

    def build(self, hp):

        self.define_search_space()

        # Verify number branches based on the number of inputs
        num_branches = len(self.input_shapes)
        branches = list()
        inputs = self.setup_input()

        # Construct each branch
        for branch in range(num_branches):

            x = inputs[branch]

            # Verify the number of dimensions of this branch input - used to determine the type of used filters
            input_shape_length = len(self.input_shapes[branch])

            # Define number of stacked blocks
            # n_blocks = hp.Int('branch-{}_num_blocks'.format(branch), 1, 3)
            n_blocks = 1
            print('n_blocks', n_blocks)
            for _block in range(n_blocks):

                block_name = 'branch-{}_block-{}-{}'.format(branch, _block, n_blocks)
                # Choose number of filters
                name = block_name + '_n_filters'
                n_filters = hp.Choice(name, values=self.num_filters, default=self.default_num_filters)
                # Choose the size of kernels
                name = block_name + '_k_size'
                k_size_x = hp.Choice(name + '_x', values=self.kernel_sizes, default=self.default_kernel_sizes)
                k_size_y = hp.Choice(name + '_y', values=self.kernel_sizes, default=self.default_kernel_sizes)
                k_size_z = hp.Choice(name + '_z', values=self.kernel_sizes, default=self.default_kernel_sizes)
                print('k_sizes - block {}:\n\t{}\n\t{}\n\t{}'.format(_block, k_size_x, k_size_y, k_size_z))
                # Choose the size of pooling filters
                name = block_name + '_p_size'
                p_size = hp.Choice(name, values=self.pool_sizes, default=self.default_pool_sizes)

                # Setup the blocks

                # 1D - One dimensional filters
                if input_shape_length == 2:
                    # X
                    b = Conv1D(filters=n_filters, kernel_size=k_size_x, padding='same')(x)
                    b = BatchNormalization()(b)
                    b = Activation('relu')(b)
                    b = MaxPooling1D(pool_size=p_size, padding='same', strides=1)(b) if p_size > 0 else b
                    # Y
                    b = Conv1D(filters=n_filters, kernel_size=k_size_y, padding='same')(b)
                    b = BatchNormalization()(b)
                    b = Activation('relu')(b)
                    b = MaxPooling1D(pool_size=p_size, padding='same', strides=1)(b) if p_size > 0 else b
                    # Z
                    b = Conv1D(filters=n_filters, kernel_size=k_size_z, padding='same')(b)
                    b = BatchNormalization()(b)
                    # Shortcut
                    shortcut = Conv1D(filters=n_filters, kernel_size=1, padding='same')(x)
                    shortcut = BatchNormalization()(shortcut)
                    # Join
                    output_block = Add()([shortcut, b])
                    x = Activation('relu')(output_block)

                    # Insert GAP layer after the final block
                    if (_block + 1) == n_blocks:
                        x = GlobalAveragePooling1D()(x)

                # 2D - Two dimensional filters
                elif input_shape_length == 3:
                    height = 1 if _block != 0 else self.input_shapes[branch][2]
                    # X
                    b = Conv2D(filters=n_filters, kernel_size=(k_size_x, height), padding='same')(x)
                    b = BatchNormalization()(b)
                    b = Activation('relu')(b)
                    b = MaxPooling2D(pool_size=(p_size, 1), padding='same', strides=(1,1))(b) if p_size > 0 else b
                    # Y
                    b = Conv2D(filters=n_filters, kernel_size=(k_size_y, 1), padding='same')(b)
                    b = BatchNormalization()(b)
                    b = Activation('relu')(b)
                    b = MaxPooling2D(pool_size=(p_size, 1), padding='same', strides=(1,1))(b) if p_size > 0 else b
                    # Z
                    b = Conv2D(filters=n_filters, kernel_size=(k_size_z, 1), padding='same')(b)
                    b = BatchNormalization()(b)
                    # Shortcut
                    shortcut = Conv2D(filters=n_filters, kernel_size=(1, height), padding='same')(x)
                    shortcut = BatchNormalization()(shortcut)
                    # Join
                    output_block = Add()([shortcut, b])
                    x = Activation('relu')(output_block)

                    # Insert GAP layer after the final block
                    if (_block + 1) == n_blocks:
                        x = GlobalAveragePooling2D()(x)
                else:
                    print('ERROR: Invalid dimension ({}).'.format(input_shape_length))


            # Store this branch on the list to further concatenation
            branches.append(x)

        # Concatenate all branches
        if num_branches > 1:
            x = Concatenate()(branches)

        # Apply Dropout regularizer
        drop = hp.Float('dropout', min_value=.0, max_value=.6, step=.1, default=.2)
        x = Dropout(rate=drop)(x)

        # Define the fully connected layer
        n_neurons = hp.Choice('neurons_units', values=self.neurons_units, default=self.default_neurons_units)
        x = Dense(units=n_neurons, activation='relu', name='fc')(x) if n_neurons > 0 else x

        # Define the output layer
        x = Dense(units=self.num_classes, activation='sigmoid', name='outputs')(x)

        model = models.Model(inputs=inputs, outputs=x)

        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        model.compile(optimizer=Adam(lr=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy']
                      )

        return model

def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
        # Arguments
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            kernel_size: default 3, kernel size of the bottleneck layer.
            stride: default 1, stride of the first layer.
            conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
        # Returns
            Output tensor for the residual block.
        """
    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x
