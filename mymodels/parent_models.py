#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Lauro Ângelo Gonçalves de Moraes
@contact: lauromoraes@ufop.edu.br
@created: 20/06/2020
"""
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv2D,
    Conv1D,
    MaxPooling1D,
    MaxPooling2D,
    AveragePooling1D,
    AveragePooling2D,
    Flatten,
    Dense,
)

from kerastuner import HyperModel


def conv_pool_block(input_tensor, n_filters=100, k_size=15, pad='same', p_size=2, p_stride=2, activ='relu'):
    x = input_tensor
    input_dim = tf.keras.backend.shape(x).shape[0]
    block1 = Conv2D(
        filters=n_filters,
        kernel_size=(k_size, input_dim),
        padding=pad,
        activation=activ)(x)
    block1 = MaxPooling2D(
        pool_size=(p_size, 1),
        strides=(p_stride, 1))(block1)
    output_tensor = block1
    return output_tensor


class BaseModel(object):
    def __init__(self, data_list, num_classes):
        self.num_classes = num_classes
        self.input_shapes = list()
        self.input_types = list()
        for d in data_list:
            self.input_shapes.append(d.shape()[1:])
            self.input_types.append(d.get_encode())
        self.num_branches = len(data_list)
        self.inputs = self.setup_input()
        self.inputs_tensors = list()
        self.outputs_tensors = list()

    def setup_input(self):
        inputs = list()
        for i, t in enumerate(self.input_types):
            # Setup input for this branch
            input_shape = self.input_shapes[i]
            # print('input_shape', input_shape)
            x = Input(shape=input_shape, name='Input_{}'.format(i))
            if self.input_types[i] == 'categorical':
                n_words = self.k ** 4
                emb_size = (n_words * 2) + 1
                x = Embedding(n_words, emb_size, input_length=input_shape[0])(x)
            inputs.append(x)
        self.inputs_tensors = inputs
        return inputs

    def build(self):
        raise NotImplementedError()


class BaseHyperModel(BaseModel, HyperModel):
    def __init__(self, data_list, num_classes):
        super(HyperModel, self).__init__()
        super(BaseModel, self).__init__(data_list, num_classes)

    def define_search_space(self):
        raise NotImplementedError()

    def build(self, hp):
        raise NotImplementedError()


class HotCNN(BaseModel):
    def __init__(self, data_list, num_classes):
        super(HotCNN, self).__init__(data_list, num_classes)

    def build(self):
        input_tensor = self.setup_input()[0]
        block1 = conv_pool_block(input_tensor, n_filters=100, k_size=15, pad='same', p_size=2, p_stride=2, activ='relu')
        block2 = conv_pool_block(block1, n_filters=250, k_size=17, pad='same', p_size=2, p_stride=2, activ='relu')

        # Flat tensors
        flat = Flatten()(block2)

        # Fully connected layers
        dense1 = Dense(128, activation='relu', name='fully_con')(flat)

        # Classification layer
        activ = 'sigmoid' if self.num_classes == 1 else 'softmax'
        output = Dense(self.num_classes, activation=activ, name='classification_layer')(dense1)
        self.outputs_tensors.append(output)

        # Create model object
        model = models.Model(inputs=self.inputs_tensors, outputs=self.outputs_tensors, name='Baseline_HotCNN_Bacillus')

        return model


def main():
    pass


if __name__ == "__main__":
    main()
