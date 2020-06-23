#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Lauro Ângelo Gonçalves de Moraes
@contact: lauromoraes@ufop.edu.br
@created: 16/06/2020
"""
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

class BaseBlock(object):
    def __init__(self, name, block_input, block_output):
        self._name = name
        self._block_input = block_input
        self._block_output = block_output

class LeNetBlock(BaseBlock):
    def __init__(self):
        pass
    def set_layers(self):
        x = self._block_input
        conv = Conv2D()




def main():
    print("Hello World!")


if __name__ == "__main__":
    main()