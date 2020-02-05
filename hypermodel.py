import numpy as np
from tensorflow import keras
from tensorflow import random as tf_random
from kerastuner import HyperModel
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)
from tensorflow.keras.optimizers import (
    Adam,
    Nadam,
)

class HotCNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        emb_size = 4
        k_sizes = [(emb_size, 3), (emb_size, 7), (emb_size, 15), (emb_size, 21)]
        p_sizes = [(1, 2), (1, 3), (1, 5)]

        model = keras.Sequential()

        model.add(Conv2D(
            filters=hp.Choice('conv01_num_filters', values=[32, 64, 128], default=128),
            activation='relu',
            kernel_size=hp.Choice('conv01_ksize', values=k_sizes, default=(4, 15))
        ))
        model.add(MaxPooling2D(
            pool_size=hp.Choice('conv01_pool', values=p_sizes, default=(1, 2))
        ))

        model.add(Dropout(rate=hp.Float('drop', min_value=.0, max_value=.6, step=.1, default=.2)))
        model.add(Flatten())
        model.add(Dense(units=self.num_classes, activation='sigmoid'))

        model.compile(optimizer=Adam(lr=hp.Float(
                'learning_rate',
                min_value=1e-4,
                max_value=1e-2,
                sampling='LOG',
                default=1e-3
            )),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
