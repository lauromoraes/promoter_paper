from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


class AugmentedGeneratorMultipleInputs(Sequence):
    """Wrapper of Multiple ImageDataGenerator"""

    def __init__(self, X, Y, bs):
        # Keras generator
        self.generator = ImageDataGenerator(width_shift_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

        # Real time multiple input data augmentation
        self.number_of_inputs = len(X)
        self.generators = [self.generator.flow(x, Y, batch_size=bs) for x in X]

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.generators[0].__len__()

    def __getitem__(self, index):
        """Getting items from the multiple generators and packing them"""
        X_batch = []
        for gen in self.generators:
            batch, Y_batch = gen.__getitem__(index)
            X_batch.append(batch)

        return X_batch, Y_batch
