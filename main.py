import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from keras import backend as K
from tensorflow import random as tf_random

from sklearn.model_selection import StratifiedShuffleSplit

from metrics import margin_loss
import mlflow

from promoter_data import PromoterData

# Set seeds
np.random.seed(1337)
tf_random.set_seed(3)


def train_test_experiment():
    kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
    kf.get_n_splits(X, y)

    print('\n', ' =' * 30)
    print('\tTRAIN / TEST PHASE')
    for i, partition in enumerate(kf.split(X[0], y)):
        print('>>> Testing PARTITION {}'.format(i))
        train_index, test_index = partition
        (x_train, y_train), (x_test, y_test) = data.load_partition(train_index, test_index)
        print('Number of samples for:\n\tTrain:\t{}\n\tTest:\t{}'.format(x_train[0].shape[0], x_test[0].shape[0]))

def hypermodel_experiment(X, y):
    from kerastuner.tuners import (RandomSearch, BayesianOptimization, Hyperband)
    from hypermodel import HotCNNHyperModel
    SEED = 17
    MAX_TRIALS = 40
    EXECUTION_PER_TRIAL = 10
    HYPERBAND_MAX_EPOCHS = 20

    NUM_CLASSES = 1  # One sigmoid neuron for binary classification
    INPUT_SHAPE = X[0].shape[1:]
    print('INPUT_SHAPE', INPUT_SHAPE)
    N_EPOCH_SEARCH = 40

    np.random.seed(SEED)
    hypermodel = HotCNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_accuracy',
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='hyperband',
        project_name='hot_cnn_promoter_01'
    )

    tuner.search_space_summary()

    kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
    kf.get_n_splits(X, y)
    print('\tHYPERMODEL SPLITS')
    for i, partition in enumerate(kf.split(X[0], y)):
        print('>>> Testing PARTITION {}'.format(i))
        train_index, test_index = partition
        (x_train, y_train), (x_test, y_test) = data.load_partition(train_index, test_index)
        print('Number of samples for:\n\tTrain:\t{}\n\tTest:\t{}'.format(x_train[0].shape[0], x_test[0].shape[0]))

        tuner.search(x_train[0], y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)

        # # Show a summary of the search
        # tuner.results_summary()
        # # Retrieve the best model.
        # best_model = tuner.get_best_models(num_models=1)[0]
        # # Evaluate the best model.
        # loss, accuracy = best_model.evaluate(x_test, y_test)
        # print(' =' * 30)
        # print('\tBest Model')
        # print('\tLoss: {}'.format(loss))
        # print('\tAcc: {}'.format(accuracy))
        # print(' =' * 30)

        # Get the best hyperparameters from the search
        params = tuner.get_best_hyperparameters()[0]
        # Build the model using the best hyperparameters
        model = tuner.hypermodel.build(params)
        # Train the best fitting model
        model.fit(X[0], y, epochs=20)
        # Check the accuracy plots
        hyperband_accuracy_df = pd.DataFrame(model.history.history)
        hyperband_accuracy_df[['loss', 'accuracy']].plot()
        plt.title('Loss & Accuracy Per EPOCH')
        plt.xlabel('EPOCH')
        plt.ylabel('Accruacy')
        plt.show()




if __name__ == "__main__":
    # Define settings
    seed = 17
    organism = 'Bacillus'
    k = 3
    tss_pos = 59
    downstream = 20
    upstream = 20

    # Set a data object
    data = PromoterData('./fasta')

    # Set dataset
    data.set_organism_sequences(organism, slice_seqs=True, tss_position=tss_pos, downstream=downstream,
                                upstream=upstream)

    # Set tokens
    data.set_tokens()

    # Get labels
    y = data.get_y()
    X = data.encode_dataset(encoder_types=1)

    print(' =' * 30)
    print('Number of inputs: {}'.format(len(X)))
    for i, x in enumerate(X):
        print('\tShape of input {}:\t{}'.format(i, x.shape))

    # train_test_experiment()
    hypermodel_experiment(X, y)
