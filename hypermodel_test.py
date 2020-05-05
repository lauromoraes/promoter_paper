import numpy as np
from kerastuner.tuners import (
    Hyperband
)

from mymodels.hypermodels import HotCNNHyperModel

def hypermodel_exec(x_train, x_test, y_train, y_test):
    SEED = 17
    MAX_TRIALS = 40
    EXECUTION_PER_TRIAL = 3
    HYPERBAND_MAX_EPOCHS = 20

    NUM_CLASSES = 1 # One sigmoid neuron for binary classification
    INPUT_SHAPE = (32, 32, 3)  # Depends on embedding type and bp lenght of dataset
    N_EPOCH_SEARCH = 40

    np.random.seed(SEED)

    hypermodel = HotCNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    # tuner = RandomSearch(
    #     hypermodel,
    #     objective='val_loss',
    #     seed=SEED,
    #     max_trials=MAX_TRIALS,
    #     executions_per_trial=EXECUTION_PER_TRIAL,
    #     directory='random_search',
    #     project_name='hot_cnn_promoter_01'
    # )

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



    tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(x_test, y_test)


