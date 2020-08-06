import os
import numpy as np

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import (Adam, Nadam, )
from tensorflow.keras.utils import plot_model


from sklearn.model_selection import StratifiedShuffleSplit

from metrics import margin_loss
from mlflow import keras as mlflow_keras

from promoter_data import PromoterData, DataChunk
from my_generator import AugmentedGeneratorMultipleInputs

import mymodels.parent_models as mymodels
from ml_logs import MlLogs
from promoter_utils import (
    # train_test,
    # get_model,
    # get_test_stats,
    _plot,
    _plot_history,
)

# Set samples
np.random.seed(1337)
if int(str(tf.__version__).split('.')[0]) >= 2:
    from tensorflow import random as tf_random

    tf_random.set_seed(3)
else:
    from tensorflow import set_random_seed

    set_random_seed(3)

def get_model(model_type='HotCNNHyperModel'):
    if model_type.endswith('HyperModel'):
        import mymodels.hypermodels as module
        print('Getting dynamic class: {}'.format(model_type))
        dynamic_model = getattr(module, model_type)
    else:
        import mymodels.cnn as module
        print('Getting defined class: {}'.format(model_type))
        dynamic_model = getattr(module, model_type)
    return dynamic_model


def hypermodel_tunning_parameters(data, y, args):
    from kerastuner.tuners import (RandomSearch, BayesianOptimization, Hyperband)
    SEED = args.samples[0]
    EXECUTION_PER_TRIAL = args.hypermodel['execution_per_trial']
    HYPERBAND_MAX_EPOCHS = args.hypermodel['hyperband_max_epochs']
    N_EPOCH_SEARCH = 30
    MODEL_TYPE = args.model_type
    PROJECT_NAME = args.experiment_name

    NUM_CLASSES = 1  # One sigmoid neuron for binary classification
    DATA = data.data
    X = [x for x in data.X]

    np.random.seed(SEED)
    # Get dynamic model class
    dynamic_model = get_model(model_type=MODEL_TYPE)
    # Instantiate object from dynamic model class
    hypermodel = dynamic_model(DATA, NUM_CLASSES)

    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_loss',
        loss='binary_crossentropy',
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='hyperband',
        project_name=PROJECT_NAME,
        metrics=['accuracy']
    )

    tuner.search_space_summary()

    kf = StratifiedShuffleSplit(n_splits=1, random_state=SEED, test_size=0.3)
    kf.get_n_splits(X, y)
    print('\t>{} HYPERMODEL START {}'.format('>' * 30, '<' * 30))
    for i, partition in enumerate(kf.split(X[0], y)):
        print('>>> Testing PARTITION {}'.format(i))
        train_index, test_index = partition
        (x_train, y_train), (x_test, y_test) = data.load_partition(train_index, test_index)
        print('Number of samples for:\n\tTrain:\t{}\n\tTest:\t{}'.format(x_train[0].shape[0], x_test[0].shape[0]))

        tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1, verbose=0)

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
    print('\t>{} HYPERMODEL END {}'.format('>' * 30, '<' * 30))
    return tuner


def get_test_stats(model, X, y):
    from ml_statistics import BaseStatistics
    y_pred = model.predict(x=X, batch_size=8)
    stats = BaseStatistics(y, y_pred)
    return stats, y_pred


# def train_test(model, data, y, args):
#     import os
#     from hypermodel_utils import load_partition, get_callbacks, get_results_table, allocate_stats, mlflow_logs
#     import tensorflow.keras.callbacks as C
#
#     MODEL_TYPE = args.model_type
#     TIMESTAMP = args.timestamp
#     samples = args.samples
#     X = [x for x in data.X]
#     FIT_MAX_EPOCHS = args.epochs
#
#     # Get the best hyperparameters from the search
#     best_tuner_mcc = -1000.0
#     best_tuner_idx = 0
#
#     results = list()
#
#     # Build the model using the best hyperparameters
#     model.summary()
#
#     kf = StratifiedShuffleSplit(n_splits=args.cv, random_state=args.samples[0])
#     kf.get_n_splits(X[0], y)
#     cv_partition = 0
#     # Append header
#     results.append(get_results_table())
#
#     best_weights = [None for _ in range(args.cv)]
#     best_stats = [None for _ in range(args.cv)]
#
#     # Cross validation loop
#     for cv_train_index, cv_test_index in kf.split(X[0], y):
#         cv_partition += 1
#         (x_train, y_train), (x_test, y_test) = data.load_partition(cv_train_index, cv_test_index)
#         callbacks = get_callbacks(args, cv_partition)
#
#         best_cv_mcc = -10000.0
#         best_cv_stats = None
#
#         # Validation samples loop
#         for s, seed in enumerate(samples):
#             print('MODEL {} - CV {} - {} Train on SEED {}'.format(MODEL_TYPE, cv_partition, s, seed))
#             weight_file_name = '{}-{}-partition_{}-seed_{}'.format(MODEL_TYPE, TIMESTAMP, cv_partition, s)
#             weight_file_name += '-epoch_{epoch:02d}.hdf5'
#             weight_path = os.path.join(args.weights_dir, weight_file_name)
#
#             callbacks[0] = C.ModelCheckpoint(weight_path, save_best_only=True, save_weights_only=True, verbose=1)
#
#             kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
#             kf.get_n_splits(x_train, y_train)
#             for t_index, v_index in kf.split(x_train[0], y_train):
#                 (xx_train, yy_train), val_data = data.load_partition(t_index, v_index)
#                 model.fit(x=xx_train, y=yy_train, batch_size=args.batch_size, epochs=FIT_MAX_EPOCHS,
#                           validation_data=val_data, callbacks=callbacks, verbose=0)
#             K.clear_session()
#
#             # # Test model fitted using seed validation set
#             # stats, y_pred = get_test_stats(model, x_test, y_test)
#             # print(stats.Mcc)
#             # print(stats.F1)
#
#         cv_idx = cv_partition - 1
#         best_stats[cv_idx], best_weights[cv_idx] = do_partition_test(
#             model, cv_partition, args, test_data=(x_test, y_test))
#         results[idx] = allocate_stats(results[idx], best_stats[cv_idx], cv_partition)
#
#     # Persist weights files
#     cv_mean_mcc = np.mean(results[idx]['mcc'])
#     tuner_folder_name = '{}-{}-tuner_{}-mcc_{:.4f}'.format(MODEL_TYPE, TIMESTAMP, idx, cv_mean_mcc)
#     folder_name = os.path.join(args.best_weights_dir, tuner_folder_name)
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     for cv in range(args.cv):
#         weight_name = '{}-{}-partition_{}-mcc_{:.4f}.hdf5'.format(MODEL_TYPE, TIMESTAMP, cv, results[idx]['mcc'][cv])
#         weight_path = os.path.join(folder_name, weight_name)
#         model.set_weights(best_weights[cv])
#         model.save_weights(weight_path)
#
#     print('EVAL MODEL {} - {}'.format(MODEL_TYPE, np.mean(results[idx]['mcc'])))
#     model = tuner.hypermodel.build(t.hyperparameters)
#     mlflow_logs(args, t.hyperparameters.values, results[idx], model, idx)
#
#     # # Train the best fitting model
#     # model.fit(X, y, epochs=FIT_MAX_EPOCHS)
#     # # Check the accuracy plots
#     # hyperband_accuracy_df = pd.DataFrame(model.history.history)
#     # hyperband_accuracy_df[['loss', 'accuracy']].plot()
#     # plt.title('Loss & Accuracy Per EPOCH')
#     # plt.xlabel('EPOCH')
#     # plt.ylabel('Accruacy')
#     # plt.show()


def test_tuners(tuner, data, y, args):
    import os
    from hypermodel_utils import load_partition, get_callbacks, get_results_table, allocate_stats, mlflow_logs
    import tensorflow.keras.callbacks as C

    MODEL_TYPE = args.model_type
    TIMESTAMP = args.timestamp
    samples = args.samples
    X = [x for x in data.X]
    FIT_MAX_EPOCHS = args.epochs

    # Get the best hyperparameters from the search
    best_tuner_mcc = -1000.0
    best_tuner_idx = 0

    tuners = tuner.oracle.get_best_trials(num_trials=1)
    results = list()

    for idx, t in enumerate(tuners):
        # loss = t.metrics.get_best_value('loss')
        # acc = t.metrics.get_best_value('accuracy')
        # print('- ' * 30)
        # print('{}\t{}'.format(loss, acc))
        # print(t.hyperparameters.values)

        # Build the model using the best hyperparameters
        # params = tuner.get_best_hyperparameters()[idx]
        params = t.hyperparameters
        print('PARAMS:', params)
        model = tuner.hypermodel.build(params)
        model.summary()

        kf = StratifiedShuffleSplit(n_splits=args.cv, random_state=args.samples[0])
        kf.get_n_splits(X[0], y)
        cv_partition = 0
        results.append(get_results_table())

        best_weights = [None for _ in range(args.cv)]
        best_stats = [None for _ in range(args.cv)]

        # Cross validation loop
        for cv_train_index, cv_test_index in kf.split(X[0], y):
            cv_partition += 1
            (x_train, y_train), (x_test, y_test) = data.load_partition(cv_train_index, cv_test_index)
            callbacks = get_callbacks(args, cv_partition)

            best_cv_mcc = -10000.0
            best_cv_stats = None

            # Validation samples loop
            for s, seed in enumerate(samples):
                print('TUNER {} - CV {} - {} Train on SEED {}'.format(idx, cv_partition, s, seed))
                weight_file_name = '{}-{}-partition_{}-seed_{}'.format(MODEL_TYPE, TIMESTAMP, cv_partition, s)
                weight_file_name += '-epoch_{epoch:02d}.hdf5'
                weight_path = os.path.join(args.weights_dir, weight_file_name)

                callbacks[0] = C.ModelCheckpoint(weight_path, save_best_only=True, save_weights_only=True, verbose=1)

                kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.05)
                kf.get_n_splits(x_train, y_train)
                for t_index, v_index in kf.split(x_train[0], y_train):
                    (xx_train, yy_train), val_data = data.load_partition(t_index, v_index)
                    model.fit(x=xx_train, y=yy_train,
                              batch_size=args.batch_size,
                              epochs=FIT_MAX_EPOCHS,
                              validation_data=val_data,
                              callbacks=callbacks,
                              class_weight={0: .2, 1: .8},
                              verbose=0
                              )

                    # # Data Augmentation =================================================
                    # train_datagen = AugmentedGeneratorMultipleInputs(xx_train, yy_train, args.batch_size)
                    # _steps_per_epoch = int(len(yy_train) / args.batch_size)
                    # print("_steps_per_epoch", _steps_per_epoch)
                    # # model.fit_generator(train_datagen.flow(xx_train, yy_train, batch_size=args.batch_size),
                    # model.fit_generator(train_datagen,
                    #                     validation_data=val_data,
                    #                     steps_per_epoch=_steps_per_epoch,
                    #                     epochs=FIT_MAX_EPOCHS,
                    #                     callbacks=callbacks,
                    #                     verbose=1)
                    # # ==================================================================

                K.clear_session()

                # # Test model fitted using seed validation set
                # stats, y_pred = get_test_stats(model, x_test, y_test)
                # print(stats.Mcc)
                # print(stats.F1)

            cv_idx = cv_partition - 1
            best_stats[cv_idx], best_weights[cv_idx] = do_partition_test(
                model, cv_partition, args, test_data=(x_test, y_test))
            results[idx] = allocate_stats(results[idx], best_stats[cv_idx], cv_partition)

        # Persist weights files
        cv_mean_mcc = np.mean(results[idx]['mcc'])
        tuner_folder_name = '{}-{}-tuner_{}-mcc_{:.4f}'.format(MODEL_TYPE, TIMESTAMP, idx, cv_mean_mcc)
        folder_name = os.path.join(args.best_weights_dir, tuner_folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for cv in range(args.cv):
            weight_name = '{}-{}-partition_{}-mcc_{:.4f}.hdf5'.format(MODEL_TYPE, TIMESTAMP, cv,
                                                                      results[idx]['mcc'][cv])
            weight_path = os.path.join(folder_name, weight_name)
            model.set_weights(best_weights[cv])
            model.save_weights(weight_path)

    for idx, t in enumerate(tuners):
        print('EVAL TUNER {} - {}'.format(idx, np.mean(results[idx]['mcc'])))
        model = tuner.hypermodel.build(t.hyperparameters)
        mlflow_logs(args, t.hyperparameters.values, results[idx], model, idx)

        # # Train the best fitting model
        # model.fit(X, y, epochs=FIT_MAX_EPOCHS)
        # # Check the accuracy plots
        # hyperband_accuracy_df = pd.DataFrame(model.history.history)
        # hyperband_accuracy_df[['loss', 'accuracy']].plot()
        # plt.title('Loss & Accuracy Per EPOCH')
        # plt.xlabel('EPOCH')
        # plt.ylabel('Accruacy')
        # plt.show()


def train_test(model, data, y, args):
    import os
    import tensorflow.keras.backend as bk
    import tensorflow as tf
    from hypermodel_utils import load_partition, get_callbacks, get_results_table, allocate_stats, mlflow_logs

    MODEL_TYPE = args.model_type
    TIMESTAMP = args.timestamp
    samples = args.samples
    X = [x for x in data.X]
    FIT_MAX_EPOCHS = args.epochs

    results = list()
    model.summary()



    kf = StratifiedShuffleSplit(n_splits=args.cv, random_state=args.samples[0])
    kf.get_n_splits(X[0], y)
    results.append(get_results_table())

    best_weights = [None for _ in range(args.cv)]
    best_stats = [None for _ in range(args.cv)]

    metrics_stats = {m: {'p{:02d}'.format(x): [] for x in range(args.cv)} for m in ('Prec', 'Sn', 'Sp', 'Acc', 'F1', 'Mcc')}
    weights_sufix = '.hdf5'

    # Cross validation loop
    for idx, (cv_train_index, cv_test_index) in enumerate(kf.split(X[0], y)):
        (x_train, y_train), (x_test, y_test) = data.load_partition(cv_train_index, cv_test_index)
        callbacks = get_callbacks(args, idx)
        best_cv_mcc = -10000.0

        # === Perform Training Phase ===
        # Iterate over samples to create different samples on same partition based on different train-val splits
        for s, seed in enumerate(samples):
            print('CV PARTITION {} - CV SEED {} | Train on SEED {}'.format(idx, s, seed))
            weight_file_prefix = '{}-{}-partition_{:02d}'.format(MODEL_TYPE, TIMESTAMP, idx)
            weight_file_name = weight_file_prefix + '-sample_{:02d}'.format(s)
            weight_file_name += '-epoch_{epoch:02d}.hdf5'
            weight_path = os.path.join(args.weights_dir, weight_file_name)

            callbacks[0] = tf.keras.callbacks.ModelCheckpoint(weight_path, save_best_only=True, save_weights_only=True, verbose=1)

            kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.05)
            kf.get_n_splits(x_train, y_train)
            histories = []
            for t_index, v_index in kf.split(x_train[0], y_train):
                (xx_train, yy_train), val_data = data.load_partition(t_index, v_index)
                history = model.fit(
                    x=xx_train,
                    y=yy_train,
                    batch_size=args.batch_size,
                    epochs=FIT_MAX_EPOCHS,
                    validation_data=val_data,
                    callbacks=callbacks,
                    class_weight={0: .2, 1: .8},
                    verbose=0
                )

                # PLOT HISTORY
                _plot_history(args, 'accuracy', history, idx, s)
                _plot_history(args, 'loss', history, idx, s)


                # # Data Augmentation =================================================
                # train_datagen = AugmentedGeneratorMultipleInputs(xx_train, yy_train, args.batch_size)
                # _steps_per_epoch = int(len(yy_train) / args.batch_size)
                # print("_steps_per_epoch", _steps_per_epoch)
                # # model.fit_generator(train_datagen.flow(xx_train, yy_train, batch_size=args.batch_size),
                # model.fit_generator(train_datagen,
                #                     validation_data=val_data,
                #                     steps_per_epoch=_steps_per_epoch,
                #                     epochs=FIT_MAX_EPOCHS,
                #                     callbacks=callbacks,
                #                     verbose=1)
                # # ==================================================================

            # # Test model fitted using seed validation set
            # stats, y_pred = get_test_stats(model, x_test, y_test)
            # print(stats.Mcc)
            # print(stats.F1)

            # === Perform Test Phase ===
            sample_weights = [x for x in os.listdir(args.weights_dir + os.path.sep) if
                             x.startswith(weight_file_prefix) and x.endswith(weights_sufix)]

            # Iterate over all weights saved on checkpoints for this sample of this partition
            for i, f in enumerate(sample_weights):
                best_sample_mcc = -10000.0
                best_sample_stats = None
                bk.clear_session()
                name = os.path.join(args.weights_dir, f)
                # print(name)
                model.load_weights(name)
                stats, y_pred = get_test_stats(model=model, X=x_test, y=y_test)

                # Select best sample weights
                if best_sample_mcc < stats.Mcc:
                    best_sample_mcc = stats.Mcc
                    best_sample_stats = stats

                # Select best weigths for this partition
                if best_cv_mcc < stats.Mcc:
                    best_cv_mcc = stats.Mcc
                    selected_weight = f
                    best_stats[idx] = stats
                    # print(stats)

            for metric in best_sample_stats.get_stats_types():
                sample_stats = best_sample_stats.to_dict()
                metrics_stats[metric]['p{:02}'.format(idx)].append(sample_stats[metric])

            # Delete temporary weights
            for i, f in enumerate(sample_weights):
                if f != selected_weight:
                    # print('Deleting weight: {}'.format(f))
                    path = args.weights_dir + '/' + f
                    os.remove(path)

        bk.clear_session()

        print(best_sample_stats)
        # Persist best weights of this partition on logs
        args.logs.set_metrics(**metrics_stats) # Log metrics
        args.logs.set_artifacts()  # Log artifacts
        model.load_weights(os.path.join(args.weights_dir, selected_weight))
        persist_model_path = os.path.join(args.best_weights_dir, 'model_{}_{}_p{:02d}'.format(MODEL_TYPE, TIMESTAMP, idx))
        # mlflow_keras.save_model(model, persist_model_path)
        # mlflow_keras.log_model(model, args.logs.get_model_path())
        mlflow_keras.save_model(model, args.logs.get_model_path(idx))
        mlflow_keras.log_model(model, 'models')
        # print('Deleting weight: {}'.format(selected_weight))
        path = args.weights_dir + '/' + selected_weight
        os.remove(path)
        

        # best_stats[idx], best_weights[idx] = do_partition_test(model, idx, args, test_data=(x_test, y_test))
        # results[idx] = allocate_stats(results[idx], best_stats[idx], idx)

    # Persist weights files
    # cv_mean_mcc = np.mean(results[idx]['mcc'])
    # tuner_folder_name = '{}-{}-tuner_{}-mcc_{:.4f}'.format(MODEL_TYPE, TIMESTAMP, idx, cv_mean_mcc)
    # folder_name = os.path.join(args.best_weights_dir, tuner_folder_name)
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    # for cv in range(args.cv):
    #     weight_name = '{}-{}-partition_{}-mcc_{:.4f}.hdf5'.format(MODEL_TYPE, TIMESTAMP, cv,
    #                                                               results[idx]['mcc'][cv])
    #     weight_path = os.path.join(folder_name, weight_name)
    #     model.set_weights(best_weights[cv])
    #     model.save_weights(weight_path)

    # for idx, t in enumerate(tuners):
    #     print('EVAL TUNER {} - {}'.format(idx, np.mean(results[idx]['mcc'])))
    #     model = tuner.hypermodel.build(t.hyperparameters)
    #     mlflow_logs(args, t.hyperparameters.values, results[idx], model, idx)

    # # Train the best fitting model
    # model.fit(X, y, epochs=FIT_MAX_EPOCHS)
    # # Check the accuracy plots
    # hyperband_accuracy_df = pd.DataFrame(model.history.history)
    # hyperband_accuracy_df[['loss', 'accuracy']].plot()
    # plt.title('Loss & Accuracy Per EPOCH')
    # plt.xlabel('EPOCH')
    # plt.ylabel('Accruacy')
    # plt.show()


def do_partition_test(model, cv_partition, args, test_data):
    import os
    import tensorflow.keras.backend as bk
    MODEL_TYPE = args.model_type
    TIMESTAMP = args.timestamp
    f_prefix = '{}-{}-partition_{}'.format(MODEL_TYPE, TIMESTAMP, cv_partition)
    f_sufix = '.hdf5'
    model_weights = [x for x in os.listdir(args.weights_dir + os.path.sep) if
                     x.startswith(f_prefix) and x.endswith(f_sufix)]
    print('Testing weigths:', model_weights)
    best_mcc = -10000.0
    selected_weight = None
    selected_stats = None

    for i, f in enumerate(model_weights):
        bk.clear_session()
        name = os.path.join(args.weights_dir, f)
        print(name)
        model.load_weights(name)
        stats, y_pred = get_test_stats(model=model, X=test_data[0], y=test_data[1])

        # Get current best weigth
        if best_mcc < stats.Mcc:
            best_mcc = stats.Mcc
            selected_weight = f
            selected_stats = stats
            print('Selected BEST')
            print(stats)

    # Persist best weights
    print(os.path.join(args.weights_dir, selected_weight))
    model.load_weights(os.path.join(args.weights_dir, selected_weight))
    weights = model.get_weights()
    # f_weight = selected_weight.split('.')[0] + '-mcc_{:.4f}.hdf5'.format(best_mcc)
    # model.save_weights(os.path.join(args.best_weights_dir, f_weight))

    # Delete temporary weights
    for i, f in enumerate(model_weights):
        print('Deleting weight: {}'.format(f))
        path = args.weights_dir + '/' + f
        os.remove(path)

    return (selected_stats, weights)


if __name__ == "__main__":
    from hypermodel_utils import load_args, load_data_chunks
    import sys
    from mlflow import keras as ml_keras

    # ml_keras.autolog()
    print(sys.argv)

    ARGS = load_args(' '.join(sys.argv[1:]))

    loss_func = 'binary_crossentropy'

    name = '{}-{}'.format(ARGS.model_type, ARGS.timestamp)
    logs = MlLogs(name=name, verbose=True)
    logs.set_params(
        organism=ARGS.organism,
        model_type=ARGS.model_type,
        n_partitions=ARGS.cv,
        n_samples=ARGS.samples,
        n_epochs=ARGS.epochs,
        batch_size=ARGS.batch_size,
        early_stop=ARGS.early_stop,
        initial_lr=ARGS.initial_lr,
        loss_function=loss_func,
    )
    ARGS.logs = logs
    ARGS.logs.set_tag('model_name', ARGS.model_type)
    ARGS.logs.set_tag('organism', ARGS.organism)

    # Define settings
    organism = ARGS.organism

    # Set a data object
    prom_data = PromoterData(ARGS.fasta_dir)
    # Set dataset
    prom_data.set_organism_sequences(organism)

    from data_augmentation import DataAug
    aug = DataAug()
    aug.switch_nucs(dataset=prom_data.positive_sequences, switch_ratio=.5)

    data_chunks = load_data_chunks(ARGS)

    # Get labels
    y = prom_data.get_y()
    data_chunks = prom_data.encode_dataset(_data=data_chunks)

    print(' =' * 30)
    print('Number of inputs: {}'.format(len(data_chunks)))
    for i, d in enumerate(data_chunks):
        print('\tShape of input {}:\t{}'.format(i, d.shape()))

    if ARGS.model_type.endswith('HyperModel'):
        print('HyperModel Optimization')

        # train_test_experiment()
        h_params = hypermodel_tunning_parameters(prom_data, y, ARGS)

        test_tuners(h_params, prom_data, y, ARGS)

    else:
        # Get dynamic model class
        dynamic_model = get_model(model_type=ARGS.model_type)
        # Instantiate object from dynamic model class
        # model = dynamic_model(data.data, 1)
        arc = mymodels.BaselineHotCNN(data_list=prom_data.data, num_classes=1)
        model = arc.build()
        model.compile(
            optimizer=Adam(lr=ARGS.initial_lr),
            loss=loss_func,
            metrics=['accuracy']
        )
        # ARGS.logs.set_tag('model_json', str(model.to_json()))
        # print(model.to_json())
        img_path = ARGS.logs.get_image_path()
        plt_model_path = '{}-{}.png'.format(ARGS.model_type, ARGS.timestamp)
        plt_model_path = os.path.join(img_path, plt_model_path)
        plot_model(model, plt_model_path, dpi=96)
        ARGS.logs.set_image(plt_model_path)

        num_params = model.count_params()
        ARGS.logs.set_params(n_train_params=num_params)
        # for key, val in model.get_config():
        #     print(key, val)

        # ==== TRAIN/TEST
        train_test(model, prom_data, y, ARGS)

        # ==== LOGS - send outputs
        run_name = '{}-{}'.format(ARGS.model_type, ARGS.timestamp)
        experiment_name = 'Prom_{:02d}-{:02}'.format(ARGS.cv, len(ARGS.samples))
        ARGS.logs.flush_session(experiment_name=experiment_name, run_name=run_name)
        ARGS.logs.clear_session()

        pass
        # x = 32792451