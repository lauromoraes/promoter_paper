#!/usr/bin/env python
"""

Usage:

       ... ...

    
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

np.random.seed(1337)

from keras import layers, models, optimizers
from keras import backend as K
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import StratifiedShuffleSplit

from metrics import margin_loss

headers = ['partition', 'mcc', 'f1', 'sn', 'sp', 'acc', 'prec', 'tp', 'fp', 'tn', 'fn']
results = {x: [] for x in headers}
# results = {'partition': [], 'mcc': [], 'f1': [], 'sn': [], 'sp': [], 'acc': [], 'prec': [], 'tp': [], 'fp': [],
#            'tn': [], 'fn': []}


def create_classifier(classifier_name, input_shape, n_class, verbose=False):
    import importlib
    module = importlib.import_module("mymodels.cnn")
    if classifier_name == 'HOTCNN01':
        classifier = module.HOTCNN01(input_shape, n_class)
    # elif classifier_name == 'HOTCNN_BACILLUS_01':
    #     from mymodels import cnn
    #     classifier = cnn.HOTCNN_BACILLUS_01(input_shape, n_class)
    # elif classifier_name == 'HOT_RES_BACILLUS_01':
    #     from mymodels import cnn
    #     classifier = getattr(cnn, 'HOT_RES_BACILLUS_01')(input_shape, n_class)
    else:
        # print(f'"{classifier_name}" is not a valid classifier name. Using DEFAULT: "HOTCNN01"')
        classifier = getattr(module, classifier_name)(input_shape, n_class)
    if verbose:
        classifier.summary()
    return classifier


def get_calls():
    from keras import callbacks as C
    import math

    cycles = 50
    calls = list()
    calls.append(None)  # Position for Chackpoint
    calls.append(C.CSVLogger(args.weights_dir + '/log.csv'))
    calls.append(C.TensorBoard(log_dir=args.weights_dir + '/tensorboard-logs/{}'.format(actual_partition),
                               batch_size=args.batch_size, histogram_freq=args.debug))
    calls.append(C.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0))
    # calls.append(C.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=0))
    calls.append(C.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch)))
    # calls.append( C.LearningRateScheduler(schedule=lambda epoch: args.lr * math.cos(1+( (epoch-1 % (args.epochs/cycles)))/(args.epochs/cycles) ) ))	
    #    calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.)) )
    return calls


def train(model, data, args, actual_partition):
    from keras import callbacks as C
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    calls = get_calls()

    lossfuncs = ['binary_crossentropy', 'mse']
    if args.n_class == 1:
        loss_func = lossfuncs[0]
    else:
        loss_func = lossfuncs[1]
    # compile the model
    #    seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    seeds = [23, 29, 31]
    # seeds = [23]
    #    seeds = [23, 29]
    for s in range(len(seeds)):
        seed = seeds[s]
        print('{} Train on SEED {}'.format(s, seed))

        weight_file_name = TIMESTAMP + '-seed_{}-partition_{}-weights.h5'.format(s, actual_partition)
        weight_path = os.path.join(args.weights_dir, weight_file_name)
        calls[0] = C.ModelCheckpoint(weight_path, save_best_only=True, save_weights_only=True, verbose=1)

        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=loss_func,
                      # loss_weights=[1., args.lam_recon],
                      metrics=['accuracy']
                      )

        kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
        kf.get_n_splits(x_train, y_train)

        for t_index, v_index in kf.split(x_train, y_train):
            X_train, X_val = x_train[t_index], x_train[v_index]
            Y_train, Y_val = y_train[t_index], y_train[v_index]

            val_data = (X_val, Y_val)

            model.fit(x=X_train, y=Y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=val_data,
                      callbacks=calls, verbose=0)

    #            model.save_weights(args.weights_dir + '/trained_model.h5')
    #            print('Trained model saved to \'%s/trained_model.h5\'' % args.weights_dir)

    #    from utils import plot_log
    #    plot_log(args.weights_dir + '/log.csv', show=True)

    return model


def test(model, data):
    from ml_statistics import BaseStatistics
    x_test, y_test = data
    Y = np.zeros(y_test.shape)
    y_pred = model.predict(x=x_test, batch_size=8)
    stats = BaseStatistics(y_test, y_pred)
    return stats, y_pred


def load_dataset(organism):
    from ml_data import SequenceNucsData, SequenceNucHotvector, SequenceMotifHot

    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)

    k = 1
    samples = SequenceNucHotvector(npath, ppath)

    X, y = samples.getX(), samples.getY()
    #    X = X.reshape(-1, 38, 79, 1).astype('float32')
    X = X.astype('int32')
    #     ini = 59
    # #    ini = 199
    #     X = X[:, (ini-30):(ini+11)]
    y = y.astype('int32')
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    return X, y, X.shape[1:]


def load_partition(train_index, test_index, X, y):
    x_train = X[train_index, :]
    y_train = y[train_index]

    x_test = X[test_index, :]
    y_test = y[test_index]

    #    y_train = to_categorical(y_train.astype('float32'))
    #    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def get_best_weight(args, actual_partition):

    # Select weights
    file_prefix = TIMESTAMP
    file_sufix = '-partition_{}-weights.h5'.format(actual_partition)
    model_weights = [x for x in os.listdir(args.weights_dir + os.path.sep) if
                     x.startswith(file_prefix) and x.endswith(file_sufix)]
    print('Testing weigths', model_weights)
    best_mcc = -10000.0
    selected_weight = None
    selected_stats = None

    # Clear model
    K.clear_session()

    # Iterate over generated weights for this partition
    for i in range(len(model_weights)):

        weight_file = model_weights[i]

        # Create new model to receive this weights
        model = create_classifier(CLASSIFIER, INPUT_SHAPE, N_CLASS)
        model.load_weights(os.path.join(args.weights_dir, weight_file))

        # Get statistics for model loaded with current weights
        stats, y_pred = test(model=model, data=(x_test, y_test))
        print('MCC = {}'.format(stats.Mcc))

        # Get current best weigth
        if best_mcc < stats.Mcc:
            best_mcc = stats.Mcc
            selected_weight = weight_file
            selected_stats = stats
            print('Selected BEST')
            print(stats)

        # Clear model
        K.clear_session()

    # Persist best weights
    model = create_classifier(CLASSIFIER, INPUT_SHAPE, N_CLASS)
    model.load_weights(os.path.join(args.weights_dir, selected_weight))
    model.save_weights(os.path.join(args.weights_dir, TIMESTAMP + '-partition_{}-best_weights.h5'.format(actual_partition)))

    K.clear_session()

    # Delete others weights
    for i in range(len(model_weights)):
        weight_file = model_weights[i]
        print('Deleting weight: {}'.format(weight_file))
        path = args.weights_dir + '/' + weight_file
        try:
            os.remove(path)
        except:
            pass

    return (selected_stats, selected_weight)


def allocate_stats(stats):
    global results

    results['partition'].append(actual_partition)
    results['mcc'].append(stats.Mcc)
    results['f1'].append(stats.F1)
    results['sn'].append(stats.Sn)
    results['sp'].append(stats.Sp)
    results['acc'].append(stats.Acc)
    results['prec'].append(stats.Prec)
    results['tp'].append(stats.tp)
    results['fp'].append(stats.fp)
    results['tn'].append(stats.tn)
    results['fn'].append(stats.fn)


if __name__ == "__main__":

    from utils import get_args

    args = get_args()

    INPUT_SHAPE = None
    EXPERIMENT_INFO = dict()

    N_CLASS = args.n_class
    CV = args.cv
    CLASSIFIER = args.model
    PATIENCE = args.patience

    TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    EXPERIMENT_INFO['timestamp'] = TIMESTAMP
    EXPERIMENT_INFO['classifier'] = CLASSIFIER
    EXPERIMENT_INFO['cv'] = CV
    # prefix_name = 'conv1_org_{}-batch_{}-kernel_{}-pool_{}-cv_{}'.format(args.organism, args.batch_size,
    #                                                                      args.kernel1_size, args.pool1_size, CV)

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # load data
    X, y, INPUT_SHAPE = load_dataset(args.organism)

    #    (x_train, y_train), (x_test, y_test) = load_imdb()

    kf = StratifiedShuffleSplit(n_splits=CV, random_state=34267)
    kf.get_n_splits(X, y)

    actual_partition = 0

    for train_index, test_index in kf.split(X, y):
        actual_partition += 1
        print('>>> Testing PARTITION {}'.format(actual_partition))
        (x_train, y_train), (x_test, y_test) = load_partition(train_index, test_index, X, y)
        print(x_train.shape)
        print(y_train.shape)

        # Define model
        model = create_classifier(CLASSIFIER, INPUT_SHAPE, N_CLASS, verbose=True)
        #        plot_model(model, to_file=args.weights_dir + '/model.png', show_shapes=True)

        # Train model and get weights
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args, actual_partition=actual_partition)
        K.clear_session()

        # Select best weights for this partition
        (stats, weight_file) = get_best_weight(args, actual_partition)
        print('Selected BEST: {} ({})'.format(weight_file, stats.Mcc))
        #        model.save_weights(args.weights_dir + '/best_trained_model_partition_{}.h5'.format(actual_partition) )
        #        print('Best Trained model for partition {} saved to \'%s/best_trained_model_partition_{}.h5\''.format(actual_partition, args.weights_dir, actual_partition))

        # Allocate results of best weights for this partition
        allocate_stats(stats)

        # break

    for metric in headers[1:]:
        EXPERIMENT_INFO[metric] = {'mean': np.mean(results[metric]), 'std': np.std(results[metric])}
        for partition, value in enumerate(results[metric]):
            EXPERIMENT_INFO[metric][str(partition)] = float(value)

    MEAN_MCC = '{:05d}'.format(int(np.mean(results['mcc']) * 10000))
    STD_MCC = '{:05d}'.format(int(np.std(results['mcc']) * 10000))
    MODEL_TYPE = args.model
    out_file_name = '{}_{}_mcc-{}-{}'.format(TIMESTAMP, MODEL_TYPE, MEAN_MCC, STD_MCC)

    # Write experiment INFO to CSV and JSON
    df = pd.DataFrame(results, columns=headers)
    aggregations = np.zeros((2, len(headers)))
    for i in range(len(headers)):
        aggregations[0, i] = np.mean(df.iloc[:, i])
        aggregations[1, i] = np.std(df.iloc[:, i])
    df2 = pd.DataFrame(aggregations, columns=headers)
    df.append(df2)
    print('+'*50)
    print(df)
    print(aggregations)
    print(df2)
    print('+' * 50)
    f_name = TIMESTAMP + '_' + CLASSIFIER
    df.to_csv(os.path.join(args.results_dir, f_name + '.csv'))
    with open(os.path.join(args.results_dir, f_name + '.json'), 'w') as fp:
        json.dump(EXPERIMENT_INFO, fp, sort_keys=True, indent=4)
