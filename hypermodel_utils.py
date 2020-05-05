import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model_type', default=0, type=int)
    parser.add_argument('--cv', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--early_stop', default=0, type=int)
    parser.add_argument('--seeds', default=3, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--fasta_dir', default=os.path.join('.', 'fasta'))
    parser.add_argument('--weights_dir', default=os.path.join('.', 'weights'))
    parser.add_argument('--best_weights_dir', default=os.path.join('.', 'best_weights'))
    parser.add_argument('--base_results_dir', default=os.path.join('.', 'results'))
    parser.add_argument('-o', '--organism', default='Bacillus',
                        help="The organism used for test. Generate auto path for fasta files. Should be specified when testing")
    parser.add_argument('--debug', default=1, type=int)
    return parser.parse_args()


def get_yaml(args):
    import yaml
    with open(args.config) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
        print('Using parameters from: {}'.format(args.config))
        print(doc)
        for k, v in doc.items():
            setattr(args, k, v)
    return args


def get_seeds(num_seeds=None):
    seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    if num_seeds and num_seeds < len(seeds):
        return seeds[:num_seeds]
    return seeds


def load_args():
    from datetime import datetime
    import os
    args = get_args()
    args = get_yaml(args)
    args.experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    args.timestamp = datetime.now().strftime("%Y_%m_%d=%H_%M_%S")
    args.seeds = get_seeds(num_seeds=args.seeds)
    print(args)
    return args

def get_results_table():
    headers = ('partition', 'mcc', 'f1', 'sn', 'sp', 'acc', 'prec', 'tp', 'fp', 'tn', 'fn')
    results = {x: [] for x in headers}
    return results

def allocate_stats(results, stats, actual_partition):
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
    return results

def mlflow_logs(args, hyperparams, results, model, idx):
    import mlflow
    import tensorflow as tf
    run_name = '{}_{}'.format(args.model_type, idx)
    with mlflow.start_run(run_name=run_name):
        set_log_params(args)
        set_log_hyperparams(hyperparams)
        set_log_metrics(results)
        plot_model_img = '{}-{}-{}.png'.format(args.model_type, args.timestamp, idx)
        tf.keras.utils.plot_model(model, to_file=plot_model_img, show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96)
        mlflow.log_artifact(plot_model_img)

def set_log_params(args):
    import mlflow
    mlflow.log_param('cv', args.cv)
    mlflow.log_param('model_type', args.model_type)
    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('seeds', args.seeds)
    # mlflow.log_param('patience', args.patience)

def set_log_hyperparams(hyperparams):
    import mlflow
    for k, v in hyperparams.items():
        mlflow.log_param(k, v)

def set_log_metrics(results):
    import mlflow
    import numpy as np
    mlflow.log_metric('mean_mcc', np.mean(results['mcc']))
    mlflow.log_metric('std_mcc', np.std(results['mcc']))

    mlflow.log_metric('mean_f1', np.mean(results['f1']))
    mlflow.log_metric('std_f1', np.std(results['f1']))

    mlflow.log_metric('mean_acc', np.mean(results['acc']))
    mlflow.log_metric('std_acc', np.std(results['acc']))

    mlflow.log_metric('mean_prec', np.mean(results['prec']))
    mlflow.log_metric('std_prec', np.std(results['prec']))

    mlflow.log_metric('mean_sn', np.mean(results['sn']))
    mlflow.log_metric('std_sn', np.std(results['sn']))

    mlflow.log_metric('mean_sp', np.mean(results['sp']))
    mlflow.log_metric('std_sp', np.std(results['sp']))


def load_data_chunks(args):
    from promoter_data import PromoterData, DataChunk
    data_chunks = list()
    for chunk in args.data:
        d = DataChunk(
            k=chunk['k'],
            encode=chunk['encode'],
            _slice=chunk['slice'])
        data_chunks.append(d)
    return data_chunks

def load_partition(train_index, test_index, X, y):
    x_train = X[train_index, :]
    y_train = y[train_index]
    x_test = X[test_index, :]
    y_test = y[test_index]

    return (x_train, y_train), (x_test, y_test)


def eval_model(X, y, args):
    from sklearn.model_selection import StratifiedShuffleSplit
    kf = StratifiedShuffleSplit(n_splits=args.cv, random_state=args.seeds[0])
    kf.get_n_splits(X, y)

    partition_idx = 0
    for train_idx, test_idx in kf.split(X, y):
        partition_idx += 1
        (x_train, y_train), (x_test, y_test) = load_partition(train_idx, test_idx, X, y)

        calls = get_callbacks(args, partition_idx)

        for s_idx, seed in enumerate(args.seeds):
            print('{} Training with SEED {}'.format(s_idx, seed))
            weight_file_name = '{}-{}-partition_{}-seed_{}'.format(args.model_type, args.timestamp, partition_idx,
                                                                   s_idx) + '-epoch_{epoch:02d}-loss_{val_loss:.2f}.hdf5'





def get_callbacks(args, partition_idx):
    import keras
    import tensorflow.keras.callbacks as C

    model_type = args.model_type
    timestamp = args.timestamp
    early_stop = args.early_stop
    t_name = args.weights_dir + '/tensorboard_logs/{}_{}_{}'.format(model_type, timestamp, partition_idx)
    t_name = t_name.replace('/', '\\')  # Correction for Windows paths

    callbacks = list()
    callbacks.append(None)  # Position for Checkpoint
    callbacks.append(C.CSVLogger(args.weights_dir + '/log.csv'))
    callbacks.append(C.TensorBoard(log_dir=t_name, histogram_freq=args.debug))
    if early_stop > 0:
        callbacks.append(C.EarlyStopping(monitor='val_loss', patience=early_stop, verbose=0))
    callbacks.append(
        C.ReduceLROnPlateau(monitor='val_loss', factor=.9, patience=10, min_lr=0.00001, cooldown=0, verbose=0))
    # calls.append(C.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch)))
    # calls.append( C.LearningRateScheduler(schedule=lambda epoch: args.lr * math.cos(1+( (epoch-1 % (args.epochs/cycles)))/(args.epochs/cycles) ) ))
    #    calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.)) )
    return callbacks
