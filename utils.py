import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import mlflow

def get_args():
    # setting the hyper parameters
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='CNN_01')
    parser.add_argument('--coding', default='onehot') # onehot, embedding
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_class', default=1, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--cv', default=2, type=int)
    parser.add_argument('--seeds', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float, help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    #    parser.add_argument('--shift_fraction', default=0.0, type=float, help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', default=1, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--weights_dir', default=os.path.join('.', 'weights'))
    parser.add_argument('--best_weights_dir', default=os.path.join('.', 'best_weights'))
    parser.add_argument('--base_results_dir', default=os.path.join('.', 'results'))
    parser.add_argument('--is_training', default=1, type=int, help="Size of embedding vector. Should > 0.")
    parser.add_argument('--weights', default=None)
    parser.add_argument('-o', '--organism', default=None, help="The organism used for test. Generate auto path for fasta files. Should be specified when testing")

    args = parser.parse_args()
    return args


def load_dataset(organism, coding_type='onehot', k=1, nrange=None):
    from .ml_data import SequenceNucsData, SequenceNucHotvector, SequenceMotifHot

    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)

    if coding_type == 'onehot':
        samples = SequenceNucHotvector(npath, ppath)
        X = samples.getX().astype('int32')
        y = samples.getY().astype('int32')
    elif coding_type == 'embedding':
        samples = SequenceNucsData(npath, ppath, k=k)
        X = samples.getX().astype('int32')
        y = samples.getY().astype('int32')

    if nrange is not None and type(nrange) == tuple and len(nrange) == 2:
        downstream = nrange[0]
        upstream = nrange[1]
        if len(X.shape) == 2:
            if X.shape[1] == 81:
                tss_pos = 59
            elif X.shape[1] == 251:
                tss_pos = 199
            X = X[:, (tss_pos-downstream):(tss_pos+upstream)]
        elif len(X.shape) == 3:
            pass

    #    X = X.reshape(-1, 38, 79, 1).astype('float32')
    #     ini = 59
    # #    ini = 199
    #     X = X[:, (ini-30):(ini+11)]
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    return X, y, X.shape[1:]

def set_log_params(args):
    mlflow.log_param('cv', args.cv)
    mlflow.log_param('model', args.model)
    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('seeds', args.seeds)
    mlflow.log_param('patience', args.patience)
    mlflow.log_param('lr', args.lr)
    mlflow.log_param('lr_decay', args.lr_decay)

def set_log_metrics(results):
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

    # mlflow.log_metric('tp', results['tp'])
    # mlflow.log_metric('fp', results['fp'])
    # mlflow.log_metric('tn', results['tn'])
    # mlflow.log_metric('fn', results['fn'])


def tsne_plot(model):
    from sklearn.manifold import TSNE
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=30, n_components=2, early_exaggeration=10, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()




def mlflow_flush(args, train_test_cv, train_val_cv):
    import mlflow
    import tensorflow as tf
    run_name = '{}_{}-{}'.format(args.model_type, train_test_cv, train_val_cv)
    with mlflow.start_run(run_name=run_name):
        pass

def mlflow_set_logs_params(**kwargs):
    import mlflow
    for k, v in kwargs.items():
        mlflow.log_param(k, v)

def mlflow_set_logs(results, **kwargs):
    for k, v in kwargs:
        # print(k, v)
        results[k].append(v)
        with mlflow.start_run(run_name=run_name):
            set_log_params(args)
            set_log_hyperparams(hyperparams)
            set_log_metrics(results)
            plot_model_img = '{}-{}-{}.png'.format(args.model_type, args.timestamp, idx)
            tf.keras.utils.plot_model(model, to_file=plot_model_img, show_shapes=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
            mlflow.log_artifact(plot_model_img)

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

# if __name__=="__main__":
#     plot_log('result/log.csv')



