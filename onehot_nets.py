#!/usr/bin/env python
"""
Keras implementation of ConvNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       python ConvNet.py
       python ConvNet.py --epochs 100
       python ConvNet.py --epochs 100 --num_routing 3
       ... ...

    
"""
import os
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


headers = ['partition','mcc','f1','sn','sp','acc','prec','tp','fp','tn', 'fn']   
results = {'partition':[],'mcc':[],'f1':[],'sn':[],'sp':[],'acc':[],'prec':[],'tp':[],'fp':[],'tn':[],'fn':[]}

max_features = 79
maxlen = 16
prefix_name = ''

def ConvNet():
    from keras import layers, models
    from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
    from keras.preprocessing import sequence
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=(4, maxlen, 1))

    # Block 1

    # 1x1 down-sample conv
    conv_x = layers.BatchNormalization()(x)
    conv_x = layers.Activation('relu')(conv_x)
    conv_x = layers.Conv2D(filters=32, kernel_size=(4,1), strides=(1,2))(conv_x)
    # LxL conv
    conv_y = layers.BatchNormalization()(conv_x)
    conv_y = layers.Activation('relu')(conv_y)
    conv_y = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(conv_y)
    # 1x1 up-sample conv
    conv_z = layers.BatchNormalization()(conv_y)
    conv_z = layers.Activation('relu')(conv_z)
    conv_z = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(conv_z)
    # Identity branch - expand channels for the sum
    shortcut_y = layers.Conv2D(filters=128, kernel_size=(4, 1), strides=(1,2))(x)
    shortcut_y = layers.BatchNormalization()(shortcut_y)

    output_block_1 = layers.add([shortcut_y, conv_z])
    output_block_1 = layers.Activation('relu')(output_block_1)

    # Block 2
    # 1x1 down-sample conv
    conv_x = layers.BatchNormalization()(output_block_1)
    conv_x = layers.Activation('relu')(conv_x)
    conv_x = layers.Conv2D(filters=32, kernel_size=(1,1), strides=(1,2))(conv_x)
    # LxL conv
    conv_y = layers.BatchNormalization()(conv_x)
    conv_y = layers.Activation('relu')(conv_y)
    conv_y = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(conv_y)
    # 1x1 up-sample conv
    conv_z = layers.BatchNormalization()(conv_y)
    conv_z = layers.Activation('relu')(conv_z)
    conv_z = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(conv_z)
    # Identity branch - expand channels for the sum
    shortcut_y = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1,2))(output_block_1)
    shortcut_y = layers.BatchNormalization()(shortcut_y)

    output_block_2 = layers.add([shortcut_y, conv_z])
    output_block_2 = layers.Activation('relu')(output_block_2)

    # Block 3
    # 1x1 down-sample conv
    conv_x = layers.BatchNormalization()(output_block_2)
    conv_x = layers.Activation('relu')(conv_x)
    conv_x = layers.Conv2D(filters=64, kernel_size=(1,1), strides=(1,2))(conv_x)
    # LxL conv
    conv_y = layers.BatchNormalization()(conv_x)
    conv_y = layers.Activation('relu')(conv_y)
    conv_y = layers.Conv2D(filters=64, kernel_size=(1, 3), padding='same')(conv_y)
    # 1x1 up-sample conv
    conv_z = layers.BatchNormalization()(conv_y)
    conv_z = layers.Activation('relu')(conv_z)
    conv_z = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same')(conv_z)
    # Identity branch - expand channels for the sum
    shortcut_y = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1,2))(output_block_2)
    shortcut_y = layers.BatchNormalization()(shortcut_y)

    output_block_3 = layers.add([shortcut_y, conv_z])
    output_block_3 = layers.Activation('relu')(output_block_3)

    # Block 4
    # 1x1 down-sample conv
    conv_x = layers.BatchNormalization()(output_block_3)
    conv_x = layers.Activation('relu')(conv_x)
    conv_x = layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,2))(conv_x)
    # LxL conv
    conv_y = layers.BatchNormalization()(conv_x)
    conv_y = layers.Activation('relu')(conv_y)
    conv_y = layers.Conv2D(filters=128, kernel_size=(1, 3), padding='same')(conv_y)
    # 1x1 up-sample conv
    conv_z = layers.BatchNormalization()(conv_y)
    conv_z = layers.Activation('relu')(conv_z)
    conv_z = layers.Conv2D(filters=1024, kernel_size=(1, 1), padding='same')(conv_z)
    # Identity branch - expand channels for the sum
    shortcut_y = layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=(1,2))(output_block_3)
    shortcut_y = layers.BatchNormalization()(shortcut_y)

    output_block_4 = layers.add([shortcut_y, conv_z])
    output_block_4 = layers.Activation('relu')(output_block_4)

    drop = layers.Dropout(0.2)(output_block_4)

    gap_layer = layers.GlobalAveragePooling2D()(drop)

    # flat1 = layers.Flatten()(pool2)

    # dense1 = layers.Dense(128, activation='relu')(gap_layer)
    # drop2 = layers.Dropout(0.1)(dense1)
    # outputs = layers.Dense(1, activation='sigmoid')(drop2)
    outputs = layers.Dense(1, activation='sigmoid')(gap_layer)

    return models.Model(inputs=[x], outputs=outputs)

def get_calls():

    from keras import callbacks as C
    import math

    cycles = 50
    calls = list()
    calls.append( C.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', save_best_only=True, save_weights_only=True, verbose=0) )
    calls.append( C.CSVLogger(args.save_dir + '/log.csv') )
    calls.append( C.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs/{}'.format(actual_partition), batch_size=args.batch_size, histogram_freq=args.debug) )
    calls.append( C.EarlyStopping(monitor='val_loss', patience=50, verbose=0))
    calls.append( C.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=0) )
    calls.append( C.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch)) )
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
    global prefix_name

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    calls = get_calls()

    lossfunc = ['mse', 'binary_crossentropy']
    # compile the model

#    validation_data=[[x_test, y_test], [y_test, x_test]]
#    validation_split=0.1
#    seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    seeds = [23, 29, 31]
#    seeds = [23, 29]
    for s in range(len(seeds)):
        seed = seeds[s]
        print('{} Train on SEED {}'.format(s, seed))
        
        name = args.save_dir + '/'+prefix_name+'-partition_{}-seed_{}-weights.h5'.format(actual_partition, s)
#        calls[0] = C.ModelCheckpoint(name + '-{epoch:02d}.h5', save_best_only=True, save_weights_only=True, verbose=1)
        calls[0] = C.ModelCheckpoint(name, save_best_only=True, save_weights_only=True, verbose=1)
        
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
            loss=lossfunc[1],
#            loss=lossfunc[0],
            # loss_weights=[1., args.lam_recon],
            metrics=['accuracy']
        )

        kf = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
        kf.get_n_splits(x_train, y_train)

        for t_index, v_index in kf.split(x_train, y_train):
            
            X_train, X_val = x_train[t_index], x_train[v_index]
            Y_train, Y_val = y_train[t_index], y_train[v_index]
            
            val_data=(X_val, Y_val)
            
            model.fit(x=X_train, y=Y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=val_data, callbacks=calls, verbose=0)

#            model.save_weights(args.save_dir + '/trained_model.h5')
#            print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

#    from utils import plot_log
#    plot_log(args.save_dir + '/log.csv', show=True)

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
    global max_features
    global maxlen
    
    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)
    
    k = 1
    max_features = 4**k
    samples = SequenceNucHotvector(npath, ppath)
    
    X, y = samples.getX(), samples.getY()
#    X = X.reshape(-1, 38, 79, 1).astype('float32')
    X = X.astype('int32')
#     ini = 59
# #    ini = 199
#     X = X[:, (ini-30):(ini+11)]
    y = y.astype('int32')
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    maxlen = X.shape[2]
    return X, y

def load_partition(train_index, test_index, X, y):
    x_train = X[train_index,:]
    y_train = y[train_index]
    
    x_test = X[test_index,:]
    y_test = y[test_index]
    
#    y_train = to_categorical(y_train.astype('float32'))
#    y_test = to_categorical(y_test.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


def get_best_weight(args, actual_partition):
    global prefix_name
              
    # Select weights 
    file_prefix = prefix_name+'-partition_{}'.format(actual_partition)
    file_sufix = '-weights.h5'
    model_weights = [ x for x in os.listdir(args.save_dir+'/') if x.startswith(file_prefix) and x.endswith(file_sufix) ]
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
        model = ConvNet()
        model.load_weights(args.save_dir + '/' + weight_file)
        
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
    model = ConvNet()
    model.load_weights(args.save_dir + '/' + selected_weight)
    model.save_weights(args.save_dir + '/'+prefix_name+'-partition_{}-best_weights.h5'.format(actual_partition))
    
    K.clear_session()
    
    # Delete others weights
    for i in range(len(model_weights)):
        weight_file = model_weights[i]
        print('Deleting weight: {}'.format(weight_file))
        path = args.save_dir + '/' + weight_file
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

def get_args(): 
    # setting the hyper parameters
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float, help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")

    parser.add_argument('--kernel1_size', default=3, type=int, help="Size of kernel of convolutional operation. Should > 0.")  # kernel1_size should > 0
    parser.add_argument('--kernel1_strides', default=2, type=int, help="strides length of convolutional operation. Should > 0.")  # kernel1_strides should > 0
    parser.add_argument('--num_kernel1', default=64, type=int, help="Number of filters on convolutional operation. Should > 0.")  # num_kernel1 should > 0
    parser.add_argument('--pool1_size', default=3, type=int, help="Size of pooling window. Should > 0.")  # pool1_size should > 0
    parser.add_argument('--pool1_strides', default=2, type=int, help="strides length of pooling window. Should > 0.")  # pool1_strides should > 0

    parser.add_argument('--kernel2_size', default=3, type=int, help="Size of kernel of convolutional operation. Should > 0.")  # kernel1_size should > 0
    parser.add_argument('--kernel2_strides', default=2, type=int, help="strides length of convolutional operation. Should > 0.")  # kernel1_strides should > 0
    parser.add_argument('--num_kernel2', default=256, type=int, help="Number of filters on convolutional operation. Should > 0.")  # num_kernel1 should > 0
    parser.add_argument('--pool2_size', default=3, type=int, help="Size of pooling window. Should > 0.")  # pool1_size should > 0
    parser.add_argument('--pool2_strides', default=2, type=int, help="strides length of pooling window. Should > 0.")  # pool1_strides should > 0

#    parser.add_argument('--shift_fraction', default=0.0, type=float, help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', default=1, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int, help="Size of embedding vector. Should > 0.")
    parser.add_argument('--weights', default=None)
    parser.add_argument('-o', '--organism', default=None, help="The organism used for test. Generate auto path for fasta files. Should be specified when testing")
    
    args = parser.parse_args()    
    return args
    
if __name__ == "__main__":
    
    prefix_name
    n_cv = 10
    
    args = get_args()

    prefix_name = 'conv1_org_{}-batch_{}-kernel_{}-pool_{}-cv_{}'.format(args.organism, args.batch_size, args.kernel1_size, args.pool1_size, n_cv)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    X, y = load_dataset(args.organism)
    
#    (x_train, y_train), (x_test, y_test) = load_imdb()

    
    kf = StratifiedShuffleSplit(n_splits=n_cv, random_state=34267)
    kf.get_n_splits(X, y)
    
    actual_partition = 0
     
    for train_index, test_index in kf.split(X, y):
        actual_partition+=1
        print('>>> Testing PARTITION {}'.format(actual_partition))
        (x_train, y_train), (x_test, y_test) = load_partition(train_index, test_index, X, y)
        print(x_train.shape)
        print(y_train.shape)
        
        # Define model
        model = ConvNet()
        model.summary()
#        plot_model(model, to_file=args.save_dir + '/model.png', show_shapes=True)
        
        # Train model and get weights
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args, actual_partition=actual_partition)
        K.clear_session()
        
        # Select best weights for this partition
        (stats, weight_file) = get_best_weight(args, actual_partition)                    
        print('Selected BEST: {} ({})'.format(weight_file, stats.Mcc))
#        model.save_weights(args.save_dir + '/best_trained_model_partition_{}.h5'.format(actual_partition) )
#        print('Best Trained model for partition {} saved to \'%s/best_trained_model_partition_{}.h5\''.format(actual_partition, args.save_dir, actual_partition))
        
        # Allocate results of best weights for this partition
        allocate_stats(stats)
        
        # break
        
    # Write results of partitions to CSV
    df = pd.DataFrame(results, columns=headers)
    df.to_csv('results_'+prefix_name+'.csv')
