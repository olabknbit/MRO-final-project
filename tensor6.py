from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.estimator import regression


def get_data():
    # Data loading and preprocessing
    from tflearn.datasets import cifar10
    (X, Y), (X_test, Y_test) = cifar10.load_data()
    X, Y = shuffle(X, Y)
    Y = to_categorical(Y, 10)
    Y_test = to_categorical(Y_test, 10)
    return (X, Y), (X_test, Y_test)


def get_network():
    # Building convolutional network
    network = input_data(shape=[None, 32, 32, 3], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    return network


def main():
    name = 'model6'
    (X, Y), (X_test, Y_test) = get_data()
    network = get_network()

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoints/' + name + '.tfl.ckpt')
    import os.path
    if os.path.exists('checkpoints/' + name + '.tfl'):
        model.load('checkpoints/' + name + '.tfl')
    model.fit({'input': X}, {'target': Y}, n_epoch=12,
              validation_set=({'input': X_test}, {'target': Y_test}),
              snapshot_step=100, show_metric=True, batch_size=96, run_id='cifar10_cnn')

    # Manually save model
    model.save('checkpoints/' + name + '.tfl')


main()
