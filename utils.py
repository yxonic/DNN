import gzip
import pickle

import numpy as np
import theano
import theano.tensor as T


def load_MNIST():
    """Function that loads MNIST dataset
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set


def shared_dataset(data_xy):
    """Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared_dataset
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')


def make_param(name, value):
    """Function that makes parameters into theano shared variables for DNN
    layers.
    """
    return theano.shared(
        value=value,
        name=name,
        borrow=True
    )
