import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from utils import make_param


class LogisticRegression:
    """Class for multi-class logistic regression.

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, input, n_in, n_out):
        """Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of theano
                      architecture

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.input = input

        # initialize parameters with zeros
        self.W = make_param(
            'W',
            np.zeros((n_in, n_out),
                     dtype=theano.config.floatX)
        )
        self.b = make_param(
            'b',
            np.zeros((n_out, ),
                     dtype=theano.config.floatX)
        )
        self.params = [self.W, self.b]

        # symbolic expression of [P(y=i)] given x
        self.p_y = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic expression of predicted y (output)
        self.output = T.argmax(self.p_y, axis=1)

    def negative_log_likelihood(self, y_true):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
            \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y_true: theano.tensor.TensorType
        :param y_true: corresponds to a vector that gives for each example
                       theano correct label
        """
        return -T.mean(T.log(self.p_y)[T.arange(y_true.shape[0]), y_true])

    def errors(self, y_true):
        """Return the error rate on y_true.

        :type y_true: theano.tensor.TensorType
        :param y_true: corresponds to a vector that gives for each example
                       theano correct label
        """
        return T.mean(T.neq(self.output, y_true))


class MLPLayer:
    """Class for a hidden layer in MLP.

    Typical hidden layer of a MLP: units are fully-connected and have
    sigmoidal activation function.
    """
    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, activation=T.tanh):
        """Initialize the parameters of the hidden layer.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # initialize parameters with uniformly-distributed random reals
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = make_param('W', W_values)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = make_param('b', b_values)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        self.activation = activation

        z = T.dot(input, self.W) + self.b
        self.output = activation(z) if activation else z


class ConvPoolLayer:
    """Combined layer of a convolution layer and a max-pooling layer.
    """
    def __init__(self, rng, input, input_shape, filter_shape, poolsize=(2, 2)):
        """Initialize parameters for ConvPoolLayer.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape input_shape

        :type input_shape: tuple of length 4
        :param input_shape: (#samples, #input feature maps, height, weight)

        :type filter_shape: tuple of length 4
        :param filter_shape: (#filters, #input feature maps, height, weight)

        :type poolsize: tuple of length 2
        :param poolsize: the pooling factor (#rows, #cols)
        """
        self.input = input

        # initialize parameters with uniformly-distributed random reals
        n_in = np.prod(filter_shape[1:])
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                 np.prod(poolsize))
        bound = np.sqrt(6. / (n_in + n_out))
        W_values = np.asarray(
            rng.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=theano.config.floatX
        )
        self.W = make_param('W', W_values)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = make_param('b', b_values)
        self.params = [self.W, self.b]

        # output of the convolution layer
        conv_out = conv2d(input=input,
                          filters=self.W,
                          filter_shape=filter_shape,
                          image_shape=input_shape)

        # output of the max-pooling layer
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize,
                                            ignore_border=True)

        # reshape b to (1, #filters, 1, 1) so that we can add it to pooled_out.
        # Notice that each bias will broadcast across samples, width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output_shape = (
            input_shape[0],
            filter_shape[0],
            (input_shape[2] - filter_shape[2] + 1) // poolsize[0],
            (input_shape[3] - filter_shape[3] + 1) // poolsize[1]
        )
