import pickle
import timeit

import numpy as np

import theano
import theano.tensor as T

import utils
import layers


def evaluate_lenet(learning_rate=0.1, n_epochs=200,
                   nkerns=[20, 50], batch_size=500):
    rng = np.random.RandomState(12345)

    print("Loading datasets...")
    train_set, valid_set, test_set = utils.load_MNIST()
    train_set_X, train_set_y = utils.shared_dataset(train_set)
    valid_set_X, valid_set_y = utils.shared_dataset(valid_set)
    test_set_X, test_set_y = utils.shared_dataset(test_set)

    # we cut data to batches so that we can efficiently load them to
    # GPU (if needed)
    n_train_batches = train_set_X.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_X.get_value(borrow=True).shape[0]
    n_test_batches = test_set_X.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()         # index to batches

    X = T.matrix("X")
    y = T.ivector("y")

    print("Building the model...")

    # now we construct a 4-layer CNN

    # our inputs are 28*28 images with only one feature map, so we
    # reshape it to (batch_size, 1, 28, 28)
    layer0_input = X.reshape((batch_size, 1, 28, 28))

    # layer0: convolution+max-pooling layer
    layer0 = layers.ConvPoolLayer(
        rng=rng,
        input=layer0_input,
        input_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # layer1: convolution+max-pooling layer
    layer1 = layers.ConvPoolLayer(
        rng=rng,
        input=layer0.output,
        input_shape=layer0.output_shape,
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # layer2: fully-connected hidden layer
    layer2 = layers.MLPLayer(
        rng=rng,
        input=layer1.output.flatten(2),
        n_in=np.prod(layer1.output_shape[1:]),
        n_out=layer1.output_shape[0],
        activation=T.tanh
    )

    # layer3: logistic regression
    layer3 = layers.LogisticRegression(input=layer2.output,
                                       n_in=batch_size, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    # construct functions to compute errors on test/validation sets
    valid_error = theano.function(
        [index],
        layer3.errors(y),
        givens={
            X: valid_set_X[index*batch_size:(index+1)*batch_size],
            y: valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    test_error = theano.function(
        [index],
        layer3.errors(y),
        givens={
            X: test_set_X[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    # a list of all parameters in this model
    params = layer0.params + layer1.params + layer2.params + layer3.params

    grads = T.grad(cost, params)

    # parameter update rule in stochastic gradient descent
    updates = [(param_i, param_i - learning_rate * grad_i)
               for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            X: train_set_X[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    predict_model = theano.function([X], layer3.output)

    print("Training...")

    # we use the early-stopping strategy
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_score = 0.
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done = False

    while (epoch < n_epochs) and (not done):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch-1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print("iter =", iter)

            train_model(minibatch_index)

            if (iter+1) % validation_frequency == 0:
                valid_errors = [valid_error(i)
                                for i in range(n_valid_batches)]
                score = 1 - np.mean(valid_errors)
                print('epoch {}, minibatch {}/{}, validation accuracy {}'
                      .format(epoch, minibatch_index + 1,
                              n_train_batches, score))
                if score > best_validation_score:
                    best_validation_score = score
                    best_iter = iter

                    # increase patience if improvement is large enough
                    if (1-score) < \
                       (1-best_validation_score) * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # test it on test set
                    test_errors = [test_error(i)
                                   for i in range(n_test_batches)]
                    test_score = 1 - np.mean(test_errors)
                    print('    test score:', test_score)

                    # store best model to file
                    with open('tmp/best_cnn.pkl', 'wb') as f:
                        pickle.dump((predict_model, batch_size), f)

            if patience <= iter:
                done = True
                break   # break the batches loop

    end_time = timeit.default_timer()
    print('Finished training. Total time:',
          (end_time - start_time) / 60, 'min')
    print('Best validation score of', best_validation_score,
          'obtained at iter', best_iter)
    print('Precision: ', test_score)

if __name__ == '__main__':
    evaluate_lenet()
