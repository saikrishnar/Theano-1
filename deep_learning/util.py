import cPickle
import gzip
import numpy
import theano
import time
from theano import tensor as T

def load(filename="mnist.pkl.gz"):
    """
    The data is stored in data/filename file. This method unzips and unpicles data
    and loads it into three variables: train, test and valid
    """
    f = gzip.open("data/{}".format(filename), "rb")
    train, valid, test = cPickle.load(f)
    f.close()
    return (train, valid, test)

def create_theano_shared(data, shared_size = 0, shared_index = 0):
    """
    data is a tuple (x, y) where x is real-array and y is integer class label
    """
    if shared_size == 0:
        shared_x = theano.shared(numpy.asarray(data[0], dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data[1], dtype=theano.config.floatX))
    else:
        shared_x = theano.shared(numpy.asarray(data[0][shared_index*shared_size:(shared_index+1)*shared_size], dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data[1][shared_index*shared_size:(shared_index+1)*shared_size], dtype=theano.config.floatX))
    return (shared_x, T.cast(shared_y, 'int32'))


def train_test_model(n_epochs,
                     train_model,
                     valid_model,
                     test_model,
                     n_train_batches,
                     n_valid_batches,
                     n_test_batches):
    """ Main loop to train, validate and test models
    :type n_epochs: int
    :type train_model: theano.function
    :type validate_model: theano.function
    :type test_model: theano.function
    :type n_train_batches: int
    :type n_valid_batches: int
    :type n_test_batches: int
    """
    # Train parameters
    valid_threshold = 0.995
    patience = 10000
    patience_increase = 2
    
    epoch = 0
    done_looping = False
    best_validation_loss = numpy.inf
    mean_test_loss = 0

    # Main train loop

    start_time = time.clock()
    while epoch < n_epochs and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            my_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if minibatch_index == n_train_batches - 1:
                validation_losses = [ valid_model(valid_index) for valid_index in xrange(n_valid_batches) ]
                mean_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                      (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        mean_validation_loss * 100.
                      )
                    )

                if mean_validation_loss < best_validation_loss:
                    if mean_validation_loss < best_validation_loss * valid_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = mean_validation_loss
                    test_losses = [ test_model(test_index) for test_index in xrange(n_test_batches) ]
                    mean_test_loss = numpy.mean(test_losses)
                    print(
                        '  epoch %i, minibatch %i/%i, test error of best model %f' %
                          (  
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            mean_test_loss * 100.
                          )
                       )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        'Optimization complete. Best validation score of %f with test performance %f. Total time %d' %
        (
          best_validation_loss * 100.,
          mean_test_loss * 100.,
          end_time - start_time
        )
     )

    
