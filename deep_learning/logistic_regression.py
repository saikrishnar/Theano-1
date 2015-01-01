import sys
import time
import numpy
import theano
import theano.tensor as T

import util

class LogisticRegression(object):
    """
    This class defines a multiclass logistic regression classifier.
    The class is created with one batch of mini-batch stochastic gradient
    descent algorithm as in input
    """
    def __init__(self, batch, n_in, n_out):
        """Initialize the parameters of Logistic Regression
        :type batch: theano.tensor.TensorType
        :param batch: Symbolic variable for one batch of mini-batch SGD

        :type n_in: int
        :param n_in: Number of features or input units
        
        :type n_out: int
        :param n_out: Number of classes or output units
        """
        # Weights
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               borrow=True,
                               name="W")
        
        # Constant term
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                               borrow=True,
                               name="b")

        # P(y_i|X, W, b) = softmax = exp(W_i.X + b_i) / sum(exp(WX + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(batch, self.W) + self.b)

        # y_pred = argmax(P(y_i | X, W, b))
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Parameters of the model
        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        """Performs -sum(log(p(y_i | X, W, b))) for given y
        Note: We perform mean instead of sum so that the algorithm
        is independent of batch size in minibatch
        
        :type y: theano.tensor.TensorType
        :param y: observed classes in minibacth
        """

        # First find p_y_given_x for given y's
        log_likelihood = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        return -T.mean(log_likelihood)


    def error(self, y):
        """ Find mean(y != y_pred) for given inputs
        
        :type y: theano.tensor.TensorType
        :param y: observed classes in minibatch
        """
        return T.mean(T.neq(self.y_pred, y))
        

def sgd_optimize(learning_rate=0.13,
                 n_epochs=1000,
                 batch_size=600):
    
    train, valid, test = util.load()
    train_size = 50000
    print "loading 0 - ", train_size, " train inputs in gpu memory"
    train_x, train_y = util.create_theano_shared(train, train_size)

    valid_size = 10000
    print "loading 0 - ", valid_size, " validation inputs in gpu memory"
    valid_x, valid_y = util.create_theano_shared(valid, valid_size)

    test_size = 10000
    print "loading 0 - ", test_size, " test inputs in gpu memory"
    test_x, test_y = util.create_theano_shared(test, test_size)

    print " building model..."
    # Create symbolic variables for models
    index = T.lscalar()
    x = T.matrix("x")
    y = T.ivector("y")
        
    # Construct the classifier
    classifier = LogisticRegression(batch=x, n_in=train[0].shape[1], n_out=10)

    # Create cost and theano functions
    cost = classifier.negative_log_likelihood(y)

    # Perform sgd
    g_W = T.grad(cost, wrt=classifier.W)
    g_b = T.grad(cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
             
    # train
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={
                                    x: train_x[index*batch_size: (index+1)*batch_size],
                                    y: train_y[index*batch_size: (index+1)*batch_size]
                                  })
                                
    # valid
    valid_model = theano.function(inputs=[index],
                                  outputs=classifier.error(y),
                                  givens = {
                                    x: valid_x[index*batch_size: (index+1)*batch_size],
                                    y: valid_y[index*batch_size: (index+1)*batch_size]
                                  })
                              
    # test
    test_model  = theano.function(inputs=[index],
                                  outputs=classifier.error(y),
                                  givens = {
                                    x: test_x[index*batch_size: (index+1)*batch_size],
                                    y: test_y[index*batch_size: (index+1)*batch_size]
                                  })


    ############################
    # Parameters for training model
    ############################
    n_train_batches = train[0].shape[0] / batch_size
    n_valid_batches = valid[0].shape[0] / batch_size
    n_test_batches  = test[0].shape[0]  / batch_size

    patience = 5000 # Look at 5000 examples irrespective of whether there is an improvement
    patience_increase = 2 # If the score is increased, scale patience by 2
    improve_threshold = 0.995 # Tolerate 0.995 of current error
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    
    best_validation_loss = numpy.inf 
    mean_test_loss = 0
    start_time = time.clock()
        
    epoch = 0
    done_looping = False
    
    print " Training the model..."
    
    while epoch < n_epochs and (not done_looping):
        epoch += 1
        
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)    
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [valid_model(j) for j in xrange(n_valid_batches)]
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
                    # Improve patience of mean_validation_loss is best_validation_loss * improve_threshold
                    if mean_validation_loss < best_validation_loss * improve_threshold:
                        patience = max(patience, iter * patience_increase)
                    # update best_validation_loss
                    best_validation_loss = mean_validation_loss
                    test_losses = [test_model(j) for j in xrange(n_test_batches)]
                    mean_test_loss = numpy.mean(test_losses)
                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
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
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., mean_test_loss * 100.)
    )
    print 'The code run for %d epochs, with %d secs, (%f epochs/sec)' % (
          epoch, end_time - start_time, 1. * epoch / (end_time - start_time))
    

if __name__ == "__main__":
    sgd_optimize()
