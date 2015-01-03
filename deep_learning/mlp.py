import numpy
import time
import theano
import theano.tensor as T

from logistic_regression import LogisticRegression
import util

class HiddenLayer(object):
    """
    A class to represent one hidden layer of multi-layer perceptron

    We assume that the network is fully connected --- all inputs of previous
    layers are connected to hidden inputs in this layer.
    """

    def __init__(self, input, n_in, n_out, random_generator, W=None, b=None, activation=T.tanh):
        """Initialization - builds inputs, weights for the hidden layer
        :type input: theano.tensor.TensorType
        :param input: input to this layer

        :type n_in: int
        :param n_in: dimensions of input layer (28 * 28 for MNIST)

        :type n_out: int
        :param n_out: dimensions of the output layer

        :type random_generator: numpy.random.RandomState
        :param random_generator: Used to initialize weights

        :type W: theano.shared
        :param W: initial weights for input units

        :type b: theano.shared
        :param b: initial constant terms

        :type activation: theano.Op
        :param activation: sigmoid, tanh etc.
        """
        self.input = input
        self.W = W
        self.b = b

        # Set W and b as theano shared variables
        if self.W == None:
            W_value = random_generator.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size = (n_in, n_out)
                      )
            if activation == theano.tensor.nnet.sigmoid:
                W_value *= 4
                
            self.W = theano.shared(numpy.asarray(W_value, dtype=theano.config.floatX),
                                   name="W",
                                   borrow=True)

        if self.b == None:
            self.b = theano.shared(numpy.zeros((n_out,), dtype=theano.config.floatX),
                                   name="b",
                                   borrow=True)

        # Set symbolic output
        lin_output = T.dot(self.input, self.W) + self.b
        if activation:
            self.output = activation(lin_output)
        else:
            self.output = lin_output

        # Set self.params to be used in other classes
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron class
    This class conbines Hidden-Layer with Logistic Layer defined in
    logistic_regression.py
    """

    def __init__(self, input, n_in, n_hidden, n_out, random_generator):
        """Constructor
        :type input: theano.tensor.TensorType
        :param input: Total external input to MLP

        :type n_in: int
        :param n_in: number of units in input layer

        :type n_hidden: int
        :param n_hidden: number of units in hidden layer

        :type n_out: int
        :param n_out: number of units in output layer

        :type random_generator: numpy.random.RandomState
        :param random_generator: Used to initialize weights in Hidden Layer
        """
        # Define hidden layer
        self.hidden_layer = HiddenLayer(input=input,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        random_generator=random_generator)

        
        # Define logistic layer
        self.logistic_layer = LogisticRegression(batch=self.hidden_layer.output,
                                                 n_in=n_hidden,
                                                 n_out=n_out)


        # L1/L2 Regularization
        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.logistic_layer.W).sum()
        self.L2_sq = (self.hidden_layer.W ** 2).sum() + (self.logistic_layer.W ** 2).sum()
        
        # Define errors and negative log likelihood
        self.negative_log_likelihood = self.logistic_layer.negative_log_likelihood
        self.error = self.logistic_layer.error

        # Parameters of the model
        self.params = self.hidden_layer.params + self.logistic_layer.params


def sgd_optimize(learning_rate=0.01,
                 n_epochs=1000,
                 batch_size=20,
                 L1_lambda=0.00,
                 L2_lambda=0.0001,
                 n_hidden=500):

    # Load inputs
    train, valid, test = util.load()
    print "loading 0 - ", train[0].shape[0], " train inputs in gpu memory"
    train_x, train_y = util.create_theano_shared(train)

    print "loading 0 - ", valid[0].shape[0], " validation inputs in gpu memory"
    valid_x, valid_y = util.create_theano_shared(valid)

    print "loading 0 - ", test[0].shape[0], " test inputs in gpu memory"
    test_x, test_y = util.create_theano_shared(test)

    # Define symbolic input matrices
    print "Building Model..."
    index = T.iscalar()
    x = T.matrix("x")
    y = T.ivector("y")
    random_generator = numpy.random.RandomState(1)

    mlp = MLP(input=x,
              n_in=train[0].shape[1],
              n_hidden=n_hidden,
              n_out=10,
              random_generator=random_generator)

    # Create cost function
    cost = mlp.negative_log_likelihood(y) + L1_lambda * mlp.L1 + L2_lambda * mlp.L2_sq

    # Error term
    error = mlp.error(y)

    # Gradients
    grads = [ T.grad(cost, param) for param in mlp.params ]

    
    #Updates
    updates = list()
    for i in range(len(grads)):
        updates.append( (mlp.params[i], mlp.params[i] - learning_rate * grads[i]) )
    
    # Define train, valid and test functions
    train_model = theano.function(
                    inputs=[index],
                    outputs=cost,
                    updates=updates,
                    givens={
                       x: train_x[index*batch_size : (index+1)*batch_size],
                       y: train_y[index*batch_size : (index+1)*batch_size]
                    })

    valid_model = theano.function(
                    inputs=[index],
                    outputs=error,
                    givens={
                       x: valid_x[index*batch_size : (index+1)*batch_size],
                       y: valid_y[index*batch_size : (index+1)*batch_size]
                    })
    
    test_model  = theano.function(
                    inputs=[index],
                    outputs=error,
                    givens={
                       x: test_x[index*batch_size : (index+1)*batch_size],
                       y: test_y[index*batch_size : (index+1)*batch_size]
                    })


    # Train
    n_train_batches = train[0].shape[0] / batch_size
    n_valid_batches = valid[0].shape[0] / batch_size
    n_test_batches = test[0].shape[0] / batch_size

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

    
if __name__ == "__main__":
    sgd_optimize()
