import os
import sys
import numpy
import theano
import time
import util
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_regression import LogisticRegression
from mlp import HiddenLayer
from autoencoder import DenoisingAutoEncoder

class StackedDenoisingAutoEncoders(object):

    def __init__(self,
                 random_generator,
                 theano_random_generator=None,
                 x_dim=28*28,
                 y_dim=10,
                 hidden_layer_sizes=[500, 500],
                 corruption_levels=[0.1, 0.1]):
        """
        """
        # Declare empty sigmoid layer array for MLP
        self.sigmoid_layers = []

        # Declare an empty array of DenoisingAutoEncoder
        self.autoencoder_layers = []
        
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        if theano_random_generator == None:
            self.theano_random_generator = RandomStreams(random_generator.randint(2 ** 30))
        else:
            self.theano_random_generator = theano_random_generator

        # Inputs using Theano
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        
        # Initialize all parameters
        for i in range(self.n_layers):
            # Define x and y dimensions
            if i == 0:
                internal_x_dim = x_dim
            else:
                internal_x_dim = hidden_layer_sizes[i - 1]
            internal_y_dim = hidden_layer_sizes[i]

            # Find inputs
            if i == 0:
                internal_input = self.x
            else:
                internal_input = self.sigmoid_layers[i-1].output

            # Define Sigmoid Layer
            self.sigmoid_layers.append(HiddenLayer(internal_input,
                                                   internal_x_dim,
                                                   internal_y_dim,
                                                   random_generator,
                                                   activation=T.nnet.sigmoid))

            # Define input
            self.autoencoder_layers.append(DenoisingAutoEncoder(random_generator,
                                                                theano_random_generator,
                                                                internal_x_dim,
                                                                internal_y_dim,
                                                                internal_input,
                                                                W=self.sigmoid_layers[i].W,
                                                                b=self.sigmoid_layers[i].b))

            # Uppdate parameters
            self.params.extend(self.sigmoid_layers[i].params)

            
        # Finally add logistic layer
        self.logistic_layer = LogisticRegression(self.sigmoid_layers[-1].output,
                                                 hidden_layer_sizes[-1],
                                                 y_dim)

        self.params.extend(self.logistic_layer.params)

        # These are two important costs
        # Finetuning after pretraining individual AutoEncoders
        self.finetune_cost = self.logistic_layer.negative_log_likelihood(self.y)

        # Error from prediction
        self.error = self.logistic_layer.error(self.y)


    def pretrain(self, train_x, batch_size):
        """Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
        for training the dA
        
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        
        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
        the dA layer
        """
        index = T.iscalar("index")
        corruption_level = T.scalar("corruption_level")
        learning_rate = T.scalar("learning_rate")

        pretrain_functions = []
        for autoencoder in self.autoencoder_layers:

            # Find cost and updates for the layer
            cost, updates = autoencoder.cost_updates(corruption_level, learning_rate)
            
            f = theano.function(
                inputs=[index,
                        theano.Param(corruption_level, default=0.2),
                        theano.Param (learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x : train_x[index * batch_size : (index + 1) * batch_size]
                })

            pretrain_functions.append(f)

        return pretrain_functions


    def finetune(self, train_x, train_y,
                 valid_x, valid_y,
                 test_x, test_y,
                 batch_size, learning_rate):
        """Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set
        
        :type batch_size: int
        :param batch_size: size of a minibatch
        
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        """
        # Define index
        index = T.iscalar("index")

        # Cost and updates in SGD
        grad = T.grad(self.finetune_cost, wrt=self.params)
        updates = list()
        for i in range(len(self.params)):
            updates.append( (self.params[i], self.params[i] - learning_rate * grad[i]) )

        # Define train, valid and test models
        train_model = theano.function(
                         inputs=[index],
                         outputs=self.finetune_cost,
                         updates = updates,
                         givens = {
                           self.x : train_x[index * batch_size : (index + 1) * batch_size],
                           self.y : train_y[index * batch_size : (index + 1) * batch_size]
                         })

        valid_model = theano.function(
                         inputs=[index],
                         outputs=self.error,
                         givens = {
                           self.x : valid_x[index * batch_size : (index + 1) * batch_size],
                           self.y : valid_y[index * batch_size : (index + 1) * batch_size]
                         })

        test_model = theano.function(
                        inputs=[index],
                        outputs=self.error,
                        givens = {
                          self.x : test_x[index * batch_size : (index + 1) * batch_size],
                          self.y : test_y[index * batch_size : (index + 1) * batch_size]
                        })

        return (train_model, valid_model, test_model)


def sgd_optimize(learning_rate=0.1,
                 pretrain_learning_rate=0.001,
                 pretrain_epochs=15,
                 finetune_epochs=1000,
                 batch_size=1):
    # Load datasets
    train, valid, test = util.load()
    print "loading 0 - ", train[0].shape[0], " train inputs in gpu memory"
    train_x, train_y = util.create_theano_shared(train)
        
    print "loading 0 - ", valid[0].shape[0], " validation inputs in gpu memory"
    valid_x, valid_y = util.create_theano_shared(valid)

    print "loading 0 - ", test[0].shape[0], " test inputs in gpu memory"
    test_x, test_y = util.create_theano_shared(test)

    n_train_batches = train[0].shape[0] / batch_size
    n_valid_batches = valid[0].shape[0] / batch_size
    n_test_batches = test[0].shape[0] / batch_size

    random_generator = numpy.random.RandomState(1)
    print "...Building model"
    sd = StackedDenoisingAutoEncoders(random_generator,
                                      hidden_layer_sizes=[1000, 1000, 1000])

    
    print "...Getting pretrain functions"
    pretrain_fns = sd.pretrain(train_x, batch_size)

    #############
    # Pretrain
    ############
    print "... Pre-training model"
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in range(sd.n_layers):
        for epoch in range(pretrain_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrain_fns[i](index=batch_index,
                                         corruption_level=corruption_levels[i],
                                         learning_rate=pretrain_learning_rate))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()
    print "Pretraining code ran for %.2fm" % (end_time - start_time) 

    #############
    # Finetune
    ############

    print "...Fine-tuning model"
    train_model, valid_model, test_model = sd.finetune(train_x, train_y,
                                                       valid_x, valid_y,
                                                       test_x, test_y,
                                                       batch_size, learning_rate)
    util.train_test_model(finetune_epochs, train_model, valid_model, test_model,
                          n_train_batches, n_valid_batches, n_test_batches)
    
if __name__ == "__main__":
    sgd_optimize()
    
