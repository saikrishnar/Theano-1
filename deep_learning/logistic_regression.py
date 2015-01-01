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
        self.p_y_given_x = T.nnet.softmax(T.dot(self.W, batch) + self.b)

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
                 batch_size=600,
                 gpu_size = 12000):
    
    train, valid, test = util.load()
    num_gpu_loads = int(numpy.ceil(train[0].shape[0] / float(gpu_size)))
    for i in range(num_gpu_loads):
        print "loading ", i*gpu_size, "-", (i+1)*gpu_size, "in gpu memory"
        train_x, train_y = util.create_theano_shared(train, gpu_size, i)
        valid_x, valid_y = util.create_theano_shared(valid, gpu_size, i)
        test_x, test_y = util.create_theano_shared(test,  gpu_size, i)
    
        n_train_batches = gpu_size / batch_size
        n_valid_batches = gpu_size / batch_size
        n_test_batches  = gpu_size / batch_size

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
        
        ######################
        # Train the model
        #####################
        
        # Clear gpu memory 
        train_x.set_value([[]])
        valid_x.set_value([[]])
        test_x.set_value([[]])

if __name__ == "__main__":
    sgd_optimize()
