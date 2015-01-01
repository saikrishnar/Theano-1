import numpy
import theano
import theano.tensor as T

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
        row_index = T.arange(y.size[0])
        likelihood = self.p_y_given_x[row_index, y]
        return -T.mean(T.log(likelihood))


    def error(self, y):
        """ Find mean(y != y_pred) for given inputs
        
        :type y: theano.tensor.TensorType
        :param y: observed classes in minibatch
        """
        return T.mean(T.neq(self.y_pred, y))
        
