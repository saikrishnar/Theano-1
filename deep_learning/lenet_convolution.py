import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import util
from mlp import HiddenLayer
from logistic_regression import LogisticRegression

class LeNetConvPoolLayer(object):
    """ Pool Layer of Convolution network """

    def __init__(self, input, filter_shape, image_shape, random_generator, pool_size=(2, 2)):
        """
        Allocate all internal parameters of convolution metwork

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor (4D) 

        :type filter_shape: 4-D list
        :param filter_shape: (size of output features, size of input features, width, height)

        :type image_shape: 4-D list
        :param image_shape: Shape of image tensor (batch_size, number of input feature maps,
                                                   image width, image height)

        :type random_generator: numpy.random.RandomState
        :param random_generator: Random number generator
        
        :type pool_size: 2-D list
        :param pool_size: downsampling (pooling) factor for max-pooling (#row, #col)
        """
        self.input = input

        # Create weight and balance matrices matrices
        # input_size of hidden layer = num input features * width * height
        input_size = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        output_size = filter_shape[0] * filter_shape[2] * filter_shape[3] / numpy.prod(pool_size)

        # Range of random initialization of W is +- sqrt(6 / (input_size + output_size))
        W_bound = numpy.sqrt(6. / (input_size + output_size))

        # Weight matrix
        self.W = theano.shared(numpy.asarray(
                                   random_generator.uniform(
                                      low = -W_bound,
                                      high = W_bound,
                                      size = filter_shape),
                                   dtype = theano.config.floatX),
                               name = "W",
                               borrow = True)
        
        # Bias matrix
        b_bound = numpy.sqrt(4. / (input_size + output_size))
        self.b = theano.shared(numpy.asarray(
                                   random_generator.uniform(
                                     low = -b_bound,
                                     high = b_bound,
                                     size = (filter_shape[0], )
                                   ),
                                   dtype = theano.config.floatX),
                               name = "b",
                               borrow = True)

        # Create convolution of W with input
        conv_out = conv.conv2d(self.input, self.W,
                               filter_shape=filter_shape,
                               image_shape=image_shape)
        
        # Create pooled output for maxpooling
        pool_out = downsample.max_pool_2d(conv_out, pool_size, ignore_border=True)

        # Finally, output
        self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store params
        self.params = [self.W, self.b]


def sgd_optimize(learning_rate=0.1,
                 n_epochs=200,
                 batch_size=500,
                 nkerns=[20, 50]):
    # Load input
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

    # Create Layer0 of Lenet Model
    layer0_input = x.reshape( (batch_size, 1, 28, 28) )
    filter_shape0 = (nkerns[0], 1, 5, 5)
    image_shape0 = (batch_size, 1, 28, 28) 
    layer0 = LeNetConvPoolLayer(layer0_input, filter_shape0, image_shape0, random_generator)
    
    # Create Layer1 of Lenet model
    filter_shape1 = (nkerns[1], nkerns[0], 5, 5)
    image_shape1 = (batch_size, nkerns[0], 12, 12)
    layer1 = LeNetConvPoolLayer(layer0.output, filter_shape1, image_shape1, random_generator)

    # Create Layer2 which is a simple MLP hidden layer
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(layer2_input, nkerns[1] * 4 * 4, 500, random_generator)

    # Finally, Layer3 is LogisticRegression layer
    layer3 = LogisticRegression(layer2.output, 500, 10)

    # Define error
    error = layer3.error(y)

    # Create cost function
    cost = layer3.negative_log_likelihood(y)

    # Gradient and update functions
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, wrt=params)
    updates = list()
    for i in range(len(params)):
        updates.append( (params[i], params[i] - learning_rate * grads[i]) )

    # Train model
    train_model = theano.function(
                    inputs=[index],
                    outputs=cost,
                    updates=updates,
                    givens = {
                       x: train_x[index*batch_size : (index+1)*batch_size],
                       y: train_y[index*batch_size : (index+1)*batch_size]
                    })

    # Valid model
    valid_model = theano.function(
                    inputs=[index],
                    outputs=error,
                    givens = {
                       x: valid_x[index*batch_size : (index+1)*batch_size],
                       y: valid_y[index*batch_size : (index+1)*batch_size]
                    })
    
    # Test Model 
    test_model  = theano.function(
                    inputs=[index],
                    outputs=error,
                    givens={
                       x: test_x[index*batch_size : (index+1)*batch_size],
                       y: test_y[index*batch_size : (index+1)*batch_size]
                    })


if __name__ == "__main__":
    sgd_optimize()
