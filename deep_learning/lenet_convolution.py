import numpy
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

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
        self.output = theano.function(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store params
        self.params = [self.W, self.b]
