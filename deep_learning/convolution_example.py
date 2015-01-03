import sys
import numpy
import theano
import pylab
import theano.tensor as T
from theano.tensor.nnet import conv
from PIL import Image

def create_convolution_filter(n_input_filters=3, n_output_filters=2, locality_size=9):
    rng = numpy.random.RandomState(100)

    # Create a 4D tensor imput
    input = T.tensor4(name="input")

    # The convolution parameters are as follows
    # We take 9 X 9 submatrix of input image
    # Use n_input_filters feature maps (e.g. 3 corresponding to R, G, B)
    # Create n_output_filters output feature maps
    w_shape = (n_output_filters, n_input_filters, locality_size, locality_size)

    # Random initialization of weights. The bounds are +-1 / sqrt(n_input_filters * locality_size * locality_size)
    w_bound = numpy.sqrt(numpy.prod(w_shape[1:]))

    # Create shared variable for weight
    W = theano.shared(numpy.asarray(
                      rng.uniform(low = -1.0 / w_bound, 
                                  high = 1.0 / w_bound,
                                  size = w_shape),
                      dtype=input.dtype),
                      name="W")

    # Bias terms are also initialized using random number generator. 
    # Two bias terms for two output maps
    b_shape = (n_output_filters, )
    b = theano.shared(numpy.asarray(
                      rng.uniform(low = -5.0, high = 5.0, size = b_shape),
                      dtype = input.dtype),
                      name="b")

    # Create convolution
    conv_out = conv.conv2d(input, W)

    # Create the output
    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
    conv_function = theano.function([input], output)
    return conv_function


def plot_convoluted_images(img, n_input_filters=3, n_output_filters=2, locality_size=9):
    # Create a 4D tensor from image of shape (1, 3, width, height)
    width = img.shape[0]
    height = img.shape[1]
    img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, width, height)
    
    # Create convoluted images
    conv_filter = create_convolution_filter(n_input_filters, n_output_filters, locality_size)
    conv_images = conv_filter(img_)

    # Plot the outputs
    pylab.subplot(1, n_output_filters+1, 1); pylab.axis('off'); pylab.imshow(img)
    pylab.gray();

    # recall that the convOp output (filtered image) is actually a "minibatch",
    # of size 1 here, so we take index 0 in the first dimension:
    for i in range(n_output_filters):
        pylab.subplot(1, n_output_filters+1, i+2); pylab.axis('off'); pylab.imshow(conv_images[0, i, :, :])

    # Finally show
    pylab.show()


if __name__ == "__main__":
    # Work on a small example
    pic = Image.open(open(sys.argv[1]))
    img = numpy.asarray(pic, dtype='float64') / 256.

    plot_convoluted_images(img, locality_size=4, n_output_filters=4)

