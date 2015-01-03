import sys
import numpy
import theano
import pylab
import theano.tensor as T
from theano.tensor.nnet import conv
from PIL import Image

rng = numpy.random.RandomState(100)

# Create a 4D tensor imput
input = T.tensor4(name="input")

# The convolution parameters are as follows
# We take 9 X 9 submatrix of input image
# Use 3 feature maps corresponding to R, G, B
# Create 2 output feature maps
w_shape = (2, 3, 9, 9)

# Random initialization of weights. The bounds are +-1 / sqrt(3 * 9 * 9)
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
b_shape = (2, )
b = theano.shared(numpy.asarray(
                    rng.uniform(low = -5.0, high = 5.0, size = b_shape),
                    dtype = input.dtype),
                  name="b")

# Create convolution
conv_out = conv.conv2d(input, W)

# Create the output
output = T.tanh(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
conv_function = theano.function([input], output)

# Work on a small example
pic = Image.open(open(sys.argv[1]))
img = numpy.asarray(pic, dtype='float64') / 256.

# Create a 4D tensor from image of shape (1, 3, width, height)
width = img.shape[0]
height = img.shape[1]
img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, width, height)
conv_images = conv_function(img_)

# Plot the outputs
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();

# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(conv_images[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(conv_images[0, 1, :, :])

# Finally show
pylab.show()

