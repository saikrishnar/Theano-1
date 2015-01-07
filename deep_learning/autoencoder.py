import os
import sys
import numpy
import theano
import time
import util
import theano.tensor as T
from util import tile_raster_images
from theano.tensor.shared_randomstreams import RandomStreams

try:
    import PIL.Image as Image
except ImportError:
    import Image
                
class DenoisingAutoEncoder(object):
    """Denoising AutoEncoder

    This class reconstructs a given input x through an encoder y.
    It first projects the input in latent space y, by usual sigmoid transform

       y = s(Wx + b)

    It then uses another sigmoid transform to transform y into a reconstructed representatio
    of x

       z = s(W'x + b')

    The parameters of the model (W, W', b, b') are chosen to optimize total entropy
    W' is sometimes constrained to be transpose of W
    
       L(x, z) = -x . log(z) - (1-x).log(1-z)
    """
    def __init__(self,
                 random_generator,
                 theano_random_generator,
                 x_dim=28*28,
                 y_dim=500,
                 input=None,
                 W=None,
                 b=None,
                 b_prime=None):
        """Constructor

        :type random_generator: numpy.random.RandomState

        :type theano_random_generator: theano.tensor.shared_randomStream.RandomStreams
        :param theano_random_generator: Theano random generator; if None is given one is
                             generated based on a seed drawn from`random_generator`
        :type x_dim: int
        :param x_dim: number of visible units (28 * 28 for MNIST)

        :type y_dim: int
        :param y_dim: number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Weight matrix to transform input x to y

        :type b: theano.tensor.TensorType
        :param b: Bias term during transformation of input x to y

        :type b_prime: theano.tensor.TensorType
        :param b: Bias term during transformation of hidden y to reconstructed z
        """
        
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Initialize weight matrices
        self.W = W
        if self.W == None:
            bounds = self.x_dim + self.y_dim
            W_init = numpy.asarray(random_generator.uniform(
                                   low = -4. * numpy.sqrt(6. / bounds),
                                   high = 4. * numpy.sqrt(6. / bounds),
                                   size = (self.x_dim, self.y_dim)
                                   ),
                                   dtype=theano.config.floatX)

            self.W = theano.shared(W_init, borrow=True, name="W")

        self.W_prime = self.W.T
                                   
        # Initialize bias terms
        self.b = b
        if self.b == None:
            self.b = theano.shared(numpy.zeros(self.y_dim, dtype=theano.config.floatX),
                                   borrow=True,
                                   name="b")

        self.b_prime = b_prime
        if self.b_prime == None:
            self.b_prime = theano.shared(numpy.zeros(self.x_dim, dtype=theano.config.floatX),
                                         borrow=True,
                                         name="b_prime")

        # intialize theano rando, generator
        if theano_random_generator == None:
            self.theano_random_generator = RandomStreams(random_generator.randint(2 ** 30))
        else:
            self.theano_random_generator = theano_random_generator
        
        # Initialize input
        if input == None:
            self.input = T.dmatrix(name="input")
        else:
            self.input = input

        # Params
        self.params = [self.W, self.b, self.b_prime]
        

    def hidden_values(self, input):
        """ Compute y = s(Wx + b)"""
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def reconstructed_input(self, hidden):
        """ Compute z = s(W'x + b') """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


    def corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
        random numbers that it should produce
        second argument is the number of trials
        third argument is the probability of success of any trial
        
        this will produce an array of 0s and 1s where 1 has a
        probability of 1 - ``corruption_level`` and 0 with
        ``corruption_level``
        
        The binomial function return int64 data type by
        default.  int64 multiplicated by the input
        type(floatX) always return float64.  To keep all data
        in floatX when floatX is float32, we set the dtype of
        the binomial to floatX. As in our case the value of
        the binomial is always 0 or 1, this don't change the
        result. This is needed to allow the gpu to work
        correctly as it only support float32 for now.
        """
        return self.theano_random_generator.binomial(size=input.shape, n=1,
                                                     p=1 - corruption_level,
                                                     dtype=theano.config.floatX) * input

    
    def cost_updates(self, corruption_level, learning_rate):
        """
        first compute y = s(Wx + b)
        then compute z = s(W'y + b')
        Then entropy loss = -x . log(z) - (1-x) . log(1-z)
        """
        x_tilde = self.corrupted_input(self.input, corruption_level)
        y = self.hidden_values(x_tilde)
        z = self.reconstructed_input(y)

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = -T.sum (self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=1)

         # note : L is now a vector, where each element is the
         #        cross-entropy cost of the reconstruction of the
         #        corresponding example of the minibatch. We need to
         #        compute the average of all these to get the cost of
        cost = T.mean(L)

        grad = T.grad(cost, wrt=self.params)

        updates = list()
        for i in range(len(self.params)):
            updates.append( (self.params[i], self.params[i] - learning_rate * grad[i]) )

        return (cost, updates)


def sgd_optimize(learning_rate=0.1,
                 n_epochs=15,
                 batch_size=20,
                 output_folder="da_images",
                 corruption_level=0.):

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

    # Define Denoising AutoEncoder
    random_generator = numpy.random.RandomState(1)
    theano_random_generator = RandomStreams(random_generator.randint(2 ** 30))
    da = DenoisingAutoEncoder(random_generator,
                              theano_random_generator,
                              x_dim=28*28,
                              y_dim=500,
                              input=x)

    
    # Define training model
    cost, updates = da.cost_updates(corruption_level=corruption_level, learning_rate=learning_rate)
    train_model = theano.function(
                    inputs=[index],
                    outputs=cost,
                    updates=updates,
                    givens= {
                      x: train_x[index * batch_size : (index+1) * batch_size]
                    })

    n_train_batches = train[0].shape[0] / batch_size

    # Train
    start_time = time.clock()
    for epoch in range(n_epochs):
        c = []
        for minibatch_index in range(n_train_batches):
            c.append(train_model(minibatch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    # Save image
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    image = Image.fromarray(tile_raster_images(
                            X=da.W.get_value(borrow=True).T,
                            img_shape=(28, 28), tile_shape=(10, 10),
                            tile_spacing=(1, 1)))
    image.save('filters_corruption_{}.png'.format(int(corruption_level * 100)))
    os.chdir('../')

if __name__ == "__main__":
    corruption_level = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    sgd_optimize(corruption_level=corruption_level)
    
