import cPickle
import gzip
import numpy
import theano
from theano import tensor as T

SHARED_SIZE = 10000

def load():
    """
    The data is stored in data/mnist.pickle.gz file. This method unzips and unpicles data
    and loads it into three variables: train, test and valid
    """
    f = gzip.open("data/mnist.pkl.gz", "rb")
    train, valid, test = cPickle.load(f)
    f.close()
    return (train, valid, test)

def create_theano_shared(data, shared_size = SHARED_SIZE, shared_index = 0):
    """
    data is a tuple (x, y) where x is real-array and y is integer class label
    """
    shared_x = theano.shared(numpy.asarray(data[0][shared_index*shared_size:(shared_index+1)*shared_size], dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data[1][shared_index*shared_size:(shared_index+1)*shared_size], dtype=theano.config.floatX))
    return (shared_x, T.cast(shared_y, 'int32'))
