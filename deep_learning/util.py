import cPickle
import gzip
import numpy
import theano
import time
from theano import tensor as T

def load(filename="mnist.pkl.gz"):
    """
    The data is stored in data/filename file. This method unzips and unpicles data
    and loads it into three variables: train, test and valid
    """
    f = gzip.open("data/{}".format(filename), "rb")
    train, valid, test = cPickle.load(f)
    f.close()
    return (train, valid, test)

def create_theano_shared(data, shared_size = 0, shared_index = 0):
    """
    data is a tuple (x, y) where x is real-array and y is integer class label
    """
    if shared_size == 0:
        shared_x = theano.shared(numpy.asarray(data[0], dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data[1], dtype=theano.config.floatX))
    else:
        shared_x = theano.shared(numpy.asarray(data[0][shared_index*shared_size:(shared_index+1)*shared_size], dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data[1][shared_index*shared_size:(shared_index+1)*shared_size], dtype=theano.config.floatX))
    return (shared_x, T.cast(shared_y, 'int32'))


def train_test_model(n_epochs,
                     train_model,
                     valid_model,
                     test_model,
                     n_train_batches,
                     n_valid_batches,
                     n_test_batches):
    """ Main loop to train, validate and test models
    :type n_epochs: int
    :type train_model: theano.function
    :type validate_model: theano.function
    :type test_model: theano.function
    :type n_train_batches: int
    :type n_valid_batches: int
    :type n_test_batches: int
    """
    # Train parameters
    patience = 10000
    patience_increase = 2    
    valid_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
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
            
            if (iter + 1) % validation_frequency == 0:
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



def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

    
