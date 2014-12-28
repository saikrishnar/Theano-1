import theano
import theano.tensor as T
import numpy


def tanh_scan():
    # Compute tanh(x(t).dot(W) + b) elementwise

    X = T.matrix("X")
    W = T.matrix("W")
    b_sym = T.vector("b_sym")

    result, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
    compute_elementwise =  theano.function(inputs=[X, W, b_sym], outputs=[result])

    # test values
    x = numpy.eye(2, dtype=theano.config.floatX)
    w = numpy.ones((2, 2), dtype=theano.config.floatX)
    b = numpy.ones((2), dtype=theano.config.floatX)
    b[1] = 2

    print "Using theano.scan"
    print compute_elementwise(x, w, b)

    print "Using numpy"
    print numpy.tanh(x.dot(w) + b)


def multi_tanh_scan():
    # Compute sequence x(t) = tanh(x(t - 1).dot(W) + y(t).dot(U) + p(T - t).dot(V))
    X = T.vector("X")
    W = T.matrix("W")
    Y = T.matrix("Y")
    U = T.matrix("U")
    P = T.matrix("P")
    V = T.matrix("V")

    result, updates = theano.scan(lambda x_timeshift, y, p: T.tanh(T.dot(x_timeshift, W) + T.dot(y, U) + T.dot(p, V)),
                                  sequences=[Y, P[::-1]], outputs_info=[X])

    compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=[result])

    # test values
    x = numpy.zeros((2), dtype=theano.config.floatX)
    x[1] = 1
    w = numpy.ones((2, 2), dtype=theano.config.floatX)
    y = numpy.ones((5, 2), dtype=theano.config.floatX)
    y[0, :] = -3
    u = numpy.ones((2, 2), dtype=theano.config.floatX)
    p = numpy.ones((5, 2), dtype=theano.config.floatX)
    p[0, :] = 3
    v = numpy.ones((2, 2), dtype=theano.config.floatX)
    
    print compute_seq(x, w, y, u, p, v)[0]

    # comparison with numpy
    x_res = numpy.zeros((5, 2), dtype=theano.config.floatX)
    x_res[0] = numpy.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
    for i in range(1, 5):
        x_res[i] = numpy.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4-i].dot(v))
    print x_res


def norm_rows():
    # Find norm of rows of Matrix X
    X = T.matrix("X")
    result, updates = theano.scan(lambda row: T.sqrt((row ** 2).sum()), sequences=[X.T])
    norms = theano.function(inputs=[X], outputs=result)

    # Test it
    inp = numpy.random.rand(16).reshape(4, 4)
    
    print "Using theano"
    print norms(inp)

    print "Using numpy"
    print numpy.sqrt((inp ** 2).sum(0))


def trace():
    X = T.matrix("X")
    results, updates = theano.scan(lambda i, j, running_sum: running_sum + X[i, j],
                                   sequences=[T.arange(X.shape[0]), T.arange(X.shape[1])],
                                   outputs_info=numpy.asarray(0.))
    result = results[-1]
    traces = theano.function(inputs=[X], outputs=result)
    
    # Test it
    x = numpy.random.rand(16).reshape(4, 4)

    print "Using theano"
    print traces(x)
    
    print "Using numpy"
    print numpy.diagonal(x).sum()


if __name__ == "__main__":
    trace()
