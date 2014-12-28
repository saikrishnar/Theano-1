import theano
import theano.tensor as T
import numpy

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
