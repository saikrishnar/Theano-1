from theano import tensor as T
from theano.ifelse import ifelse 
import theano, time, numpy

a = T.scalar("a")
b = T.scalar("b")

x = T.matrix("x")
y = T.matrix("y")

z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = theano.function([a, b, x, y], z_switch, mode=theano.Mode(linker="vm"))
f_lazy = theano.function(inputs=[a, b, x, y], outputs=z_lazy, mode=theano.Mode(linker="vm"))

val1 = 0.
val2 = 1.
big_mat1 = numpy.ones((10000, 1000))
big_mat2 = numpy.ones((10000, 1000))

n_times = 10

tic = time.clock()
for i in range(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print 'time spent evaluating both values %f sec' % (time.clock() - tic)

tic = time.clock()
for i in range(n_times):
    f_lazy(val1, val2, big_mat1, big_mat2)
print 'time spent evaluating one value %f sec' % (time.clock() - tic)

