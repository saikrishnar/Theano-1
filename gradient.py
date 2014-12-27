import theano.tensor as T
from theano import function
from theano import pp

def grad(fn, args):
  gy = T.grad(fn, args)
  print pp(gy)
  return gy

if __name__ == "__main__":
  x = T.dscalar("x")
  y = 1 / (1 + T.exp(-x))
  g = grad(y, x)
  f = function([x], g)
  print f(4)
  print f(94.2)
