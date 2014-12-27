import theano
import theano.tensor as T
from theano import function

def jacobian(fn, args):
  J, updates = theano.scan(lambda i, fn,args : T.grad(fn[i], args), sequences=T.arange(fn.shape[0]), 
                           non_sequences=[fn, args])
  f = function([args], J, updates=updates)
  return f

def jacobian_times_vector(fn, args, vector, weights):
  J = T.Rop(fn, weights, vector)
  f = function([weights, vector, fn], J)
  return f

if __name__ == "__main__":
  x = T.dvector("x")
  y = 1 / (1 + T.exp(-x))
  f = jacobian(y, x)
  print f([4, 4])
