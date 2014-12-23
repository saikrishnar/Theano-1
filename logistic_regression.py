import numpy
import theano
import theano.tensor as T


def logistic_regression(X, y, training_steps=10000, step_size=0.01):
  N, feats = X.shape

  # Declare Theano symbolic variables
  input_X = T.matrix("X")
  input_y = T.vector("y")
  w = theano.shared(numpy.random.randn(feats), name="w")
  b = theano.shared(0., name="b")

  # Compute sigmoid and prediction
  g = T.dot(input_X, w) + b
  sigmoid = 1 / (1 + T.exp(-g))
  prediction = sigmoid > 0.5

  # Compute mle and Gradient descent sxpression
  mle = -input_y * T.log(sigmoid) - (1-input_y) * T.log(1-sigmoid)
  cost = mle.mean() + step_size * (w ** 2).sum() 
  gw, gb = T.grad(cost, [w, b])   

  # Compile
  train = theano.function(
          inputs=[input_X, input_y],
          outputs=[prediction, mle],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
  predict = theano.function(inputs=[input_X], outputs=prediction)

  # Train
  for i in range(training_steps):
    pred, err = train(X, y)

  print "Final model:"
  predicted_values = predict(X)
  print "target values for D:", y
  print "prediction on D:", predicted_values
  print "Accuracy:", 1. * (predicted_values == y).sum() / len(y) 

if __name__ == "__main__":
  N = 400
  feats = 784
  X = numpy.random.randn(N, feats)
  y = numpy.random.randint(size=N, low=0, high=2)
  logistic_regression(X, y)
