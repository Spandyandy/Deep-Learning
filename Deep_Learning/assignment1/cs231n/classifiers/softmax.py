import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  D = W.shape[0]



  for i in range(num_train):
    scores = X[i,:].dot(W)
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    loss += -np.log(probabilities[y[i]])
    for k in range(num_classes):
      dW[:, k] += X.T[:, i]*(probabilities[k]-(y[i] == k))
  loss /= num_train
  dW /= num_train

  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
#   (D, C) = W.shape
#   N = X.shape[0]
  
#   scores = X.dot(W)
#   scores -= np.max(scores)

#   y_matrix = np.zeros(shape(N,C))
#   y_matrix[range(N), y] = 1
#   result = np.log(np.exp(scores)) - np.sum(np.multiply(scores, y_matrix),axis=0)
#   loss = np.sum(result)
  N = X.shape[0]
  C = W.shape[1]
  f = X.dot(W)
  f -= np.matrix(np.max(f, axis=1)).T
  term1 = -f[np.arange(N), y]
  sum_j = np.sum(np.exp(f), axis=1)
  term2 = np.log10(sum_j)
  loss = term1 + term2
  loss /= N 
  loss += 0.5 * reg * np.sum(W * W)
  coef = np.exp(f) / np.matrix(sum_j).T
  coef[np.arange(N),y] -= 1
  dW = X.T.dot(coef)
  dW /= N
  dW += reg*W


  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

