import numpy as np
from random import shuffle
from past.builtins import xrange

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
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    
#     from IPython.core.debugger import Tracer; Tracer()()
    
    scores -= np.max(scores) 
    p = np.exp(scores[y[i]]) / np.sum(np.exp(scores))
    loss += - np.log(p)
    
    
#     # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
#     log_c = np.max(f_i)
#     f_i -= log_c

#     # Compute loss (and add to it, divided later)
#     # L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
#     sum_i = 0.0
#     for f_i_j in f_i:
#       sum_i += np.exp(f_i_j)
#     loss += -f_i[y[i]] + np.log(sum_i)
    
#     for j in range(num_classes):
#       p = np.exp(f_i[j])/sum_i
#       dW[j, :] += (p-(j == y[i])) * X[:, i]

    
    # to calculate the gradient we need to differentiate the loss function with respect to the weights
    #from IPython.core.debugger import Tracer; Tracer()()
#     scores[y[i]] -= 1
#     dW[i] += scores

#     from IPython.core.debugger import Tracer; Tracer()()
    for j in xrange(scores.shape[0]):
#       from IPython.core.debugger import Tracer; Tracer()()
#       print('scores.sum(): %f' % scores.sum())
#       print('scores[j]: %f' % scores[j])
#       dW[j, :] += (p-(j == y[i])) * X[:, i]
#       from IPython.core.debugger import Tracer; Tracer()()
      p = np.exp(scores[j]) / np.sum(np.exp(scores))
      if j == y[i]:
        dW[:,j] += (p - 1) * X[i]
      else:
        dW[:,j] += p * X[i]
    
    # L_i = -np.log(np.exp(X[i].dot(W)[y[i]]))- np.log(np.sum(np.exp(X[i].dot(W))))
    
    # d/dW_i = - log( e^X_i.W[y_i] / Sum_j e^X_j.W )
    # d/dW_i = - log( e^X_i.W[y_i] ) + log( Sum_j e^X_j.W )
    # d/dW_i = log( Sum_j e^X_j.W ) - log( e^X_i.W[y_i] )
    # d/dW_i = log( Sum_j e^X_j.W ) - log( e^X_i.W[y_i] )
    
    # i = y_i
    # d/dW_i = log( Sum_j e^X_j.W ) - log( e^X_i.W[y_i] )
    # d/dW_i = X_i*e^X_j.W/( Sum_j e^X_j.W ) - X_i*e^X_i.W[y_i]/e^X_i.W[y_i]
    
    # i != y_i
    # d/dW_i = log( Sum_j e^X_j.W ) - log( e^X_i.W[y_i] )
    # d/dW_i = X_i*e^X_j.W/( Sum_j e^X_j.W ) 
    
    
    # d/dW_i = (Sum_j X_j)* ( Sum_j e^X_j.W )/( Sum_j e^X_j.W ) - X_i*e^X_i.W[y_i]/( e^X_i.W[y_i] )
    # d/dW_i = (Sum_j X_j) - X_y_i
    
    # so seems like we need derivative of the scores, not the loss? e^X_i.W[y_i] / Sum_j e^X_j.W ?
    # but for SVM we calculated derivative of the loss function, but for a specific training example and each output
    # depending on whether it was the correct one or not ...?
    
#     ## this is all from SVM
#     correct_class_score = scores[y[i]]
#     # simply count the xnumber of classes that didn’t meet the desired margin (and hence contributed to the loss function) and then the data vector xixi scaled by this number is the gradient
#     number_incorrect_classes = 0
#     for j in xrange(num_classes):
#       if j == y[i]:
#         continue

    
#       margin = scores[j] - correct_class_score + 1 # note delta = 1
#       if margin > 0:
#         number_incorrect_classes +=1
#         loss += margin
#         # so W is created via np.random.randn(3073, 10) * 0.0001 
#         # dW is then in the same shape with zeros so what is dW[:,j]? j in range 1 to 10
#         # so presumably dW[:,j] is a vector of 3073
        
#         dW[:,j] += X[i,:].transpose() # so this has to be cumulative for some reason
#     dW[:,y[i]] -= number_incorrect_classes * X[i,:].transpose() # as does this ...

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
#   loss += reg * np.sum(W * W)
#   dW += 2 * reg * W
    
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores) #scores 500 x 10
  correct_class_scores = scores[np.arange(num_train),y]
  
  correct_class_probs = np.exp(correct_class_scores) / np.sum(np.exp(scores), axis=1)
  loss = np.sum(-1 * np.log(correct_class_probs))
     
  all_probs = (np.exp(scores).T / np.sum(np.exp(scores), axis=1).T).T
  correct_mask = np.zeros_like(all_probs)
  correct_mask[np.arange(y.shape[0]),y] = 1
  dW = X.T.dot(all_probs - correct_mask)

  loss /= num_train
  loss += reg * np.sum(W * W)
    
  dW /= num_train
  dW += reg*W

   
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

