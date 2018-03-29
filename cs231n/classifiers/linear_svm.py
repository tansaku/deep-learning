import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    # simply count the number of classes that didn’t meet the desired margin (and hence contributed to the loss function) and then the data vector xixi scaled by this number is the gradient
    number_incorrect_classes = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        number_incorrect_classes +=1
        loss += margin
        # so W is created via np.random.randn(3073, 10) * 0.0001 
        # dW is then in the same shape with zeros so what is dW[:,j]? j in range 1 to 10
        # so presumably dW[:,j] is a vector of 3073
        
        dW[:,j] += X[i,:].transpose() # so this has to be cumulative for some reason
    dW[:,y[i]] -= number_incorrect_classes * X[i,:].transpose() # as does this ...

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # W is our current Weights (3072, 10)?
  # X is our training set (num_train, 3072)?
  # scores --> (num_train, 10) e.g. (for 3 classes) [[.2, .5. -.4],[.3, .6. -.1],[.6, .2. .4]]
  # y is the correct classes for our training set (num_train) e.g. [1,0,2]
  # reg is regularization parameter
    
  num_train = X.shape[0]
  delta = 1.0
  
  scores = X.dot(W)  # gives us scores for for all training items for each category [num_train, 10]
  correct_class_scores = scores[np.arange(num_train),y]
  margins = np.maximum(0, scores.T - correct_class_scores + delta) # => [10,500]
    
  loss = np.sum(margins)

  loss /= num_train
  loss -= delta # remove contribution from correct classes 

  # add regularization
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # dW [3073,10]  
  # so we want to take maybe a mask of the margins(?), so any non-zero margin that is not
  # the correct output category implies that the corresponding training example should be
  # added to the dW, while those corresponding to the correct output category should have the 
  # corresponding training example deleted from the dW multipled by the number of non-zero margins

  #                                         x         => y (x.W = scores)-> (margins) -> mask (1,2,1)       
  # X => y (X.W) is training examples like [12,34,52] => 1 (.34,-.9,0.1) -> (1.24,1,1); [1,2,7] => 2 (.1,.2,3.5) -> (-2.4,-2.3,1)
  #                                                                                                              -> mask (0,0,1)
  # to mask is a manipulation of margins where 
  #
  #  a) all positive numbers are set to one
  #  b) all negative numbers are set to zero
  #  c) all correct index items are set equal to the number of positives in their row (other than themselves)
  #  d) all rows with a single one in the correct index could be set to zero
  # dW add X if corresponding margin is positive
  # sum over the training examples but exclude those with corresponding zero margins, and modify the correct 
  # category by -#incorrect --> so need some way of converting margins to those numbers, e.g. 1, 0 and -#incorrect
  # and then multiple by X and then sum over X????
    
  #from IPython.core.debugger import Tracer; Tracer()()   
  # set all the correct indicies to zero  
  margins[y, np.arange(num_train)] = 0 
  # adjust all to true and false (for positive and negative)  
  margins_boolean = margins>0 # this makes margins_boolean of type bool
  # work out the numbers of incorrect for each training example
  margins_sum = np.sum(margins_boolean, axis=0)
  # set the correct index for each training example to the number of incorrect
  margins_int = margins_boolean.astype(int)
  margins_int[y, np.arange(num_train)] = -margins_sum 
    
  #margins_negs_all_zero = margins*(margins>0)   
  #margins_pos_all_one = margins*(margins<0)   
  #from IPython.core.debugger import Tracer; Tracer()() 
  dW = margins_int.dot(X).T # 
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
