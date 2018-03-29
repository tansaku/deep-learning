import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        # dists[i,j] = X_train[j] - X[i]

        # X [[45, 76, ..., 34], [41, 12, ..., 123], ...]
        # X[i,:]
        # e.g. X[0,:] => [45, 76, ..., 34]
        # e.g. X_train[3,:] => [88, 36, ..., 32]
        # X[i,:] - self.X_train[j,:] ==> [-43, 40, ..., 2]

        dists[i,j] = (np.sum(np.square(X[i,:] - self.X_train[j,:])))**(1/2)
    
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################

    
    
      # X => [[45, 76, ..., 34], [41, 12, ..., 123], ...] ; num_test => number of test examples e.g. 100
      # self.X_train =  [[12, 3, ..., 34], [14, 12, ..., 123], ...] ; num_train => number of training examples, e.g. 1000
      # dists => [[0,0,0,0,0,...0],[0,0,0,0,0,...0], ..., [0,0,0,0,0,...0]] in dimensions num_test x num_train
    
      # i => 0, 1, 2 ... 100
      # dists[3,:] => [0,0,0,0,0,...0]
      # X[i,:] => a specific test example
      # self.X_train - X[i,:] => subtracts the test example from every training example
        
        dists[i,:] = np.sum(np.square(self.X_train - X[i,:]), axis = 1)**(1/2)
    
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

      # X => [[45, 76, ..., 34], [41, 12, ..., 123], ...] ; num_test => number of test examples e.g. 100
      # self.X_train =  [[12, 3, ..., 34], [14, 12, ..., 123], ...] ; num_train => number of training examples, e.g. 1000 x 3720
      # dists => [[0,0,0,0,0,...0],[0,0,0,0,0,...0], ..., [0,0,0,0,0,...0]] in dimensions num_test x num_train, e.g. 100 x 1000
    
    # (Xi,j - Yi,j)^2 where X train and Y test 
    # Xi,j^2 + Yi,j^2 -2 Xi,j * Yi,j where X train and Y test 
    
    X_train_norms = np.sum(self.X_train ** 2, axis=1) # Xi,j^2
#     print(X_train_norms.shape)
    X_norms = np.sum(X ** 2, axis=1, keepdims=True) # Yi,j^2
#     print(X_norms.shape)
    cross = -2.0 * X.dot(self.X_train.T) # -2 Xi,j * Yi,j
#     print(cross.shape)
    dists = np.sqrt(X_norms + cross + X_train_norms)  # X_train_norms => 1000 x 1000 matrix
    
#     dists = np.sum(np.square(self.X_train - X), axis = 1)**(1/2)

#     train_squares = np.diag(self.X_train.dot(self.X_train.transpose())) #an array of 5000
#     im = -2*X.dot(self.X_train.transpose()) + train_squares # 500x5000 matrix
#     dists = np.sqrt((im.transpose()+np.diag(X.dot(X.transpose()))).transpose())
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # test image i
      # dists[i,:] => [1.2, 4.4, 3.6, ... ]
      # np.argsort(dists[i,:]) => [0, 2, 1,... ]
      # np.argsort(dists[i,:])[0:k] => [0]
        
      closest_y = self.y_train[np.argsort(dists[i,:])[0:k]]  # [0,0,3,4,6]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = stats.mode(closest_y).mode
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

