import numpy as np
import scipy as sp

class KNearestNeighbors():
    """
    Implementation of the K Nearest Neighbors algorithm
    """

    def __init__(self, training_set, test_set, k=1, type="classification"):
        """
        training_set : numpy n x 3 or n x 2 matrix
        test_set: numpy n x 3 or n x 2 matrix
        k : integer indicating the number of neareast neighbors
        type: string indicating a classification problem or regression problem
        """
        self.training_set = training_set
        self.test_set = test_set
        self.k = k
        self.type = type
        
        
    def _distance_between_points(self):
        """
        Computes euclidean distance between points
        returns a floating point number
        """
        pass
        
    def fit(self):
        """
        Fits a KNN model to the training set
        returns a training error and test error
        """
        pass