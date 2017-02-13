import numpy as np
from scipy import spatial as sp

class KNearestNeighbors():
    """
    Implementation of the K Nearest Neighbors algorithm
    """

    def __init__(self, train_x, train_y , k=1, type="classification"):
        """
        train_x: list
        train_y: list
        k : integer indicating the number of neareast neighbors
        type: string indicating a classification problem or regression problem
        """
        self.train_x = train_x
        self.train_y = train_y
        self.k = k
        self.type = type
        
        
    def predict(self, test_x):
        pass
        
    def score(self, test_x, test_y):
        pass